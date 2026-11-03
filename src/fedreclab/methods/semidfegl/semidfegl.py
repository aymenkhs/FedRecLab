"""SemiDFEGL server: clustering, fake common items and sliced payloads.

Source: Qu et al., "Semi-decentralized Federated Ego Graph Learning for
Recommendation", WWW 2023.

Simulation shortcuts
--------------------
- The server copies each cluster member's stored ego vector into the
  payload in place of a device-to-device link. Interaction lists are not
  part of that payload.
- Augmented degree is computed on the server from the training split and
  the current assignment; a deployed client would send the integer with
  its ego vector.
- Item rows for this client's training ids, fake bridges and BPR
  negatives are sliced by the simulator.
- Neighbour messages are one round old (lagged). Self is current.

Deviations from Qu et al.
-------------------------
- Neighbour ego vectors do not advance through LightGCN layers inside a
  round. A same-round multi-hop relay would require overriding
  ``run_round``.
- ``n_fake`` means items per cluster (Figure 5 / section 4.1.4), not
  clusters per item.
- The loss and the ranking score both dot the layer-combined user
  vector with layer-0 item rows. Held-out items are not on the ego
  graph, so they have no layer-averaged item embedding. Layer-averaged
  item states stay inside message passing. Clients upload the stepped
  layer-0 item rows, and the next round propagates from those rows.
  The ego upload is the layer-combined user vector.
- Local item updates use SGD; the user embedding uses Adam with
  persisted moments. Item Adam moments cannot live on the device, and
  one fresh Adam step per round would be ≈ signSGD on items.
- Privacy (pseudo embeddings, L1 clip, Laplace) is off by default; the
  paper does not publish clip or noise scales.
- LightGCN only; GCN / GAT / GraphSAGE from section 4.6 are not
  implemented.
- Recommend ``FullClientSelection``. Other selectors are allowed; the
  bridge still uses every cluster member's stored ego, so non-selected
  vectors stay stale.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from fedreclab.federated.base_server import BaseFederatedServer
from fedreclab.federated.client_selection import FullClientSelection
from fedreclab.federated.utils import ParamIO
from fedreclab.methods.semidfegl.semidfegl_model import (
    fuzzy_cmeans,
    xavier_uniform_table,
)

logger = logging.getLogger(__name__)


def default_param_ios() -> dict[str, ParamIO]:
    """Return the SemiDFEGL ParamIO map.

    Returns
    -------
    dict of str to ParamIO
        ``item_embeddings`` and ``ego_embeddings``, both server-resident
        row-layout parameter groups.

    Examples
    --------
    >>> ios = default_param_ios()
    >>> sorted(ios)
    ['ego_embeddings', 'item_embeddings']
    >>> ios["item_embeddings"].layout
    'rows'
    """
    return {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="params",
            apply_to="server",
            layout="rows",
        ),
        "ego_embeddings": ParamIO(
            residency="server",
            exchange="params",
            apply_to="server",
            layout="rows",
        ),
    }


class SemiDFEGL(BaseFederatedServer):
    """Semi-decentralized federated ego-graph learning (Qu et al., 2023).

    Parameters
    ----------
    embedding_dim : int, optional
        Width of user and item embeddings. Default 64.
    num_groups : int, optional
        Fuzzy c-means clusters. Default 100 (MovieLens-1M setting).
    n_fake : int, optional
        Fake common items per cluster. Default 1.
    num_layers : int, optional
        LightGCN depth ``K``. Default 4. Used only to validate against
        the client; propagation runs on the client.
    fuzziness : float, optional
        Fuzzy c-means exponent. Default 2.0.
    cmeans_max_iter : int, optional
        Fuzzy c-means iteration cap. Default 100.
    cmeans_tol : float, optional
        Fuzzy c-means membership tolerance. Default ``1e-5``.
    bpr_negatives : int, optional
        Uniform BPR negatives sampled once per client per round and
        placed in the payload. Default 1.
    **kwargs
        Forwarded to :class:`BaseFederatedServer`.

    Raises
    ------
    ValueError
        If a hyper-parameter is out of range, a client has zero or
        several users, or the client embedding width disagrees.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> server = SemiDFEGL(
    ...     interaction_dataset=dataset,
    ...     client_algorithm=SemiDFEGLClientAlgorithm(),
    ...     num_rounds=10,
    ...     client_selection_strategy=FullClientSelection(dataset.num_clients),
    ...     aggregation_strategies=FedAvg(num_clients=dataset.num_clients),
    ...     seed=0,
    ... )
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_groups: int = 100,
        n_fake: int = 1,
        num_layers: int = 4,
        fuzziness: float = 2.0,
        cmeans_max_iter: int = 100,
        cmeans_tol: float = 1e-5,
        bpr_negatives: int = 1,
        **kwargs: Any,
    ) -> None:
        for name, value in (
            ("embedding_dim", embedding_dim),
            ("num_groups", num_groups),
            ("n_fake", n_fake),
            ("bpr_negatives", bpr_negatives),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 0:
            raise ValueError(f"num_layers must be a non-negative int, got {num_layers!r}")
        if float(fuzziness) <= 1.0:
            raise ValueError(f"fuzziness must be > 1, got {fuzziness}")
        if (
            not isinstance(cmeans_max_iter, int)
            or isinstance(cmeans_max_iter, bool)
            or cmeans_max_iter < 1
        ):
            raise ValueError(
                f"cmeans_max_iter must be a positive int, got {cmeans_max_iter!r}"
            )
        if float(cmeans_tol) < 0.0:
            raise ValueError(f"cmeans_tol must be >= 0, got {cmeans_tol}")

        self.embedding_dim = int(embedding_dim)
        self.num_groups = int(num_groups)
        self.n_fake = int(n_fake)
        self.num_layers = int(num_layers)
        self.fuzziness = float(fuzziness)
        self.cmeans_max_iter = int(cmeans_max_iter)
        self.cmeans_tol = float(cmeans_tol)
        self.bpr_negatives = int(bpr_negatives)

        self._user_to_client: dict[int, int] = {}
        self._client_to_user: dict[int, int] = {}
        self._user_train_items: dict[int, np.ndarray] = {}
        self._user_cluster: dict[int, int] = {}
        self._cluster_users: dict[int, list[int]] = {}
        self._cluster_fake_items: dict[int, np.ndarray] = {}
        self._user_degree: dict[int, int] = {}
        self._selector_warned = False

        client = kwargs.get("client_algorithm")
        if client is not None and getattr(client, "init_seed", None) is None:
            client.init_seed = kwargs.get("seed")

        super().__init__(param_ios=default_param_ios(), **kwargs)

        client_dim = getattr(self.client_algorithm, "embedding_dim", None)
        if client_dim is not None and int(client_dim) != self.embedding_dim:
            raise ValueError(
                f"SemiDFEGL embedding_dim={self.embedding_dim} does not match "
                f"client embedding_dim={client_dim}"
            )
        client_layers = getattr(self.client_algorithm, "num_layers", None)
        if client_layers is not None and int(client_layers) != self.num_layers:
            raise ValueError(
                f"SemiDFEGL num_layers={self.num_layers} does not match "
                f"client num_layers={client_layers}"
            )
        self._index_users_and_train_items()

    def _index_users_and_train_items(self) -> None:
        """Map clients to users and cache training item sets.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any client has zero or more than one user.
        """
        partition = self.interaction_dataset.client_partition
        for client_id in range(self.num_clients):
            users = partition.client_ids_2_user_ids[client_id]
            if len(users) != 1:
                raise ValueError(
                    "SemiDFEGL requires exactly one user per client; "
                    f"client {client_id} has {len(users)} users {users}"
                )
            user_id = int(users[0])
            self._client_to_user[client_id] = user_id
            self._user_to_client[user_id] = client_id

        train = self.interaction_dataset.train_interactions
        for user_id, group in train.groupby("user_id"):
            items = np.unique(np.asarray(group["item_id"], dtype=np.int64))
            self._user_train_items[int(user_id)] = items

        for user_id in self._user_to_client:
            if user_id not in self._user_train_items:
                self._user_train_items[user_id] = np.zeros((0,), dtype=np.int64)

    def _initialize_server_parameters(self) -> dict[str, Any]:
        """Create Xavier item and ego tables.

        Returns
        -------
        dict
            ``item_embeddings`` of shape ``(num_items, d)`` and
            ``ego_embeddings`` of shape ``(num_users, d)``.
        """
        if self.seed is None:
            raise ValueError(
                "SemiDFEGL needs seed= for Xavier tables and BPR negatives"
            )
        gen = torch.Generator()
        gen.manual_seed(int(self.seed))
        items = xavier_uniform_table(
            self.interaction_dataset.num_items,
            self.embedding_dim,
            generator=gen,
        )
        # Ego table starts as Xavier user embeddings (paper init phase).
        egos = xavier_uniform_table(
            self.interaction_dataset.num_users,
            self.embedding_dim,
            generator=gen,
        )
        return {
            "item_embeddings": items.detach().cpu().numpy().astype(np.float32),
            "ego_embeddings": egos.detach().cpu().numpy().astype(np.float32),
        }

    def _before_training(self) -> None:
        """Warn once when the selector is not full participation.

        Returns
        -------
        None
        """
        if self._selector_warned:
            return
        if isinstance(self.client_selection_module, FullClientSelection):
            return
        logger.info(
            "SemiDFEGL recommends FullClientSelection; got %s. "
            "The device-to-device bridge still uses every cluster member's "
            "stored ego vector, while only the selected cohort uploads, so "
            "non-selected vectors stay stale.",
            type(self.client_selection_module).__name__,
        )
        self._selector_warned = True

    def _on_round_start_server(self) -> None:
        """Cluster users and items; pick fake bridges per cluster.

        Returns
        -------
        None
        """
        self._run_clustering()

    def _cmeans_rng(self) -> np.random.Generator:
        """Return the RNG for this round's fuzzy c-means.

        Returns
        -------
        numpy.random.Generator
            Seeded from ``(seed, current_round)``.

        Raises
        ------
        ValueError
            If ``seed`` is unset.
        """
        if self.seed is None:
            raise ValueError("SemiDFEGL needs seed= for fuzzy c-means")
        round_id = 0 if self.current_round is None else int(self.current_round)
        return np.random.default_rng([int(self.seed), round_id])

    def _run_clustering(self) -> None:
        """Update cluster assignment and fake common items.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If clustering fails or a cluster needs more fake items than
            the catalogue provides.
        """
        egos = np.asarray(self.global_parameters["ego_embeddings"], dtype=np.float64)
        items = np.asarray(self.global_parameters["item_embeddings"], dtype=np.float64)
        num_users = int(egos.shape[0])
        num_items = int(items.shape[0])
        points = np.concatenate([egos, items], axis=0)
        n_clusters = min(self.num_groups, points.shape[0])
        membership, _centres = fuzzy_cmeans(
            points,
            num_clusters=n_clusters,
            fuzziness=self.fuzziness,
            max_iter=self.cmeans_max_iter,
            tol=self.cmeans_tol,
            rng=self._cmeans_rng(),
        )

        user_membership = membership[:num_users]
        item_membership = membership[num_users:]
        user_labels = np.argmax(user_membership, axis=1).astype(np.int64)

        self._user_cluster = {}
        self._cluster_users = {}
        for user_id in range(num_users):
            if user_id not in self._user_to_client:
                continue
            label = int(user_labels[user_id])
            self._user_cluster[user_id] = label
            self._cluster_users.setdefault(label, []).append(user_id)

        # Drop empty (no-user) clusters; keep only clusters with users.
        self._cluster_fake_items = {}
        for label, users in list(self._cluster_users.items()):
            if len(users) == 0:
                del self._cluster_users[label]
                continue
            scores = item_membership[:, label]
            # Stable tie-break: higher membership first, then smaller item id.
            order = np.lexsort((np.arange(num_items), -scores))
            take = min(self.n_fake, num_items)
            if take < self.n_fake:
                raise ValueError(
                    f"Need n_fake={self.n_fake} items but catalogue has {num_items}"
                )
            fake = order[:take].astype(np.int64)
            self._cluster_fake_items[label] = fake

        self._user_degree = {}
        for user_id, label in self._user_cluster.items():
            train_items = self._user_train_items.get(user_id, np.zeros(0, dtype=np.int64))
            fake = self._cluster_fake_items[label]
            train_set = set(int(i) for i in train_items)
            extra = sum(1 for i in fake if int(i) not in train_set)
            degree = int(train_items.size) + int(extra)
            if degree < 1:
                raise ValueError(
                    f"user {user_id} has augmented degree 0; need at least one "
                    "training or fake neighbour"
                )
            self._user_degree[user_id] = degree

    def _payload_rng(self, client_id: int) -> np.random.Generator:
        """Return the RNG for this client's BPR negative draw.

        Parameters
        ----------
        client_id : int
            Client the payload is for.

        Returns
        -------
        numpy.random.Generator
        """
        if self.seed is None:
            raise ValueError("SemiDFEGL needs seed= for BPR negatives")
        round_id = 0 if self.current_round is None else int(self.current_round)
        return np.random.default_rng([int(self.seed), int(client_id), round_id, 17])

    def _sample_bpr_negatives(
        self,
        user_id: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw BPR negatives outside the user's training items.

        Parameters
        ----------
        user_id : int
            Remapped user id.
        rng : numpy.random.Generator
            Source of the draw.

        Returns
        -------
        ndarray of int64
            Shape ``(bpr_negatives,)``.

        Raises
        ------
        ValueError
            If too few catalogue ids remain.
        """
        num_items = self.interaction_dataset.num_items
        excluded = set(int(i) for i in self._user_train_items.get(user_id, []))
        # Also exclude fake items that will be used as negatives so the
        # dedicated BPR sample is distinct when possible.
        label = self._user_cluster[user_id]
        fake = self._cluster_fake_items[label]
        train_set = excluded
        fake_neg = {int(i) for i in fake if int(i) not in train_set}
        pool = np.array(
            [i for i in range(num_items) if i not in excluded],
            dtype=np.int64,
        )
        if pool.size < self.bpr_negatives:
            raise ValueError(
                f"Need {self.bpr_negatives} BPR negatives for user {user_id} but "
                f"only {pool.size} catalogue ids remain outside training items"
            )
        # Prefer ids that are not already fake negatives, then fill.
        preferred = np.array(
            [i for i in pool if int(i) not in fake_neg],
            dtype=np.int64,
        )
        chosen: list[int] = []
        if preferred.size > 0:
            take = min(self.bpr_negatives, int(preferred.size))
            picks = rng.choice(preferred, size=take, replace=False)
            chosen.extend(int(x) for x in np.atleast_1d(picks))
        if len(chosen) < self.bpr_negatives:
            remaining = np.array(
                [i for i in pool if int(i) not in set(chosen)],
                dtype=np.int64,
            )
            need = self.bpr_negatives - len(chosen)
            picks = rng.choice(remaining, size=need, replace=False)
            chosen.extend(int(x) for x in np.atleast_1d(picks))
        return np.asarray(chosen, dtype=np.int64)

    def _build_server_payload(self, client_id: int | None = None) -> dict[str, Any]:
        """Return the sliced payload for one client.

        Parameters
        ----------
        client_id : int or None, optional
            Client the payload is for. ``None`` is rejected: SemiDFEGL
            payloads are always client-specific.

        Returns
        -------
        dict
            Keys ``item_ids``, ``item_embeddings``, ``fake_item_ids``,
            ``bpr_negative_ids``, ``cluster_user_ids``,
            ``cluster_ego_embeddings``, ``cluster_degrees``,
            ``cluster_size``.

        Raises
        ------
        ValueError
            If ``client_id`` is ``None`` or clustering has not run.
        """
        if client_id is None:
            raise ValueError(
                "SemiDFEGL._build_server_payload requires a client_id; "
                "payloads are always client-specific"
            )
        if not self._user_cluster:
            raise ValueError(
                "SemiDFEGL clustering has not run; _on_round_start_server "
                "must execute before building payloads"
            )

        user_id = self._client_to_user[int(client_id)]
        label = self._user_cluster[user_id]
        train_items = self._user_train_items[user_id]
        fake_items = self._cluster_fake_items[label]
        bpr_negs = self._sample_bpr_negatives(user_id, self._payload_rng(int(client_id)))

        item_ids = np.unique(
            np.concatenate(
                [
                    np.asarray(train_items, dtype=np.int64),
                    np.asarray(fake_items, dtype=np.int64),
                    np.asarray(bpr_negs, dtype=np.int64),
                ]
            )
        )
        item_table = np.asarray(
            self.global_parameters["item_embeddings"], dtype=np.float32
        )
        ego_table = np.asarray(
            self.global_parameters["ego_embeddings"], dtype=np.float32
        )

        cluster_users = list(self._cluster_users[label])
        cluster_egos = ego_table[np.asarray(cluster_users, dtype=np.int64)]
        cluster_degrees = np.asarray(
            [self._user_degree[u] for u in cluster_users],
            dtype=np.float32,
        )

        return {
            "item_ids": item_ids.astype(np.int64, copy=False),
            "item_embeddings": item_table[item_ids].astype(np.float32, copy=False),
            "fake_item_ids": np.asarray(fake_items, dtype=np.int64),
            "bpr_negative_ids": np.asarray(bpr_negs, dtype=np.int64),
            "train_item_ids": np.asarray(train_items, dtype=np.int64),
            "cluster_user_ids": np.asarray(cluster_users, dtype=np.int64),
            "cluster_ego_embeddings": cluster_egos.astype(np.float32, copy=False),
            "cluster_degrees": cluster_degrees,
            "cluster_size": int(len(cluster_users)),
            "user_degree": int(self._user_degree[user_id]),
        }
