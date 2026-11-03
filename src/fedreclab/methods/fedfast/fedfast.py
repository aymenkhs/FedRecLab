"""Federated FedFast server (Muhammad et al., KDD 2020).

Faithful aggregation for item embeddings uses ``ChangeMagnitudeAvg`` and
``FedAvg`` on the linear head. Subordinate user-embedding propagation and
reclustering are implemented by the method-private ``_ActvAgg`` helper and
materialised in ``_on_round_end_server``. Enable that path with
``actv_agg=True`` (default) without importing ``_actvagg``.

Client sampling uses ``ActvSampClientSelection`` (Algorithm 2): ``m/p``
clients from each cluster. The paper's prose also mentions round-robin
sampling; that variant is not implemented.

Simulation shortcuts:
  The server reads all clients' user embeddings for k-means and writes
  non-selected (subordinate) ``ClientState`` user embeddings after each
  round. Plaintext row-layout item uploads reveal which items a delegate
  trained on. Initial clusters ``G0`` are built from centralised
  train-profile statistics (mean rating, interaction count, item-hist
  entropy).

Deviations from the paper:
  User embeddings are client-resident (Option A), not a server-side full
  table. Undersized k-means clusters during ActvSAMP are filled from the
  remaining population (documented on the selector). NeuMF / NCF is not
  implemented; only GMF. Evaluation uses the library ranking protocol.

Defaults: ``embedding_dim=10``, ``num_clusters=20`` (paper experimental
settings). Pair with ``ActvSampClientSelection`` and choose ``m`` divisible
by ``num_clusters``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from fedreclab.federated.aggregation import BaseAggregation
from fedreclab.federated.base_server import BaseFederatedServer
from fedreclab.federated.client_selection import ActvSampClientSelection
from fedreclab.federated.utils import ParamIO
from fedreclab.methods.fedfast._actvagg import (
    _ActvAgg,
    _FedFastClusterState,
    _kmeans_labels,
)
from fedreclab.methods.fedfast.fedfast_model import GeneralMatrixFactorization

logger = logging.getLogger(__name__)


def default_param_ios() -> dict[str, ParamIO]:
    """Return the FedFast ParamIO map.

    User embeddings stay in ``ClientState`` and are not exchanged.
    Subordinate writes happen on round end via ``_ActvAgg``.

    Returns
    -------
    dict of str to ParamIO
        ``item_embeddings`` (rows) and ``output_weights`` (dense).

    Examples
    --------
    >>> pio = default_param_ios()
    >>> pio["item_embeddings"].layout
    'rows'
    """
    return {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="params",
            apply_to="server",
            layout="rows",
        ),
        "output_weights": ParamIO(
            residency="server",
            exchange="params",
            apply_to="server",
        ),
    }


class FedFast(BaseFederatedServer):
    """FedFast GMF server with ActvSAMP / ActvAGG wiring.

    Parameters
    ----------
    embedding_dim : int, optional
        GMF embedding width. Default is 10.
    num_clusters : int, optional
        Number of ActvSAMP / ActvAGG clusters ``p``. Default is 20.
    actv_agg : bool, optional
        If True (default) and ``aggregation_strategies`` is omitted,
        install ``ChangeMagnitudeAvg`` on items and ``FedAvg`` on the
        output head, and run subordinate user updates on round end. If
        False, the caller must pass ``aggregation_strategies`` and
        subordinate propagation is skipped.
    change_eps : float, optional
        Threshold for ``ChangeMagnitudeAvg``. Default is 0.0.
    aggregation_strategies : BaseAggregation or dict or None, optional
        Explicit aggregators override the ``actv_agg`` defaults when
        provided.

    Raises
    ------
    ValueError
        If ``actv_agg=False`` and ``aggregation_strategies`` is missing, or
        if ``num_clusters`` is invalid.

    Examples
    --------
    >>> server = FedFast(  # doctest: +SKIP
    ...     embedding_dim=8,
    ...     num_clusters=2,
    ...     interaction_dataset=data,
    ...     client_algorithm=FedFastClientAlgorithm(embedding_dim=8),
    ...     client_selection_strategy=ActvSampClientSelection(K, 2),
    ...     num_rounds=10,
    ... )
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        num_clusters: int = 20,
        actv_agg: bool = True,
        change_eps: float = 0.0,
        **kwargs: Any,
    ):
        self.embedding_dim = int(embedding_dim)
        self.num_clusters = int(num_clusters)
        self.actv_agg = bool(actv_agg)
        self.change_eps = float(change_eps)

        if self.num_clusters < 1:
            raise ValueError(
                f"num_clusters must be >= 1, got {self.num_clusters}"
            )

        dataset = kwargs.get("interaction_dataset")
        if dataset is None:
            raise TypeError("FedFast requires interaction_dataset=")

        aggregation_strategies = kwargs.pop("aggregation_strategies", None)
        if aggregation_strategies is None:
            if not self.actv_agg:
                raise ValueError(
                    "actv_agg=False requires explicit aggregation_strategies"
                )
            aggregation_strategies = _ActvAgg.default_aggregation_strategies(
                num_clients=dataset.num_clients,
                change_eps=self.change_eps,
            )
        elif self.actv_agg and isinstance(aggregation_strategies, BaseAggregation):
            # Single aggregator applied to every exchanged group: keep it,
            # but still run subordinate user logic when actv_agg=True.
            pass

        client = kwargs.get("client_algorithm")
        if client is not None:
            client_dim = getattr(client, "embedding_dim", None)
            if client_dim is not None and int(client_dim) != self.embedding_dim:
                raise ValueError(
                    f"FedFast embedding_dim={self.embedding_dim} does not match "
                    f"client embedding_dim={client_dim}"
                )

        super().__init__(
            param_ios=default_param_ios(),
            aggregation_strategies=aggregation_strategies,
            **kwargs,
        )

        self._cluster_state = _FedFastClusterState(
            num_clusters=self.num_clusters,
            seed=self.seed,
        )
        self._actv_agg_helper = _ActvAgg(
            self._cluster_state, change_eps=self.change_eps
        )

    def _initialize_server_parameters(self):
        """Create initial item table and output head.

        Returns
        -------
        dict
            ParamIO groups from ``GeneralMatrixFactorization``.
        """
        model = GeneralMatrixFactorization(
            num_items=self.interaction_dataset.num_items,
            embedding_dim=self.embedding_dim,
        )
        if self.device is not None:
            model = model.to(self.device)
        return model.get_server_state_groups()

    def _before_training(self) -> None:
        """Build ``G0`` from train-profile stats and install ActvSAMP groups.

        Returns
        -------
        None
        """
        features = self._profile_feature_matrix()
        labels = _kmeans_labels(
            features,
            num_clusters=min(self.num_clusters, features.shape[0]),
            seed=self.seed,
        )
        # If K < requested p, _kmeans_labels already capped via min above.
        p_eff = int(labels.max()) + 1
        groups: list[list[int]] = [[] for _ in range(p_eff)]
        for cid, lab in enumerate(labels):
            groups[int(lab)].append(int(cid))
        groups = [g for g in groups if g]
        self._cluster_state.num_clusters = len(groups)
        self._cluster_state.groups = groups
        self._sync_selector_groups(groups)

    def _on_round_start_server(self) -> None:
        """Snapshot all user embeddings before delegates train.

        Returns
        -------
        None
        """
        if not self.actv_agg:
            return
        emb = self._collect_client_mean_embeddings()
        self._actv_agg_helper.snapshot_user_embeddings(emb)

    def _on_round_end_server(self) -> None:
        """Apply subordinate embeddings and refresh ActvSAMP groups.

        Returns
        -------
        None
        """
        if not self.actv_agg:
            return
        if self.selected_client_ids is None:
            return
        emb = self._collect_client_mean_embeddings()
        pending, groups = self._actv_agg_helper.compute_subordinate_updates(
            emb,
            selected_client_ids=self.selected_client_ids,
            current_round=int(self.current_round or 0),
        )
        for cid, vec in pending.items():
            self._write_client_user_embedding(int(cid), vec)
        self._sync_selector_groups(groups)

    def _sync_selector_groups(self, groups: list[list[int]]) -> None:
        """Call ``set_groups`` when the selector supports ActvSAMP.

        Parameters
        ----------
        groups : list of list of int
            New partition ``G``.

        Returns
        -------
        None
        """
        selector = self.client_selection_module
        if isinstance(selector, ActvSampClientSelection):
            selector.set_groups(groups)
            return
        logger.warning(
            "FedFast cluster groups were updated but the client selection "
            "strategy %s has no set_groups; ActvSAMP will not use them.",
            type(selector).__name__,
        )

    def _collect_client_mean_embeddings(self) -> dict[int, np.ndarray]:
        """Return one mean user-embedding vector per client.

        Returns
        -------
        dict of int to ndarray
            Shape ``(embedding_dim,)`` per client.
        """
        out: dict[int, np.ndarray] = {}
        for cid, state in self.client_states.items():
            table = np.asarray(
                state.local_parameters["user_embeddings"], dtype=np.float64
            )
            if table.ndim == 1:
                vec = table
            else:
                vec = table.mean(axis=0)
            out[int(cid)] = vec.reshape(-1)
        return out

    def _write_client_user_embedding(
        self, client_id: int, mean_vec: np.ndarray
    ) -> None:
        """Overwrite a client's user table with a broadcast mean vector.

        Multi-user clients receive the same vector on every local row
        (subordinate propagation transfers a single embedding delta).

        Parameters
        ----------
        client_id : int
            Subordinate client.
        mean_vec : ndarray
            New embedding, shape ``(embedding_dim,)``.

        Returns
        -------
        None
        """
        state = self.client_states[client_id]
        old = state.local_parameters["user_embeddings"]
        vec = np.asarray(mean_vec, dtype=np.float64).reshape(-1)
        if isinstance(old, torch.Tensor):
            if old.ndim == 1:
                new = torch.as_tensor(vec, dtype=old.dtype)
            else:
                new = (
                    torch.as_tensor(vec, dtype=old.dtype)
                    .unsqueeze(0)
                    .repeat(old.shape[0], 1)
                )
            state.local_parameters["user_embeddings"] = new
            return
        old_np = np.asarray(old)
        if old_np.ndim == 1:
            state.local_parameters["user_embeddings"] = vec.astype(
                old_np.dtype, copy=False
            )
        else:
            state.local_parameters["user_embeddings"] = np.broadcast_to(
                vec.astype(old_np.dtype, copy=False), old_np.shape
            ).copy()

    def _profile_feature_matrix(self) -> np.ndarray:
        """Build per-client train-profile features for ``G0``.

        Features are mean rating (or 1.0 if no rating column), log1p
        interaction count, and Shannon entropy of the item histogram.
        Training interactions only.

        Returns
        -------
        ndarray
            Shape ``(K, 3)``.

        Raises
        ------
        ValueError
            If a client has no training interactions.
        """
        train = self.interaction_dataset.train_interactions
        partition = self.interaction_dataset.client_partition
        features = np.zeros((self.num_clients, 3), dtype=np.float64)
        has_rating = "rating" in train.columns
        for cid in range(self.num_clients):
            users = partition.client_ids_2_user_ids[cid]
            mask = train["user_id"].isin(users)
            rows = train.loc[mask]
            if len(rows) == 0:
                raise ValueError(
                    f"Client {cid} has no training interactions; cannot "
                    "build FedFast profile features"
                )
            if has_rating:
                mean_rating = float(rows["rating"].mean())
            else:
                mean_rating = 1.0
            count = float(len(rows))
            item_counts = rows["item_id"].value_counts().to_numpy(dtype=np.float64)
            probs = item_counts / item_counts.sum()
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            features[cid] = [mean_rating, np.log1p(count), entropy]
        return features
