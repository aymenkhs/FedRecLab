"""SemiDFEGL client: lagged LightGCN local train, privacy and scoring.

Source: Qu et al., WWW 2023. Local BPR on the augmented ego graph;
device-to-device mixing uses detached neighbour ego vectors from the
previous round (simulation shortcut).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from fedreclab.federated.mechanisms import ClipLaplaceLDP, PseudoItemSampling
from fedreclab.federated.utils import (
    extract_named_optimizer_state,
    restore_named_optimizer_state,
)
from fedreclab.methods.semidfegl.semidfegl_model import (
    propagate_semidfegl,
    score_dot,
    xavier_uniform_table,
)


class SemiDFEGLClientAlgorithm(BaseClientAlgorithm):
    """Local training and scoring for SemiDFEGL.

    The user embedding is updated with Adam whose moments persist in
    ``ClientState``. Item rows are updated with SGD at the same ``lr``
    and ``adam_weight_decay`` as weight decay. Item moments cannot be
    kept on the device (no resident item table); a fresh Adam state each
    round with ``local_epochs=1`` would reduce to signSGD on items.

    Parameters
    ----------
    embedding_dim : int, optional
        Width of embeddings. Must match the server. Default 64.
    num_layers : int, optional
        LightGCN depth ``K``. Default 4.
    local_epochs : int, optional
        Local passes over the BPR objective per round. Default 1.
    lr : float, optional
        Learning rate for user Adam and item SGD. Default ``0.001``.
    adam_weight_decay : float, optional
        Weight decay for user Adam and item SGD. Default ``1e-4``.
    bpr_reg : float, optional
        Extra L2 on layer-0 user and scored item rows. Default ``0.0``.
    fake_items_as_negatives : bool, optional
        Treat fake items the user did not train on as BPR negatives.
        Default True.
    num_pseudo_items : int, optional
        Privacy pseudo embeddings appended after training. Default 0.
    clip_threshold : float or None, optional
        L1 clip on uploaded item embedding values. ``None`` disables.
    laplace_scale : float, optional
        Laplace scale on uploaded item values. ``0`` disables.

    Raises
    ------
    ValueError
        If a hyper-parameter is out of range.

    Examples
    --------
    >>> client = SemiDFEGLClientAlgorithm(embedding_dim=8, num_layers=1)
    >>> client.num_pseudo_items, client.adam_weight_decay
    (0, 0.0001)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 4,
        local_epochs: int = 1,
        lr: float = 0.001,
        adam_weight_decay: float = 1e-4,
        bpr_reg: float = 0.0,
        fake_items_as_negatives: bool = True,
        num_pseudo_items: int = 0,
        clip_threshold: float | None = None,
        laplace_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim < 1:
            raise ValueError(f"embedding_dim must be a positive int, got {embedding_dim!r}")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 0:
            raise ValueError(f"num_layers must be a non-negative int, got {num_layers!r}")
        if not isinstance(local_epochs, int) or isinstance(local_epochs, bool) or local_epochs < 1:
            raise ValueError(f"local_epochs must be a positive int, got {local_epochs!r}")
        if float(lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if float(adam_weight_decay) < 0.0:
            raise ValueError(f"adam_weight_decay must be >= 0, got {adam_weight_decay}")
        if float(bpr_reg) < 0.0:
            raise ValueError(f"bpr_reg must be >= 0, got {bpr_reg}")
        if (
            not isinstance(num_pseudo_items, int)
            or isinstance(num_pseudo_items, bool)
            or num_pseudo_items < 0
        ):
            raise ValueError(
                f"num_pseudo_items must be a non-negative int, got {num_pseudo_items!r}"
            )

        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)
        self.adam_weight_decay = float(adam_weight_decay)
        self.bpr_reg = float(bpr_reg)
        self.fake_items_as_negatives = bool(fake_items_as_negatives)
        self.num_pseudo_items = int(num_pseudo_items)
        self.clip_threshold = clip_threshold
        self.laplace_scale = float(laplace_scale)
        self.clip_laplace = ClipLaplaceLDP(clip=clip_threshold, scale=laplace_scale)
        self.pseudo_sampler = PseudoItemSampling(self.num_pseudo_items)
        self.num_items: int | None = None
        self.init_seed: int | None = None

    def init_model_skeleton(self) -> None:
        """No shared module; SemiDFEGL builds round-local parameters.

        Returns
        -------
        None
        """
        return None

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: pd.DataFrame,
        users_in_client: list[int],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Create a Xavier user embedding for the single local user.

        Parameters
        ----------
        client_id : int
            Client id (unused beyond seeding).
        total_num_users : int
            Catalogue of users.
        num_users : int
            Users on this client; must be 1.
        num_items : int
            Catalogue size.
        client_train_data_view : pandas.DataFrame
            Training interactions.
        users_in_client : list of int
            Remapped user ids; length must be 1.

        Returns
        -------
        local_parameters : dict
            ``user_embedding`` of shape ``(d,)``.
        optimizer_state : None
            Adam moments start empty.

        Raises
        ------
        ValueError
            If the client does not own exactly one user.
        """
        if len(users_in_client) != 1:
            raise ValueError(
                "SemiDFEGL requires exactly one user per client; "
                f"got {users_in_client}"
            )
        self.num_items = int(num_items)
        # Deterministic per-client Xavier from a derived seed when possible.
        gen = torch.Generator()
        seed = 0 if self.init_seed is None else int(self.init_seed)
        gen.manual_seed(seed + 10007 * int(client_id) + 17)
        row = xavier_uniform_table(1, self.embedding_dim, generator=gen).reshape(-1)
        return {"user_embedding": row.detach().cpu().clone()}, None

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """Run BPR on the augmented ego graph and upload item / ego rows.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Must carry the SemiDFEGL sliced ``global_payload``.

        Returns
        -------
        ClientTrainingResponse
            Item-row and ego-row packages, updated client state.

        Raises
        ------
        KeyError
            If a required payload key is missing.
        ValueError
            If the loss or an embedding is non-finite, or the client has
            the wrong number of users.
        """
        t0 = time.time()
        if self.device is None:
            raise RuntimeError(
                "device is not set. The server assigns client.device from "
                "server.device before initialise; standalone callers must set it."
            )
        if len(training_request.users_in_client) != 1:
            raise ValueError(
                "SemiDFEGL requires exactly one user per client; "
                f"got {training_request.users_in_client}"
            )
        user_id = int(training_request.users_in_client[0])
        payload = training_request.global_payload
        for key in (
            "item_ids",
            "item_embeddings",
            "fake_item_ids",
            "bpr_negative_ids",
            "train_item_ids",
            "cluster_user_ids",
            "cluster_ego_embeddings",
            "cluster_degrees",
            "cluster_size",
            "user_degree",
        ):
            if key not in payload:
                raise KeyError(f"global_payload missing required key {key!r}")

        device = torch.device(self.device)
        item_ids = np.asarray(payload["item_ids"], dtype=np.int64)
        item_table = np.asarray(payload["item_embeddings"], dtype=np.float32)
        if item_ids.shape[0] != item_table.shape[0]:
            raise ValueError(
                f"item_ids length {item_ids.shape[0]} != item_embeddings rows "
                f"{item_table.shape[0]}"
            )
        id_to_local = {int(i): loc for loc, i in enumerate(item_ids.tolist())}

        train_ids = np.asarray(payload["train_item_ids"], dtype=np.int64)
        fake_ids = np.asarray(payload["fake_item_ids"], dtype=np.int64)
        bpr_neg_ids = np.asarray(payload["bpr_negative_ids"], dtype=np.int64)
        if train_ids.size == 0:
            raise ValueError(
                f"client {training_request.client_id} has no training items"
            )

        # Graph nodes: training positives ∪ fake bridges.
        graph_ids = np.unique(np.concatenate([train_ids, fake_ids]))
        graph_local = np.asarray([id_to_local[int(i)] for i in graph_ids], dtype=np.int64)
        fake_set = set(int(i) for i in fake_ids)
        item_is_fake = torch.tensor(
            [int(i) in fake_set for i in graph_ids],
            dtype=torch.bool,
            device=device,
        )

        cluster_user_ids = np.asarray(payload["cluster_user_ids"], dtype=np.int64)
        cluster_egos = np.asarray(payload["cluster_ego_embeddings"], dtype=np.float32)
        cluster_degrees = np.asarray(payload["cluster_degrees"], dtype=np.float32)
        cluster_size = int(payload["cluster_size"])
        user_degree = float(payload["user_degree"])

        neighbor_mask = cluster_user_ids != user_id
        neighbor_egos_np = cluster_egos[neighbor_mask]
        neighbor_degrees_np = cluster_degrees[neighbor_mask]

        user_np = np.asarray(
            training_request.client_state.local_parameters["user_embedding"],
            dtype=np.float32,
        ).reshape(-1)
        if user_np.shape[0] != self.embedding_dim:
            raise ValueError(
                f"stored user_embedding width {user_np.shape[0]} != "
                f"embedding_dim={self.embedding_dim}"
            )

        user_emb = nn.Parameter(torch.as_tensor(user_np, dtype=torch.float32, device=device))
        # Only graph items are updated by message passing / BPR on graph
        # edges; BPR negatives that are not graph nodes are scored from a
        # separate trainable table slice so they can be uploaded.
        graph_emb = nn.Parameter(
            torch.as_tensor(item_table[graph_local], dtype=torch.float32, device=device)
        )

        neg_only_ids = [
            int(i)
            for i in bpr_neg_ids
            if int(i) not in set(int(x) for x in graph_ids)
        ]
        if self.fake_items_as_negatives:
            fake_neg_ids = [
                int(i) for i in fake_ids if int(i) not in set(int(x) for x in train_ids)
            ]
        else:
            fake_neg_ids = []
        # Fake negatives already sit in graph_emb; dedicated BPR negatives
        # that are not graph nodes need their own parameters.
        neg_only_local_global = neg_only_ids
        if neg_only_local_global:
            neg_only_emb = nn.Parameter(
                torch.as_tensor(
                    item_table[[id_to_local[i] for i in neg_only_local_global]],
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            neg_only_emb = None

        neighbor_egos = torch.as_tensor(
            neighbor_egos_np, dtype=torch.float32, device=device
        )
        neighbor_degrees = torch.as_tensor(
            neighbor_degrees_np, dtype=torch.float32, device=device
        )

        # Adam on the resident user embedding; SGD on round-local items.
        # A fresh Adam state on items with one full-batch step is ≈ signSGD.
        user_optimizer = torch.optim.Adam(
            [user_emb],
            lr=self.lr,
            weight_decay=self.adam_weight_decay,
        )
        item_params = [graph_emb]
        if neg_only_emb is not None:
            item_params.append(neg_only_emb)
        item_optimizer = torch.optim.SGD(
            item_params,
            lr=self.lr,
            weight_decay=self.adam_weight_decay,
        )
        restore_named_optimizer_state(
            user_optimizer,
            {"user_embedding": user_emb},
            training_request.client_state.optimizer_state,
        )

        graph_id_to_row = {int(g): r for r, g in enumerate(graph_ids.tolist())}
        train_graph_rows = torch.as_tensor(
            [graph_id_to_row[int(i)] for i in train_ids],
            dtype=torch.long,
            device=device,
        )
        neg_graph_rows = [
            graph_id_to_row[i] for i in fake_neg_ids if i in graph_id_to_row
        ]

        all_losses: list[float] = []
        num_training_examples = 0

        for _ in range(self.local_epochs):
            user_optimizer.zero_grad()
            item_optimizer.zero_grad()
            user_c, _item_states = propagate_semidfegl(
                user_emb0=user_emb,
                item_emb0=graph_emb,
                item_is_fake=item_is_fake,
                neighbor_egos=neighbor_egos,
                user_degree=user_degree,
                neighbor_degrees=neighbor_degrees,
                cluster_size=cluster_size,
                num_layers=self.num_layers,
            )
            # Ranking dots this ego vector with the stored item row.
            # Held-out items are not graph neighbours, so the item factor
            # is the layer-0 parameter. Layer-averaged item states stay
            # inside propagate_semidfegl.
            pos_scores = score_dot(user_c, graph_emb[train_graph_rows])

            neg_score_parts: list[torch.Tensor] = []
            if neg_graph_rows:
                neg_g = graph_emb[
                    torch.as_tensor(neg_graph_rows, dtype=torch.long, device=device)
                ]
                neg_score_parts.append(score_dot(user_c, neg_g))
            if neg_only_emb is not None:
                neg_score_parts.append(score_dot(user_c, neg_only_emb))

            if not neg_score_parts:
                raise ValueError(
                    f"client {training_request.client_id} has no BPR negatives; "
                    "increase bpr_negatives or enable fake_items_as_negatives"
                )
            neg_scores = torch.cat(neg_score_parts, dim=0)

            # Pair every positive with every negative (paper double sum).
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            bpr = -torch.nn.functional.logsigmoid(diff).mean()
            if self.bpr_reg > 0.0:
                reg_terms = [user_emb.pow(2).sum(), graph_emb.pow(2).sum()]
                if neg_only_emb is not None:
                    reg_terms.append(neg_only_emb.pow(2).sum())
                bpr = bpr + self.bpr_reg * torch.stack(reg_terms).sum()
            if not torch.isfinite(bpr):
                raise ValueError(
                    f"Non-finite SemiDFEGL loss for client "
                    f"{training_request.client_id}: {float(bpr.detach().cpu())}"
                )
            bpr.backward()
            user_optimizer.step()
            item_optimizer.step()
            all_losses.append(float(bpr.detach().cpu()))
            num_training_examples += int(train_ids.size) * int(neg_scores.numel())

        # Ego upload is the layer-combined user (the ranking left factor
        # and the device-to-device message). Item upload is the stepped
        # layer-0 parameter, which the next round propagates from.
        with torch.no_grad():
            user_c, _item_states = propagate_semidfegl(
                user_emb0=user_emb,
                item_emb0=graph_emb,
                item_is_fake=item_is_fake,
                neighbor_egos=neighbor_egos,
                user_degree=user_degree,
                neighbor_degrees=neighbor_degrees,
                cluster_size=cluster_size,
                num_layers=self.num_layers,
            )
            ego_row = user_c.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(ego_row).all():
                raise ValueError(
                    f"Non-finite ego embedding for client {training_request.client_id}"
                )

        upload_ids: list[int] = [int(i) for i in graph_ids]
        upload_vals = [graph_emb.detach().cpu().numpy().astype(np.float32)]
        if neg_only_emb is not None:
            upload_ids.extend(neg_only_local_global)
            upload_vals.append(neg_only_emb.detach().cpu().numpy().astype(np.float32))
        values = np.concatenate(upload_vals, axis=0).astype(np.float32)
        rows = np.asarray(upload_ids, dtype=np.int64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite item embeddings for client {training_request.client_id}"
            )

        item_upload = {"rows": rows, "values": values}
        ego_upload = {
            "rows": np.asarray([user_id], dtype=np.int64),
            "values": ego_row.reshape(1, -1).astype(np.float32),
        }

        updates_packages = {
            "item_embeddings": self._update_package(
                training_request, "item_embeddings", item_upload
            ),
            "ego_embeddings": self._update_package(
                training_request, "ego_embeddings", ego_upload
            ),
        }

        user_state = extract_named_optimizer_state(
            user_optimizer,
            {"user_embedding": user_emb},
            names_to_keep=["user_embedding"],
        )
        updated_state = ClientStateBase(
            client_id=training_request.client_id,
            local_parameters={
                "user_embedding": user_emb.detach().cpu().clone(),
            },
            optimizer_state=user_state,
            metadata={
                "train_item_ids": train_ids.astype(np.int64, copy=True),
            },
        )

        return ClientTrainingResponse(
            client_id=training_request.client_id,
            updated_client_state=updated_state,
            updates_packages=updates_packages,
            training_losses=all_losses,
            training_duration=time.time() - t0,
            num_training_examples=num_training_examples,
        )

    def apply_privacy(
        self,
        training_request: ClientTrainingRequest,
        training_response: ClientTrainingResponse,
    ) -> ClientTrainingResponse:
        """Append matched pseudo embeddings and optionally clip + noise.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Request that produced ``training_response``.
        training_response : ClientTrainingResponse
            Unprotected training result.

        Returns
        -------
        ClientTrainingResponse
            Same object, with ``item_embeddings`` possibly rewritten.

        Raises
        ------
        RuntimeError
            If privacy is enabled and ``seed`` is missing.
        ValueError
            If the pseudo pool is too small or a value is non-finite.
        """
        if self.num_pseudo_items == 0 and not self.clip_laplace.enabled:
            return training_response
        if training_request.seed is None:
            raise RuntimeError(
                "SemiDFEGL privacy requires seed= on the training request"
            )
        if self.num_items is None:
            raise RuntimeError(
                "num_items is unknown; call initialize_local_parameters first"
            )

        pkg = training_response.updates_packages["item_embeddings"]
        rows = np.asarray(pkg.param_value["rows"], dtype=np.int64).copy()
        values = np.asarray(pkg.param_value["values"], dtype=np.float32).copy()

        train_ids = training_response.updated_client_state.metadata.get(
            "train_item_ids"
        )
        if train_ids is None:
            train_ids = np.asarray(
                training_request.global_payload["train_item_ids"], dtype=np.int64
            )
        train_set = set(int(i) for i in np.asarray(train_ids, dtype=np.int64))
        positive_mask = np.array([int(r) in train_set for r in rows], dtype=bool)
        if not positive_mask.any():
            raise ValueError(
                "Cannot match privacy pseudo embeddings: the upload has no "
                "training-positive rows"
            )
        positive_values = values[positive_mask]

        rng = np.random.default_rng(
            [
                int(training_request.seed),
                int(training_request.client_id),
                int(training_request.current_round),
                91,
            ]
        )
        if self.num_pseudo_items > 0:
            excluded = train_set.union(int(r) for r in rows)
            pseudo_ids = self.pseudo_sampler.sample_ids(
                self.num_items, excluded, rng
            )
            # Match absolute embeddings to the empirical diagonal Gaussian
            # of uploaded positive embeddings (paper §3.2.3).
            mean = positive_values.mean(axis=0)
            std = positive_values.std(axis=0, ddof=0)
            pseudo_values = rng.normal(
                loc=mean, scale=std, size=(self.num_pseudo_items, values.shape[1])
            ).astype(np.float32)
            if not np.isfinite(pseudo_values).all():
                raise ValueError("Matched pseudo embeddings contain non-finite values")
            rows = np.concatenate([rows, pseudo_ids]).astype(np.int64, copy=False)
            values = np.concatenate([values, pseudo_values], axis=0).astype(
                np.float32, copy=False
            )

        if self.clip_laplace.enabled:
            protected = self.clip_laplace.apply([values], rng)[0]
            values = np.asarray(protected, dtype=np.float32)

        training_response.updates_packages["item_embeddings"] = self._update_package(
            training_request,
            "item_embeddings",
            {"rows": rows, "values": values},
        )
        return training_response

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: dict[int, np.ndarray],
        global_parameters: dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: dict[str, Any],
    ) -> dict[int, np.ndarray]:
        """Score candidates by ego-row times item-row (equation 8).

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Must contain ``ego_embeddings`` and ``item_embeddings``.
        client_state : ClientStateBase
            Unused for scoring; the ego table is server-resident.
        extra_payload : dict
            Unused.

        Returns
        -------
        dict of int to ndarray
            One score array per user.

        Raises
        ------
        KeyError
            If a required global table is missing.
        """
        _ = client_id, client_state, extra_payload
        egos = np.asarray(global_parameters["ego_embeddings"], dtype=np.float32)
        items = np.asarray(global_parameters["item_embeddings"], dtype=np.float32)
        out: dict[int, np.ndarray] = {}
        for user_id, candidates in candidate_items_per_user.items():
            cand = np.asarray(candidates, dtype=np.int64)
            ego = egos[int(user_id)]
            scores = items[cand] @ ego
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"Non-finite SemiDFEGL scores for user {user_id}"
                )
            out[int(user_id)] = scores.astype(np.float32, copy=False)
        return out
