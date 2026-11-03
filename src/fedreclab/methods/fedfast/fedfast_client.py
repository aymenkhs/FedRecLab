"""FedFast client algorithm — local GMF training and scoring.

Source: Muhammad et al., "FedFast: Going Beyond Average for Faster Training
of Federated Recommender Systems", KDD 2020.

ClientState holds the client-resident user embedding table and optional
Adam moments. Item embeddings and the linear head arrive in the server
payload each round.

Training protocol (paper §3.1 / §4): observed interactions are treated as
implicit positives; unobserved items are sampled as negatives; the client
minimises binary cross-entropy with logits. ``label_mode="rating"`` with
``loss="mse"`` is a documented deviation for explicit ratings.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from fedreclab.federated.utils import (
    assert_client_names_disjoint,
    assert_skeleton_server_coverage,
    extract_named_optimizer_state,
    flatten_grouped_state,
    restore_named_optimizer_state,
)
from fedreclab.methods.fedfast.fedfast_model import GeneralMatrixFactorization

logger = logging.getLogger(__name__)

_CLIENT_TENSOR_NAMES = ("user_embeddings.weight",)
_OPTIMIZER_STATE_SCOPES = ("client_resident", "none", "all")
_LABEL_MODES = ("binary", "rating")
_LOSSES = ("bce", "mse")
_ALL_SCOPE_WARN_BYTES = 1 << 30


class FedFastClientAlgorithm(BaseClientAlgorithm):
    """Local training and scoring for FedFast GMF.

    Parameters
    ----------
    embedding_dim : int, optional
        Width of user and item embeddings. Default is 10.
    local_epochs : int, optional
        Local SGD epochs per round. Default is 1.
    lr : float, optional
        Adam learning rate. Default is ``1e-3``.
    batch_size : int, optional
        Mini-batch size. Default is 128.
    neg_samples : int, optional
        Unobserved items sampled per positive. Default is 4.
    label_mode : {"binary", "rating"}, optional
        ``"binary"`` (paper) sets every observed interaction to 1.
        ``"rating"`` keeps the raw rating as the positive target.
    loss : {"bce", "mse"}, optional
        ``"bce"`` (paper) is BCE-with-logits. ``"mse"`` is squared error.
    optimizer_state_scope : {"client_resident", "none", "all"}, optional
        Which Adam moments to persist in ``ClientState``. Default is
        ``"client_resident"``.
    item_upload_include_negatives : bool, optional
        If True, row-layout item uploads include negatives sampled this
        round. Default is False (positives only).

    Raises
    ------
    ValueError
        If ``optimizer_state_scope``, ``label_mode``, or ``loss`` is
        unknown, or if BCE is paired with rating-valued labels.

    Examples
    --------
    >>> client = FedFastClientAlgorithm()
    >>> client.label_mode, client.loss, client.embedding_dim
    ('binary', 'bce', 10)
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        local_epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        neg_samples: int = 4,
        label_mode: str = "binary",
        loss: str = "bce",
        optimizer_state_scope: str = "client_resident",
        item_upload_include_negatives: bool = False,
    ):
        super().__init__()
        if optimizer_state_scope not in _OPTIMIZER_STATE_SCOPES:
            raise ValueError(
                f"optimizer_state_scope must be one of {_OPTIMIZER_STATE_SCOPES}, "
                f"got {optimizer_state_scope!r}"
            )
        if label_mode not in _LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {_LABEL_MODES}, got {label_mode!r}"
            )
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
        if loss == "bce" and label_mode != "binary":
            raise ValueError(
                "loss='bce' requires label_mode='binary'; rating-valued "
                "targets are outside [0, 1]. Use loss='mse' for explicit ratings."
            )
        self.embedding_dim = int(embedding_dim)
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.neg_samples = int(neg_samples)
        self.label_mode = label_mode
        self.loss = loss
        self.optimizer_state_scope = optimizer_state_scope
        self.item_upload_include_negatives = bool(item_upload_include_negatives)
        self.num_items = None
        self._skeleton: GeneralMatrixFactorization | None = None
        self._warned_all_scope = False

    def init_model_skeleton(self) -> None:
        """Build the shared GMF skeleton once ``num_items`` and ``device`` are set.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If ``num_items`` or ``device`` is missing.
        """
        if self._skeleton is not None:
            return
        if self.num_items is None:
            raise RuntimeError("init_model_skeleton requires num_items to be set")
        if self.device is None:
            raise RuntimeError(
                "device is not set. The server assigns client.device from "
                "server.device before initialise; standalone callers must set it."
            )
        self._skeleton = GeneralMatrixFactorization(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
        ).to(self.device)
        server_names = set(flatten_grouped_state(self._skeleton.get_server_state_groups()))
        assert_skeleton_server_coverage(self._skeleton, server_names)
        assert_client_names_disjoint(server_names, _CLIENT_TENSOR_NAMES)

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: Any,
        users_in_client: List[int],
    ) -> Tuple[Dict[str, Any], None]:
        """Create the client-resident user embedding table.

        Parameters
        ----------
        client_id : int
            Client being initialised.
        total_num_users : int
            Global user count (used only for the optional memory warning).
        num_users : int
            Number of users assigned to this client.
        num_items : int
            Catalogue size after remapping.
        client_train_data_view : any
            Unused; present for the base interface.
        users_in_client : list of int
            Global remapped user ids on this client.

        Returns
        -------
        dict
            ``user_embeddings`` table and ``user_id_to_local`` map.
        None
            No initial optimiser state.
        """
        del client_id, client_train_data_view
        if self.num_items is None:
            self.num_items = num_items
        self.init_model_skeleton()
        self._maybe_warn_all_scope(total_num_users)

        table = torch.empty(num_users, self.embedding_dim)
        nn.init.normal_(table, std=0.01)
        user_id_to_local = {uid: idx for idx, uid in enumerate(users_in_client)}
        return {
            "user_embeddings": table,
            "user_id_to_local": user_id_to_local,
        }, None

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """Run local GMF epochs and upload item/output groups.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Payload, local data, and client state for this round.

        Returns
        -------
        ClientTrainingResponse
            Updated client state, sparse item upload, dense output upload,
            and training statistics.
        """
        t0 = time.time()
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device

        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data
        user_id_to_local = client_state.local_parameters["user_id_to_local"]

        model.load_server_weights(training_request.global_payload)
        model.train()

        named_params, user_table = self._inject_user_parameters(client_state)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [user_table], lr=self.lr
        )
        restore_named_optimizer_state(
            optimizer, named_params, client_state.optimizer_state
        )

        user_col, item_col = "user_id", "item_id"
        rating_col = "rating" if "rating" in training_data.columns else None

        pos_global_uids = training_data[user_col].values
        pos_item_ids = training_data[item_col].values
        pos_ratings = (
            training_data[rating_col].values
            if rating_col
            else np.ones(len(training_data), dtype=np.float32)
        )
        pos_local_uids = np.array([user_id_to_local[u] for u in pos_global_uids])

        user_pos_items: Dict[int, set] = {}
        for local_uid, iid in zip(pos_local_uids, pos_item_ids):
            user_pos_items.setdefault(int(local_uid), set()).add(int(iid))

        all_losses: List[float] = []
        num_training_examples = 0
        touched_items = set(int(i) for i in pos_item_ids)
        if training_request.seed is None:
            rng = None
        else:
            rng = np.random.default_rng(
                [
                    training_request.seed,
                    training_request.client_id,
                    training_request.current_round,
                ]
            )

        for _ in range(self.local_epochs):
            samples_x, samples_y = self._build_epoch_samples(
                pos_local_uids, pos_item_ids, pos_ratings, user_pos_items, rng,
            )
            num_training_examples += len(samples_x)
            if self.item_upload_include_negatives:
                touched_items.update(int(i) for i in samples_x[:, 1])

            perm = (
                np.random.permutation(len(samples_x))
                if rng is None
                else rng.permutation(len(samples_x))
            )
            samples_x = samples_x[perm]
            samples_y = samples_y[perm]

            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, len(samples_x), self.batch_size):
                bx = torch.tensor(
                    samples_x[start : start + self.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                by = torch.tensor(
                    samples_y[start : start + self.batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                preds = model(bx, user_table)
                loss = self._compute_loss(preds, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                num_batches += 1

            if num_batches > 0:
                all_losses.append(epoch_loss / num_batches)

        server_groups = model.get_server_state_groups()
        server_groups["item_embeddings"] = self._item_embedding_upload(
            training_request, server_groups["item_embeddings"], touched_items
        )
        updates_packages = {
            group_name: self._update_package(training_request, group_name, group_value)
            for group_name, group_value in server_groups.items()
        }

        updated_state = ClientStateBase(
            client_id=client_id,
            local_parameters={
                "user_embeddings": user_table.detach().cpu().clone(),
                "user_id_to_local": user_id_to_local,
            },
            optimizer_state=self._scoped_optimizer_state(optimizer, named_params),
        )
        return ClientTrainingResponse(
            client_id=client_id,
            updated_client_state=updated_state,
            updates_packages=updates_packages,
            training_losses=all_losses,
            training_duration=time.time() - t0,
            num_training_examples=num_training_examples,
        )

    def _item_embedding_upload(
        self,
        training_request: ClientTrainingRequest,
        item_tables: Dict[str, Any],
        touched_items: set,
    ):
        """Full tables if layout is dense; sparse ``{rows, values}`` otherwise."""
        param_ios = training_request.param_ios or {}
        layout = getattr(param_ios.get("item_embeddings"), "layout", "rows")
        if layout == "dense":
            return item_tables
        rows = np.array(sorted(touched_items), dtype=np.int64)
        values = {}
        for key, table in item_tables.items():
            tensor = torch.as_tensor(table)
            values[key] = tensor[torch.as_tensor(rows, dtype=torch.long)].detach().cpu().clone()
        return {"rows": rows, "values": values}

    def _build_epoch_samples(
        self,
        pos_local_uids: np.ndarray,
        pos_item_ids: np.ndarray,
        pos_ratings: np.ndarray,
        user_pos_items: Dict[int, set],
        rng: np.random.Generator | None,
    ):
        """Build one epoch of positives plus sampled negatives.

        Parameters
        ----------
        pos_local_uids : ndarray
            Local user indices of observed interactions.
        pos_item_ids : ndarray
            Item ids of observed interactions.
        pos_ratings : ndarray
            Raw ratings; ignored when ``label_mode="binary"``.
        user_pos_items : dict of int to set
            Observed item ids per local user.
        rng : numpy.random.Generator or None
            Seeded generator, or ``None`` for global ``np.random``.

        Returns
        -------
        samples_x : ndarray
            User–item pairs, shape ``(n, 2)``.
        samples_y : ndarray
            Labels, shape ``(n,)``.
        """
        n_pos = len(pos_local_uids)
        n_total = n_pos * (1 + self.neg_samples)
        samples_x = np.empty((n_total, 2), dtype=np.int64)
        samples_y = np.empty(n_total, dtype=np.float32)
        idx = 0
        draw = np.random.randint if rng is None else rng.integers
        for i in range(n_pos):
            local_uid = int(pos_local_uids[i])
            pos_item = int(pos_item_ids[i])
            pos_label = 1.0 if self.label_mode == "binary" else float(pos_ratings[i])
            samples_x[idx] = [local_uid, pos_item]
            samples_y[idx] = pos_label
            idx += 1
            excl = user_pos_items.get(local_uid, set())
            for _ in range(self.neg_samples):
                neg_item = int(draw(0, self.num_items))
                while neg_item in excl:
                    neg_item = int(draw(0, self.num_items))
                samples_x[idx] = [local_uid, neg_item]
                samples_y[idx] = 0.0
                idx += 1
        return samples_x[:idx], samples_y[:idx]

    def _compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the configured training loss.

        Parameters
        ----------
        preds : torch.Tensor
            Model scores, shape ``(batch,)``.
        targets : torch.Tensor
            Labels, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Scalar mean loss.
        """
        if self.loss == "bce":
            return torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)
        if self.loss == "mse":
            return torch.nn.functional.mse_loss(preds, targets)
        raise RuntimeError(f"unrecognised loss {self.loss!r}")

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: Dict[int, np.ndarray],
        global_parameters: Dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: Dict[str, Any],
    ) -> Dict[int, np.ndarray]:
        """Score candidate items with the assembled GMF model.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Server-resident GMF groups for this round.
        client_state : ClientStateBase
            Holds the client-resident user embedding table.
        extra_payload : dict
            Unused by FedFast scoring.

        Returns
        -------
        dict of int to ndarray
            Scores per user, aligned with that user's candidates.
        """
        del client_id, extra_payload
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device
        user_id_to_local = client_state.local_parameters["user_id_to_local"]

        model.load_server_weights(global_parameters)
        _, user_table = self._inject_user_parameters(client_state)
        model.eval()

        scores: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                local_uid = user_id_to_local[user_id]
                n_items = len(candidate_items)
                user_ids = torch.full(
                    (n_items,), local_uid, dtype=torch.long, device=device
                )
                item_ids = torch.tensor(
                    candidate_items, dtype=torch.long, device=device
                )
                x = torch.stack([user_ids, item_ids], dim=1)
                scores[user_id] = model(x, user_table).cpu().numpy()
        return scores

    def _inject_user_parameters(self, client_state: ClientStateBase):
        table = nn.Parameter(
            torch.as_tensor(
                client_state.local_parameters["user_embeddings"],
                device=self.device,
            ).clone()
        )
        named_params = {n: p for n, p in self._skeleton.named_parameters()}
        named_params["user_embeddings.weight"] = table
        return named_params, table

    def _scoped_optimizer_state(self, optimizer, named_params):
        if self.optimizer_state_scope == "none":
            return None
        names_to_keep = None
        if self.optimizer_state_scope == "client_resident":
            names_to_keep = _CLIENT_TENSOR_NAMES
        return extract_named_optimizer_state(optimizer, named_params, names_to_keep)

    def _maybe_warn_all_scope(self, num_clients: int) -> None:
        if self.optimizer_state_scope != "all" or self._warned_all_scope:
            return
        self._warned_all_scope = True
        n_params = sum(p.numel() for p in self._skeleton.parameters())
        bytes_est = n_params * 2 * 4 * max(num_clients, 1)
        if bytes_est >= _ALL_SCOPE_WARN_BYTES:
            logger.warning(
                "optimizer_state_scope='all' stores Adam moments for every server "
                "parameter on every client (~%.1f GiB estimated). Prefer "
                "'client_resident' unless you have a state store.",
                bytes_est / _ALL_SCOPE_WARN_BYTES,
            )
