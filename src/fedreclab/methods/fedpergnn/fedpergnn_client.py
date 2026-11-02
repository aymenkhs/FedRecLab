"""FedPerGNN client algorithm.

The shared GAT skeleton lives on this object and is loaded from the server
payload each call. ClientState holds the owner's user embedding and a
name-keyed optimiser dict; that embedding is the source of truth for the
owner's trajectory. The server user table is a mirror updated by the sparse
upload each round (expansion / neighbours). With neighbor_embeddings_trainable,
other clients can still write the owner's row in the mirror, but that no
longer becomes this client's next starting point.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from torch import nn

from src.fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from src.fedreclab.federated.utils import (
    assert_client_names_disjoint,
    assert_skeleton_server_coverage,
    extract_named_optimizer_state,
    restore_named_optimizer_state,
)
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel

logger = logging.getLogger(__name__)

_CLIENT_TENSOR_NAMES = ("user_embedding.weight",)
_OPTIMIZER_STATE_SCOPES = ("client_resident", "none", "all")
_ALL_SCOPE_WARN_BYTES = 1 << 30


class FedPerGNNClientAlgorithm(BaseClientAlgorithm):

    def __init__(
        self,
        hidden_dim: int = 64,
        history_len: int = 20,
        neighbor_len: int = 10,
        local_epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
        neighbor_embeddings_trainable: bool = False,
        optimizer_state_scope: str = "client_resident",
    ) -> None:
        super().__init__()
        if optimizer_state_scope not in _OPTIMIZER_STATE_SCOPES:
            raise ValueError(
                f"optimizer_state_scope must be one of {_OPTIMIZER_STATE_SCOPES}, "
                f"got {optimizer_state_scope!r}"
            )
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.neighbor_embeddings_trainable = neighbor_embeddings_trainable
        self.optimizer_state_scope = optimizer_state_scope

        self._num_items: int | None = None
        self._skeleton: FedPerGNNModel | None = None
        self._warned_all_scope = False

    def init_model_skeleton(self):
        if self._skeleton is not None:
            return
        if self._num_items is None:
            raise RuntimeError("init_model_skeleton requires num_items to be set")
        if self.device is None:
            raise RuntimeError(
                "device is not set. The server assigns client.device from "
                "server.device before initialise; standalone callers must set it."
            )
        self._skeleton = FedPerGNNModel(
            num_items=self._num_items,
            hidden_dim=self.hidden_dim,
            history_len=self.history_len,
            neighbor_len=self.neighbor_len,
        ).to(self.device)
        server_names = set(self._skeleton.get_server_state())
        assert_skeleton_server_coverage(self._skeleton, server_names)
        assert_client_names_disjoint(server_names, _CLIENT_TENSOR_NAMES)

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view,
        users_in_client: List[int],
    ) -> Tuple[Dict[str, Any], None]:
        if self._num_items is None:
            self._num_items = num_items
        self.init_model_skeleton()
        self._maybe_warn_all_scope(total_num_users)

        user_emb = torch.empty(1, self.hidden_dim)
        nn.init.normal_(user_emb, std=0.01)
        return {"user_embedding.weight": user_emb}, None

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        t0 = time.time()
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device

        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data

        payload = training_request.global_payload
        model.load_server_state(payload["model_state"])
        if "item_embeddings" in payload:
            model.load_item_embeddings(payload["item_embeddings"])
        model.train()

        named_params, user_emb = self._inject_user_parameter(client_state)
        optimizer = torch.optim.Adam(list(model.parameters()) + [user_emb], lr=self.lr)
        restore_named_optimizer_state(optimizer, named_params, client_state.optimizer_state)

        users_in_client = training_request.users_in_client
        central_user_id = int(users_in_client[0]) if users_in_client else 0

        item_col = "item_id"
        rating_col = "rating" if "rating" in training_data.columns else None

        item_ids_np = training_data[item_col].values.astype(np.int64)
        if rating_col is not None:
            ratings_np = training_data[rating_col].values.astype(np.float32)
        else:
            ratings_np = np.ones(len(training_data), dtype=np.float32)

        expanded_for_client = self._require_expansion(training_request.extra_payload, client_id)
        history_items_np = expanded_for_client.get("history_item_ids")
        neighbor_emb_np = expanded_for_client.get("neighbor_embeddings")
        neighbor_user_ids_np = expanded_for_client.get("neighbor_user_ids")

        sum_emb: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.hidden_dim, dtype=np.float32))
        sum_grad: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.hidden_dim, dtype=np.float32))
        count_grad: Dict[int, int] = defaultdict(int)
        count_emb: Dict[int, int] = defaultdict(int)
        if self.neighbor_embeddings_trainable and neighbor_emb_np is not None and neighbor_user_ids_np is not None:
            L, N = neighbor_user_ids_np.shape
            for ell in range(L):
                for n in range(N):
                    uid = int(neighbor_user_ids_np[ell, n])
                    if uid >= 0:
                        sum_emb[uid] += neighbor_emb_np[ell, n, :]
                        count_emb[uid] += 1

        num_examples = len(item_ids_np)
        all_losses: List[float] = []
        num_training_examples = 0
        if training_request.seed is None:
            rng = None
        else:
            rng = np.random.default_rng(
                [training_request.seed, training_request.client_id, training_request.current_round]
            )

        for _ in range(self.local_epochs):
            indices = np.arange(num_examples)
            if rng is None:
                np.random.shuffle(indices)
            else:
                rng.shuffle(indices)

            item_ids_np_shuffled = item_ids_np[indices]
            ratings_np_shuffled = ratings_np[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_examples, self.batch_size):
                end = min(start + self.batch_size, num_examples)
                if start >= end:
                    break

                b_item_ids = torch.as_tensor(
                    item_ids_np_shuffled[start:end], dtype=torch.long, device=device
                )
                b_ratings = torch.as_tensor(
                    ratings_np_shuffled[start:end], dtype=torch.float32, device=device
                )

                b_hist_items = torch.as_tensor(
                    np.repeat(history_items_np[None, :], repeats=b_item_ids.size(0), axis=0),
                    dtype=torch.long,
                    device=device,
                )

                if neighbor_emb_np is not None:
                    b_neighbor_emb = torch.as_tensor(
                        np.repeat(neighbor_emb_np[None, :, :, :], repeats=b_item_ids.size(0), axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                    if self.neighbor_embeddings_trainable:
                        b_neighbor_emb.requires_grad_(True)
                else:
                    b_neighbor_emb = None

                optimizer.zero_grad()
                preds = model(
                    item_ids=b_item_ids,
                    history_item_ids=b_hist_items,
                    user_embedding=user_emb,
                    neighbor_embeddings=b_neighbor_emb,
                )
                loss = torch.mean((preds - b_ratings) ** 2)
                loss.backward()

                if self.neighbor_embeddings_trainable and b_neighbor_emb is not None and b_neighbor_emb.grad is not None and neighbor_user_ids_np is not None:
                    grad_np = b_neighbor_emb.grad.detach().cpu().numpy()
                    B, L, N, H = grad_np.shape
                    for ell in range(L):
                        for n in range(N):
                            uid = int(neighbor_user_ids_np[ell, n])
                            if uid >= 0:
                                sum_grad[uid] += grad_np[:, ell, n, :].sum(axis=0)
                                count_grad[uid] += B

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                num_training_examples += b_item_ids.size(0)

            if num_batches > 0:
                all_losses.append(epoch_loss / num_batches)

        numpy_state: Dict[str, Any] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in model.get_model_state().items()
        }

        item_rows = set(int(i) for i in item_ids_np)
        if history_items_np is not None:
            item_rows.update(int(i) for i in np.asarray(history_items_np).reshape(-1))
        item_upload = self._embedding_upload(
            training_request,
            "item_embeddings",
            model.get_item_embeddings(),
            item_rows,
        )

        user_embedding_row = user_emb.detach().cpu().numpy()
        user_ids_list: List[int] = [central_user_id]
        embeddings_list: List[np.ndarray] = [np.asarray(user_embedding_row.squeeze(0), dtype=np.float32)]
        if self.neighbor_embeddings_trainable and count_emb and count_grad:
            for uid in count_emb:
                if uid == central_user_id:
                    continue
                if count_grad[uid] > 0:
                    mean_emb = sum_emb[uid] / count_emb[uid]
                    mean_grad = sum_grad[uid] / count_grad[uid]
                    updated = mean_emb - self.lr * mean_grad
                    user_ids_list.append(uid)
                    embeddings_list.append(np.asarray(updated, dtype=np.float32))

        user_upload = self._embedding_upload(
            training_request,
            "user_embeddings",
            np.stack(embeddings_list, axis=0).astype(np.float32),
            user_ids_list,
            values_already_rows=True,
        )

        updates_packages = {
            "model_state": self._update_package(training_request, "model_state", numpy_state),
            "item_embeddings": self._update_package(training_request, "item_embeddings", item_upload),
            "user_embeddings": self._update_package(training_request, "user_embeddings", user_upload),
        }

        updated_state = ClientStateBase(
            client_id=client_id,
            local_parameters={
                "user_embedding.weight": user_emb.detach().cpu().clone(),
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

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: Dict[int, np.ndarray],
        global_parameters: Dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: Dict[str, Any],
    ) -> Dict[int, np.ndarray]:
        """Score candidate items with the assembled FedPerGNN model.

        See ``BaseClientAlgorithm.compute_scores`` for the ranking-evaluation
        contract. This method returns scores only; it does not compute a
        validation loss.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Server-resident GAT state and optional item embeddings.
        client_state : ClientStateBase
            Holds the owner user embedding.
        extra_payload : dict
            Must contain the expanded subgraph for ``client_id``.

        Returns
        -------
        dict of int to ndarray
            Unbounded inner-product scores per user, aligned with that
            user's candidates. Higher is better.

        Raises
        ------
        KeyError
            If ``extra_payload`` has no expansion for ``client_id``.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> scores = client.compute_scores(0, {0: np.array([1, 2])}, payload, state, extra)
        >>> scores[0].shape
        (2,)
        """
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device

        model.load_server_state(global_parameters["model_state"])
        if "item_embeddings" in global_parameters:
            model.load_item_embeddings(global_parameters["item_embeddings"])
        _, user_emb = self._inject_user_parameter(client_state)
        model.eval()

        expanded_for_client = self._require_expansion(extra_payload, client_id)
        history_items_np = expanded_for_client.get("history_item_ids")
        neighbor_emb_np = expanded_for_client.get("neighbor_embeddings")

        scores: Dict[int, np.ndarray] = {}

        with torch.no_grad():
            for user_id, cand_items in candidate_items_per_user.items():
                b = len(cand_items)
                item_ids = torch.as_tensor(cand_items, dtype=torch.long, device=device)
                hist_items = torch.as_tensor(
                    np.repeat(history_items_np[None, :], repeats=b, axis=0),
                    dtype=torch.long,
                    device=device,
                )

                if neighbor_emb_np is not None:
                    neigh = torch.as_tensor(
                        np.repeat(neighbor_emb_np[None, :, :, :], repeats=b, axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    neigh = None

                preds = model(
                    item_ids=item_ids,
                    history_item_ids=hist_items,
                    user_embedding=user_emb,
                    neighbor_embeddings=neigh,
                )
                scores[user_id] = preds.cpu().numpy()

        return scores

    def _embedding_upload(
        self,
        training_request: ClientTrainingRequest,
        group_name: str,
        table,
        row_ids,
        values_already_rows: bool = False,
    ):
        """Full table if layout is dense; sparse {rows, values} otherwise."""
        param_ios = training_request.param_ios or {}
        layout = getattr(param_ios.get(group_name), "layout", "rows")
        rows = np.asarray(list(row_ids), dtype=np.int64)
        if values_already_rows:
            values = np.asarray(table, dtype=np.float32)
        else:
            tensor = torch.as_tensor(table)
            values = tensor[torch.as_tensor(rows, dtype=torch.long)].detach().cpu().numpy()
        if layout == "dense":
            if values_already_rows:
                raise ValueError(
                    f"Dense layout for '{group_name}' needs the full table; "
                    "got pre-sliced rows"
                )
            return table if not hasattr(table, "detach") else table.detach().cpu().clone()
        return {"rows": rows, "values": np.asarray(values, dtype=np.float32)}

    def _inject_user_parameter(self, client_state: ClientStateBase):
        weight = client_state.local_parameters["user_embedding.weight"]
        user_emb = nn.Parameter(torch.as_tensor(weight, device=self.device).clone())
        named_params = {n: p for n, p in self._skeleton.named_parameters()}
        named_params["user_embedding.weight"] = user_emb
        return named_params, user_emb

    def _scoped_optimizer_state(self, optimizer, named_params):
        if self.optimizer_state_scope == "none":
            return None
        names_to_keep = None
        if self.optimizer_state_scope == "client_resident":
            names_to_keep = _CLIENT_TENSOR_NAMES
        return extract_named_optimizer_state(optimizer, named_params, names_to_keep)

    def _maybe_warn_all_scope(self, num_clients: int):
        if self.optimizer_state_scope != "all" or self._warned_all_scope:
            return
        self._warned_all_scope = True
        n_params = sum(p.numel() for p in self._skeleton.parameters())
        bytes_est = n_params * 2 * 4 * max(num_clients, 1)
        if bytes_est >= _ALL_SCOPE_WARN_BYTES:
            logger.warning(
                "optimizer_state_scope='all' stores Adam moments for every server "
                "parameter on every client (~%.1f GiB estimated). This is memory-heavy; "
                "prefer 'client_resident' unless you have a state store.",
                bytes_est / _ALL_SCOPE_WARN_BYTES,
            )

    @staticmethod
    def _require_expansion(extra_payload, client_id: int):
        expanded_payload_all = extra_payload or {}
        expanded_for_client = expanded_payload_all.get("expanded_user_data", {}).get(client_id, None)
        if expanded_for_client is None:
            raise KeyError(
                "FedPerGNN requires server-provided expansion payload at "
                f"extra_payload['expanded_user_data'][{client_id}]."
            )
        return expanded_for_client
