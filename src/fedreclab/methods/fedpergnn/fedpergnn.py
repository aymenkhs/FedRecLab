from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.base_client import ClientTrainingResponse
from src.fedreclab.federated.utils import ParamIO, RoundStatistics
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


class FedPerGNN(BaseFederatedServer):
    def __init__(
        self,
        hidden_dim: int = 64,
        history_len: int = 20,
        neighbor_len: int = 10,
        warmup_rounds: int = 1,
        neighbor_embeddings_trainable: bool = False,
        user_embeddings_aggregate: bool = False,
        expanded_users: Optional[Dict[int, np.ndarray]] = None,
        history_item_ids: Optional[Dict[int, np.ndarray]] = None,
        expansion_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len
        self.warmup_rounds = warmup_rounds

        # Flags: neighbor_embeddings_trainable applies only to neighbor rows; central user is always trainable.
        self.neighbor_embeddings_trainable = neighbor_embeddings_trainable
        self.user_embeddings_aggregate = user_embeddings_aggregate

        # Optional externally-provided expansion structure
        self._provided_neighbor_user_ids = expanded_users
        self._provided_history_item_ids = history_item_ids
        self.expansion_seed = expansion_seed

        # Expansion cache (integer structure only):
        # client_id -> {"history_item_ids": (L,), "neighbor_user_ids": (L,N)}
        self._expansion_cache: Dict[int, Dict[str, np.ndarray]] = {}

        # Precomputed indices for fast expansion construction
        self._item_to_users: Dict[int, np.ndarray] = {}
        self._user_to_items: Dict[int, np.ndarray] = {}

        # One logical parameter group for all server-side model parameters.
        param_ios = {
            "model_state": ParamIO(
                residency="server",
                exchange="params",
                apply_to="server",
            ),
            # Canonical user embedding table (server-side). Exchanged so clients can load/send sparse updates.
            # apply_to="custom": we scatter or per-user aggregate then scatter in _apply_aggregated_result_custom.
            "user_embeddings": ParamIO(
                residency="server",
                exchange="params",
                apply_to="custom",
            ),
        }

        super().__init__(param_ios=param_ios, **kwargs)

    def _initialize_server_parameters(self) -> Dict[str, Any]:
        """
        Create initial global model parameters using a temporary model.

        The temporary model is never used directly for training; only its
        server-side state is extracted and stored in ``self.global_parameters``.
        """
        model = FedPerGNNModel(
            num_items=self.interaction_dataset.num_items,
            hidden_dim=self.hidden_dim,
            history_len=self.history_len,
            neighbor_len=self.neighbor_len,
        )
        server_state = model.get_server_state()

        # Ensure everything is converted to numpy for aggregation backends.
        numpy_state: Dict[str, Any] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in server_state.items()
        }

        # Canonical user embedding table on server (used for expansion). Not aggregated yet.
        rng = np.random.default_rng(self.expansion_seed)
        user_embeddings = rng.normal(0.0, 0.01, size=(self.interaction_dataset.num_users, self.hidden_dim)).astype(np.float32)

        return {
            "model_state": numpy_state,
            "user_embeddings": user_embeddings,
        }

    def _before_training(self):
        self._build_fast_indices()
        self._construct_expanded_users()

    def _build_fast_indices(self):
        """Precompute item->users and user->items arrays for fast expansion."""
        df = self.interaction_dataset.train_interactions
        if not isinstance(df, pd.DataFrame):
            raise TypeError("interaction_dataset.train_interactions must be a pandas DataFrame")

        # item -> unique users
        item_groups = df.groupby("item_id")["user_id"].unique()
        self._item_to_users = {int(item_id): users.astype(np.int64) for item_id, users in item_groups.items()}

        # user -> ordered items for history (prefer timestamp ordering if present)
        if "timestamp" in df.columns:
            df_sorted = df.sort_values(by="timestamp")
        else:
            df_sorted = df

        user_groups = df_sorted.groupby("user_id")["item_id"].apply(lambda x: x.values.astype(np.int64))
        self._user_to_items = {int(user_id): items for user_id, items in user_groups.items()}

    def _construct_expanded_users(self):
        """
        Construct and cache expansion structure.

        Override this method to implement custom expansion logic (clustering,
        periodic refresh, etc.).

        If `expanded_users` / `history_item_ids` were provided at init time,
        they are used directly; otherwise we compute them from train interactions.

        Future generalization note (not implemented yet):
          - refresh expanded users every X rounds.
        """
        rng = np.random.default_rng(self.expansion_seed)

        num_clients = self.num_clients

        for client_id in range(num_clients):
            # Determine the (single) user id for this client (CrossDevice splitting)
            users_in_client = self.interaction_dataset.client_partition.client_ids_2_user_ids[client_id]
            if len(users_in_client) == 0:
                continue
            user_id = int(users_in_client[0])

            # History item ids
            if self._provided_history_item_ids is not None and client_id in self._provided_history_item_ids:
                hist = np.asarray(self._provided_history_item_ids[client_id], dtype=np.int64)
                hist = hist[: self.history_len]
                if hist.shape[0] < self.history_len:
                    hist = np.pad(hist, (0, self.history_len - hist.shape[0]), constant_values=0)
            else:
                items = self._user_to_items.get(user_id, np.array([], dtype=np.int64))
                hist = items[-self.history_len :]
                if hist.shape[0] < self.history_len:
                    hist = np.pad(hist, (0, self.history_len - hist.shape[0]), constant_values=0)

            # Neighbor user ids per history position
            if self._provided_neighbor_user_ids is not None and client_id in self._provided_neighbor_user_ids:
                neigh_ids = np.asarray(self._provided_neighbor_user_ids[client_id], dtype=np.int64)
                # Expect shape (L,N); pad/crop as needed
                if neigh_ids.ndim != 2:
                    raise ValueError("expanded_users[client_id] must have shape (L,N)")
                neigh_ids = neigh_ids[: self.history_len, : self.neighbor_len]
                if neigh_ids.shape[0] < self.history_len:
                    pad0 = self.history_len - neigh_ids.shape[0]
                    neigh_ids = np.pad(neigh_ids, ((0, pad0), (0, 0)), constant_values=-1)
                if neigh_ids.shape[1] < self.neighbor_len:
                    pad1 = self.neighbor_len - neigh_ids.shape[1]
                    neigh_ids = np.pad(neigh_ids, ((0, 0), (0, pad1)), constant_values=-1)
            else:
                neigh_ids = np.full((self.history_len, self.neighbor_len), -1, dtype=np.int64)
                for ell in range(self.history_len):
                    item_id = int(hist[ell])
                    if item_id == 0:
                        continue
                    cand_users = self._item_to_users.get(item_id)
                    if cand_users is None or cand_users.size == 0:
                        continue
                    # Exclude self
                    cand_users = cand_users[cand_users != user_id]
                    if cand_users.size == 0:
                        continue
                    # Deterministic choice: shuffle with seed for reproducibility, then take first N
                    # (If you prefer pure deterministic without shuffle, remove permutation.)
                    perm = rng.permutation(cand_users.size)
                    chosen = cand_users[perm][: self.neighbor_len]
                    neigh_ids[ell, : chosen.size] = chosen

            self._expansion_cache[client_id] = {
                "history_item_ids": hist.astype(np.int64),
                "neighbor_user_ids": neigh_ids.astype(np.int64),
            }

    def _post_selection_callback(self):
        if self.selected_client_ids is None:
            return

        self._materialize_expanded_user_data(self.selected_client_ids)

    def _on_round_end_server(self):
        if self.evaluator is None:
            return
        if self.validation_rounds_interval <= 0:
            return
        current_round = int(self.current_round or 0)
        if current_round % int(self.validation_rounds_interval) != 0:
            return
        self._materialize_expanded_user_data(list(range(self.num_clients)))

    def _materialize_expanded_user_data(self, client_ids: List[int]):
        """
        Materialize dense expansion payload (Option A) for a list of clients.

        Writes into:
          self.extra_payload["expanded_user_data"][client_id] = {
              "history_item_ids": (L,),
              "neighbor_embeddings": (L,N,H),
          }
        """
        # Ensure payload container exists
        if self.extra_payload is None:
            self.extra_payload = {}
        expanded = self.extra_payload.setdefault("expanded_user_data", {})

        E = self.global_parameters["user_embeddings"]  # (U,H) numpy
        H = self.hidden_dim
        L = self.history_len
        N = self.neighbor_len

        # WarmupA: all-zero neighbors for early rounds
        current_round = int(self.current_round or 0)
        use_neighbors = current_round >= self.warmup_rounds

        for client_id in client_ids:
            cache = self._expansion_cache.get(client_id)
            if cache is None:
                raise KeyError(f"Missing expansion cache for client_id={client_id}. Did _before_training run?")

            hist = cache["history_item_ids"]  # (L,)
            neigh_ids = cache["neighbor_user_ids"]  # (L,N) with -1 padding

            if not use_neighbors:
                neigh_emb = np.zeros((L, N, H), dtype=np.float32)
            else:
                neigh_emb = np.zeros((L, N, H), dtype=np.float32)
                valid = neigh_ids >= 0
                if valid.any():
                    flat_ids = neigh_ids[valid].astype(np.int64)
                    neigh_emb[valid] = E[flat_ids]

            expanded[client_id] = {
                "history_item_ids": hist,
                "neighbor_embeddings": neigh_emb,
                "neighbor_user_ids": neigh_ids,
            }

    def _aggregate_user_embeddings(
        self,
        training_responses: List[ClientTrainingResponse],
    ) -> np.ndarray:
        """
        Produce updated canonical user_embeddings table from sparse client updates.
        When user_embeddings_aggregate=False: scatter each client's (user_ids, embeddings).
        When user_embeddings_aggregate=True: group by user_id, call configured aggregator per user, then scatter.
        """
        table = np.array(self.global_parameters["user_embeddings"], copy=True)
        param_io = self.param_ios["user_embeddings"]
        agg = self.aggregation_strategies["user_embeddings"]

        # Collect sparse updates: list of (client_id, user_ids, embeddings) aligned with training_responses
        updates_by_client: List[tuple] = []
        for resp in training_responses:
            pkg = resp.updates_packages.get("user_embeddings")
            if pkg is None:
                continue
            val = pkg.param_value
            if isinstance(val, dict) and "user_ids" in val and "embeddings" in val:
                uid = np.asarray(val["user_ids"], dtype=np.int64)
                emb = np.asarray(val["embeddings"], dtype=np.float32)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                updates_by_client.append((resp.client_id, uid, emb))
            else:
                continue

        if not updates_by_client:
            return table

        # Simple scatter when user-level aggregation is disabled.
        if not self.user_embeddings_aggregate:
            for (client_id, user_ids, embeddings) in updates_by_client:
                for i, u in enumerate(user_ids):
                    if 0 <= u < table.shape[0]:
                        table[int(u)] = embeddings[i]
            return table

        # Group by user_id: user_id -> list of (client_id, embedding_row)
        per_user: Dict[int, List[tuple]] = defaultdict(list)
        for (client_id, user_ids, embeddings) in updates_by_client:
            for i, u in enumerate(user_ids):
                u_int = int(u)
                if 0 <= u_int < table.shape[0]:
                    per_user[u_int].append(
                        (client_id, np.asarray(embeddings[i], dtype=np.float32))
                    )

        stats = self.current_round_statistics
        for user_id, contribs in per_user.items():
            if len(contribs) == 0:
                continue
            client_ids = [c[0] for c in contribs]
            prev_row = np.asarray(table[user_id], dtype=np.float32)
            round_info_user = RoundStatistics(
                current_round=stats.current_round,
                selected=client_ids,
                responded=client_ids,
                dropped=[],
                num_training_examples={cid: stats.num_training_examples.get(cid, 1) for cid in client_ids},
                training_losses={cid: stats.training_losses.get(cid, []) for cid in client_ids},
                training_durations={cid: stats.training_durations.get(cid, 0.0) for cid in client_ids},
            )
            updates_user = [c[1] for c in contribs]
            aggregated_row = agg.aggregate(
                round_info_user,
                prev_row,
                updates_user,
                param_io,
            )

            if hasattr(aggregated_row, "detach"):
                aggregated_row = aggregated_row.detach().cpu().numpy()
            else:
                aggregated_row = np.asarray(aggregated_row, dtype=np.float32)
            table[user_id] = aggregated_row

        return table

    def aggregate_parameters(
        self,
        training_responses: List[ClientTrainingResponse],
    ):
        """Override to handle user_embeddings with sparse updates and per-user aggregation."""
        for param_name, param_io in self.param_ios.items():
            if param_io.exchange is None or param_io.exchange == "none":
                continue
            if param_name == "user_embeddings":
                updated_table = self._aggregate_user_embeddings(training_responses)
                self._apply_aggregated_result(
                    param_name=param_name,
                    aggregated_result=updated_table,
                    param_io=param_io,
                    selected_client_ids=self.current_round_statistics.selected,
                )
                continue
            # Default path for other params
            _convert = getattr(
                super(),
                "_BaseFederatedServer__convert_to_framework_compatible",
            )
            if param_io.residency == "server":
                previous_parameters = self.global_parameters[param_name]
            else:
                previous_parameters = [
                    self.client_states[cid].local_parameters[param_name]
                    for cid in self.current_round_statistics.selected
                ]
                previous_parameters = _convert(previous_parameters)
            updates = [
                r.updates_packages[param_name].param_value
                for r in training_responses
            ]
            updates = _convert(updates)
            aggregated_results = self.aggregation_strategies[param_name].aggregate(
                self.current_round_statistics,
                previous_parameters,
                updates,
                param_io,
            )
            self._apply_aggregated_result(
                param_name=param_name,
                aggregated_result=aggregated_results,
                param_io=param_io,
                selected_client_ids=self.current_round_statistics.selected,
            )

    def _apply_aggregated_result_custom(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: List[int],
    ):
        """Apply aggregated user_embeddings table to global_parameters."""
        if param_name == "user_embeddings":
            self.global_parameters["user_embeddings"] = aggregated_result
            return
        raise NotImplementedError(
            f"Custom apply not implemented for parameter '{param_name}'"
        )


