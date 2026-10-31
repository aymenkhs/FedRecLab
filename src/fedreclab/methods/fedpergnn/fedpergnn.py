from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


class FedPerGNN(BaseFederatedServer):
    """Federated GNN personalization (Wu et al., 2022).

    Simulation shortcuts:
      The server holds a canonical user-embedding table used as an expansion
      mirror (neighbour lookup). Each client's own embedding lives in
      ClientState and is uploaded every round so the table stays a copy of
      last-round owner rows plus neighbour proposals. user_embeddings_aggregate
      chooses how neighbour proposals for a row are combined into that mirror;
      it does not write back into the owner's ClientState.
      neighbor_embeddings_trainable still lets other clients propose edits to
      your row in the mirror; those edits no longer become your next local
      starting point.
    """
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
            # Expansion mirror of user embeddings. apply_to="custom" scatters
            # (or per-user aggregates) sparse client uploads; owners train from ClientState.
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

    def _parse_user_embedding_update(
        self,
        client_id: int,
        value: Any,
        n_users: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse one client's sparse user-embedding upload.

        Parameters
        ----------
        client_id : int
            Sender, used in error messages.
        value : any
            Expected ``{"user_ids": array, "embeddings": array}``.
        n_users : int
            Rows in the server embedding table.

        Returns
        -------
        user_ids : ndarray of int64, shape (k,)
            Uploaded user indices.
        embeddings : ndarray of float32, shape (k, hidden_dim)
            Rows aligned with ``user_ids``.

        Raises
        ------
        ValueError
            If ``value`` is not a dict with ``user_ids`` and ``embeddings``.
        IndexError
            If any uploaded id is outside ``[0, n_users)``.

        Examples
        --------
        >>> ids, emb = server._parse_user_embedding_update(
        ...     0, {"user_ids": np.array([1]), "embeddings": np.zeros(8)}, 3
        ... )
        >>> ids.tolist()
        [1]
        """
        if not (isinstance(value, dict) and "user_ids" in value and "embeddings" in value):
            raise ValueError(
                f"Client {client_id} user_embeddings update must be a dict "
                f"with 'user_ids' and 'embeddings', got "
                f"{type(value).__name__}: {value!r}"
            )
        user_ids = np.asarray(value["user_ids"], dtype=np.int64)
        embeddings = np.asarray(value["embeddings"], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        for u in user_ids:
            u_int = int(u)
            if u_int < 0 or u_int >= n_users:
                raise IndexError(
                    f"Client {client_id} uploaded user_id={u_int} which is "
                    f"outside the embedding table [0, {n_users})"
                )
        return user_ids, embeddings

    def _aggregate_user_embeddings(
        self,
        updates: List[Any],
        participant_ids: List[int],
    ) -> np.ndarray:
        """Produce the updated canonical user_embeddings table.

        When ``user_embeddings_aggregate`` is False, scatter each client's
        ``(user_ids, embeddings)``. When True, group by user id, call the
        configured aggregator per user, then scatter.

        Parameters
        ----------
        updates : list
            Sparse uploads, each ``{"user_ids", "embeddings"}``, aligned with
            ``participant_ids``.
        participant_ids : list of int
            Clients that sent ``user_embeddings`` this round.

        Returns
        -------
        ndarray
            Copy of the server table with uploaded rows applied.

        Raises
        ------
        ValueError
            If an update is not a dict with ``user_ids`` and ``embeddings``.
        IndexError
            If an uploaded user id is outside the table.

        Examples
        --------
        Scatter two rows from one client into a 3-user table::

            table = server._aggregate_user_embeddings(
                [{"user_ids": np.array([0, 1]), "embeddings": np.ones((2, H))}],
                [0],
            )
        """
        table = np.array(self.global_parameters["user_embeddings"], copy=True)
        n_users = int(table.shape[0])
        param_io = self.param_ios["user_embeddings"]
        agg = self.aggregation_strategies["user_embeddings"]

        updates_by_client: List[tuple] = []
        for client_id, value in zip(participant_ids, updates):
            user_ids, embeddings = self._parse_user_embedding_update(
                client_id, value, n_users
            )
            updates_by_client.append((client_id, user_ids, embeddings))

        if not updates_by_client:
            return table

        if not self.user_embeddings_aggregate:
            for (_client_id, user_ids, embeddings) in updates_by_client:
                for i, u in enumerate(user_ids):
                    table[int(u)] = embeddings[i]
            return table

        per_user: Dict[int, List[tuple]] = defaultdict(list)
        for (client_id, user_ids, embeddings) in updates_by_client:
            for i, u in enumerate(user_ids):
                per_user[int(u)].append(
                    (client_id, np.asarray(embeddings[i], dtype=np.float32))
                )

        for user_id, contribs in per_user.items():
            client_ids = [c[0] for c in contribs]
            prev_row = np.asarray(table[user_id], dtype=np.float32)
            aggregated_row = agg.aggregate(
                self._round_statistics_for_participants(client_ids),
                prev_row,
                [c[1] for c in contribs],
                param_io,
            )

            if hasattr(aggregated_row, "detach"):
                aggregated_row = aggregated_row.detach().cpu().numpy()
            else:
                aggregated_row = np.asarray(aggregated_row, dtype=np.float32)
            table[user_id] = aggregated_row

        return table

    def _aggregate_group(
        self,
        param_name: str,
        param_io: ParamIO,
        updates: List[Any],
        participant_ids: List[int],
    ) -> None:
        """Scatter or per-user-aggregate ``user_embeddings``; default otherwise.

        Parameters
        ----------
        param_name : str
            ParamIO group being aggregated.
        param_io : ParamIO
            Declaration for ``param_name``.
        updates : list
            Per-participant values from ``_collect_updates``.
        participant_ids : list of int
            Clients aligned with ``updates``.

        Returns
        -------
        None

        Examples
        --------
        ``model_state`` uses the base path; ``user_embeddings`` does not::

            server._aggregate_group("user_embeddings", param_io, updates, [0])
        """
        if param_name == "user_embeddings":
            updated_table = self._aggregate_user_embeddings(
                updates, participant_ids
            )
            self._apply_aggregated_result(
                param_name=param_name,
                aggregated_result=updated_table,
                param_io=param_io,
                selected_client_ids=participant_ids,
            )
            return
        super()._aggregate_group(param_name, param_io, updates, participant_ids)

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


