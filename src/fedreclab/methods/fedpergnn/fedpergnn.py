from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


def default_param_ios() -> dict[str, ParamIO]:
    return {
        "model_state": ParamIO(
            residency="server", exchange="params", apply_to="server",
        ),
        "item_embeddings": ParamIO(
            residency="server", exchange="params", apply_to="server", layout="rows",
        ),
        "user_embeddings": ParamIO(
            residency="server", exchange="params", apply_to="server", layout="rows",
        ),
    }


class FedPerGNN(BaseFederatedServer):
    """Federated GNN personalization (Wu et al., 2022).

    Item and user embedding tables are row-layout groups. The configured
    aggregator runs over the clients that uploaded each row (equal mix when
    several neighbour proposals share a row). Owner ClientState is not
    overwritten by the mirror.

    Simulation shortcuts:
      The server holds a canonical user-embedding table used as an expansion
      mirror (neighbour lookup). Each client's own embedding lives in
      ClientState and is uploaded every round. neighbour_embeddings_trainable
      still lets other clients propose edits to your row in the mirror; those
      edits no longer become your next local starting point. Plaintext sparse
      item uploads reveal which items this client trained on.

    Deviations from Wu et al.:
      The rating predictor is the SI inner product of the user hidden
      state and the target item embedding. The public Keras repo's
      concat-plus-``Dense(1, sigmoid)`` head is not used. Training is
      still MSE on observed ratings while the library evaluator reports
      ranking metrics; the paper reports RMSE.

    Neighbour expansion and the initial user table use ``expansion_seed`` if
    set, otherwise the server ``seed``. Both default to None (unseeded).
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        history_len: int = 20,
        neighbor_len: int = 10,
        warmup_rounds: int = 1,
        neighbor_embeddings_trainable: bool = False,
        expanded_users: Optional[Dict[int, np.ndarray]] = None,
        history_item_ids: Optional[Dict[int, np.ndarray]] = None,
        expansion_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len
        self.warmup_rounds = warmup_rounds

        # neighbor_embeddings_trainable applies only to neighbor rows; central user is always trainable.
        self.neighbor_embeddings_trainable = neighbor_embeddings_trainable

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

        super().__init__(param_ios=default_param_ios(), **kwargs)

    def _expansion_rng(self):
        """RNG for expansion sampling and the initial user table.

        ``expansion_seed`` wins if set; otherwise the server ``seed`` is
        used so varying the experiment seed also varies neighbours. If
        both are None, returns ``np.random`` (unseeded).

        Returns
        -------
        numpy.random.Generator or module
            Seeded generator, or ``np.random`` when no seed is set.

        Examples
        --------
        >>> s = FedPerGNN.__new__(FedPerGNN)
        >>> s.expansion_seed, s.seed = None, 0
        >>> s._expansion_rng().integers(10) == np.random.default_rng(0).integers(10)
        True
        """
        seed = self.expansion_seed if self.expansion_seed is not None else self.seed
        if seed is None:
            return np.random
        return np.random.default_rng(seed)

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
        model_state = model.get_model_state()
        numpy_state: Dict[str, Any] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in model_state.items()
        }
        item_embeddings = np.asarray(model.get_item_embeddings(), dtype=np.float32)

        rng = self._expansion_rng()
        user_embeddings = rng.normal(
            0.0, 0.01, size=(self.interaction_dataset.num_users, self.hidden_dim)
        ).astype(np.float32)

        return {
            "model_state": numpy_state,
            "item_embeddings": item_embeddings,
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
        rng = self._expansion_rng()

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


