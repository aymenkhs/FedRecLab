from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel

# Set on the client. The server copies the three widths; the flag is
# client-only because only local Adam consults it.
_CLIENT_OWNED_SETTINGS = (
    "hidden_dim",
    "history_len",
    "neighbor_len",
    "neighbor_embeddings_trainable",
)
_ARCHITECTURE_ATTRS = ("hidden_dim", "history_len", "neighbor_len")


def default_param_ios() -> dict[str, ParamIO]:
    # TODO: switch model_state, item_embeddings and user_embeddings to
    # exchange="gradient" or "delta" so the server applies
    # Theta <- Theta - alpha * g. Uploads are still parameters.
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

    ``hidden_dim``, ``history_len`` and ``neighbor_len`` are copied from
    ``client_algorithm`` at construction. They are not arguments of this
    class. The client allocates its user embedding before the server
    builds the global table, so the client is the source of the width.
    ``neighbor_embeddings_trainable`` is not stored here. It only decides
    whether that client's Adam treats neighbour rows as parameters. The
    mirror accepts the uploaded rows either way. The anchor user's
    embedding is always trainable on the client; the flag does not cover it.

    Parameters
    ----------
    warmup_rounds : int, optional
        Rounds during which neighbour embeddings in the expansion payload
        are zeros. Default is 1.
    expanded_users : dict of int to ndarray or None, optional
        Caller-supplied neighbour user ids, keyed by client id, shape
        ``(history_len, neighbor_len)``. Empty slots are -1. Default is
        None, which samples neighbours from the training interactions.
    history_item_ids : dict of int to ndarray or None, optional
        Caller-supplied history catalogue ids, keyed by client id.
        Default is None, which builds each history from that user's
        training items.
    expansion_seed : int or None, optional
        Seed for neighbour sampling and the initial user table. If None,
        the server ``seed`` is used. Default is None.
    **kwargs
        Arguments of :class:`BaseFederatedServer`. ``client_algorithm``
        must expose ``hidden_dim``, ``history_len`` and ``neighbor_len``.

    Raises
    ------
    TypeError
        If ``hidden_dim``, ``history_len``, ``neighbor_len`` or
        ``neighbor_embeddings_trainable`` are passed here, or if
        ``client_algorithm`` does not expose the three widths.

    Simulation shortcuts:
      The server holds a canonical user-embedding table used as an expansion
      mirror (neighbour lookup). Each client's own embedding lives in
      ClientState and is uploaded every round. When the client sets
      ``neighbor_embeddings_trainable``, other clients can still propose
      edits to your row in the mirror; those edits no longer become your
      next local starting point. A trainable neighbour proposal is that
      row after the same local Adam steps as the proposer's owner
      embedding. Plaintext sparse item uploads reveal which items this
      client trained on.

    Deviations from Wu et al.:
      The rating predictor is the SI inner product of the user hidden
      state and the target item embedding. The public Keras repo's
      concat-plus-``Dense(1, sigmoid)`` head is not used: that form is
      additive, so ranking inside a user ignores the GNN state.

      ``FedPerGNNClientAlgorithm.task="rating"`` (default) keeps the
      paper loss (MSE on observed ratings, no negatives). The library
      evaluator is still top-K ranking, not RMSE. Record ``task`` with
      every result (copied onto ``extra_payload["task"]``).

      ``task="ranking"`` is the library ranking protocol (binary labels,
      sampled negatives, BCE-with-logits), not the paper. Paper
      pseudo-interacted items are synthetic gradient rows for privacy.
      They are appended to the item upload by ``apply_privacy`` and
      never enter the loss; they are not ranking negatives.

      Privacy defaults are off (``num_pseudo_items=0``, no clip, no
      Laplace noise). Wu et al. use M=1000, an L1 clip of 0.1 and
      Laplace scale 0.2. Those knobs are opt-in while the groups are
      still exchanged as parameters: the protected step is written
      back into the parameter rows, and the server averages those
      rows. The paper instead averages gradients and applies
      Theta <- Theta - alpha * g. See the TODO on ``default_param_ios``.
      Pseudo gradients use a diagonal Gaussian, the per-coordinate
      mean and standard deviation of the observed-item deltas. A full
      covariance is singular when a user has fewer interactions than
      the embedding width.

    Empty history slots are -1, not catalogue id 0. Id 0 stays in the
    history, expands to neighbours, and is uploaded only when this
    client trains it or lists it in that history. The module keeps a
    padding row at embedding index 0; that row is not part of the
    exchanged item table. Catalogue item ``i`` is embedding row ``i + 1``.

    Neighbour expansion and the initial user table use ``expansion_seed`` if
    set, otherwise the server ``seed``. Both default to None (unseeded).
    """
    def __init__(
        self,
        *,
        warmup_rounds: int = 1,
        expanded_users: Optional[Dict[int, np.ndarray]] = None,
        history_item_ids: Optional[Dict[int, np.ndarray]] = None,
        expansion_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        misplaced = [name for name in _CLIENT_OWNED_SETTINGS if name in kwargs]
        if misplaced:
            names = ", ".join(misplaced)
            raise TypeError(
                f"{names} are set on FedPerGNNClientAlgorithm, not on "
                "FedPerGNN. The server copies hidden_dim, history_len and "
                "neighbor_len from client_algorithm. "
                "neighbor_embeddings_trainable is client-only and applies "
                "only to neighbour rows; the anchor user embedding is "
                "always trainable."
            )
        client = kwargs.get("client_algorithm")
        if client is not None:
            missing = [name for name in _ARCHITECTURE_ATTRS if not hasattr(client, name)]
            if missing:
                names = ", ".join(missing)
                raise TypeError(
                    f"{type(client).__name__} is missing {names}. FedPerGNN "
                    "reads hidden_dim, history_len and neighbor_len from "
                    "client_algorithm."
                )
            self.hidden_dim = client.hidden_dim
            self.history_len = client.history_len
            self.neighbor_len = client.neighbor_len
        self.warmup_rounds = warmup_rounds

        # Optional externally-provided expansion structure
        self._provided_neighbor_user_ids = expanded_users
        self._provided_history_item_ids = history_item_ids
        self.expansion_seed = expansion_seed

        # Expansion cache (integer structure only):
        # client_id -> {"history_item_ids": (L,) with -1 padding,
        #               "neighbor_user_ids": (L, N) with -1 padding}
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
        task = getattr(self.client_algorithm, "task", None)
        if task is not None:
            self.extra_payload["task"] = task

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

    def _pad_history(self, item_ids: np.ndarray) -> np.ndarray:
        """Right-pad catalogue ids with -1 up to ``history_len``.

        Parameters
        ----------
        item_ids : ndarray
            Catalogue ids in history order. Id 0 is a real item. Longer
            inputs keep the last ``history_len`` ids.

        Returns
        -------
        ndarray of int64
            Shape ``(history_len,)``. Entries equal to -1 are empty slots.

        Raises
        ------
        ValueError
            If any id is negative. Negative values are reserved for the
            empty-slot marker.

        Examples
        --------
        >>> server = FedPerGNN.__new__(FedPerGNN)
        >>> server.history_len = 3
        >>> server._pad_history(np.array([0, 2])).tolist()
        [0, 2, -1]
        """
        ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        if ids.size > self.history_len:
            ids = ids[-self.history_len :]
        if np.any(ids < 0):
            raise ValueError(
                "history item ids must be catalogue ids >= 0; "
                "negative ids are reserved for empty history slots"
            )
        hist = np.full(self.history_len, -1, dtype=np.int64)
        hist[: ids.size] = ids
        return hist

    def _construct_expanded_users(self):
        """
        Construct and cache expansion structure.

        Override this method to implement custom expansion logic (clustering,
        periodic refresh, etc.).

        Histories built from training interactions keep catalogue id 0 and
        fill unused slots with -1. A caller-supplied ``history_item_ids``
        array is cropped to ``history_len`` and the tail is filled with -1.
        Id 0 in that array is catalogue item 0. Mark an empty slot with
        -1 so it stays aligned with a caller-supplied neighbour array.

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

            # History item ids. -1 is an empty slot; 0 is a catalogue item.
            if self._provided_history_item_ids is not None and client_id in self._provided_history_item_ids:
                supplied = np.asarray(
                    self._provided_history_item_ids[client_id], dtype=np.int64
                ).reshape(-1)
                supplied = supplied[: self.history_len]
                hist = np.full(self.history_len, -1, dtype=np.int64)
                hist[: supplied.shape[0]] = supplied
            else:
                items = self._user_to_items.get(user_id, np.array([], dtype=np.int64))
                hist = self._pad_history(items)

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
                    if item_id < 0:
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
              "history_item_ids": (L,) with -1 in empty slots,
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

            hist = cache["history_item_ids"]  # (L,) with -1 in empty slots
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


