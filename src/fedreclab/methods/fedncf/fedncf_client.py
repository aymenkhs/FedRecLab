"""FedNCF client algorithm -- local training and scoring for Federated NCF.

Source : Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.

ClientState holds only client-resident tensors (the two user embedding tables)
plus a name-keyed optimiser dict. The shared NCF skeleton lives on this
algorithm object and is loaded from the server payload each call.

Training protocol (paper §3.1, §4.1–4.2):
  Observed interactions are implicit positives (label 1) regardless of the
  raw rating. Four negatives (label 0) are sampled per positive. The client
  optimises BCE-with-logits, which is equivalent to the paper's sigmoid
  prediction head plus BCE. Ranking uses the raw scores (logits under
  BCE); order matches sigmoid probabilities.

  ``label_mode="rating"`` with ``loss="mse"`` is a documented deviation:
  positives keep the raw rating and negatives stay 0. That is explicit
  rating prediction under a ranking evaluator, not the KBS protocol.

The prediction layer is part of N: loaded from the server payload,
trained, and uploaded. It is not re-derived each round. See
``fedncf_model`` (module docstring) for why dummy GMF/MLP heads are
omitted.

optimizer_state_scope:
  - "client_resident" (default): persist Adam moments for user tables only
  - "none": no optimiser state is kept
  - "all": persist moments for every assembled parameter (memory-heavy)

item_upload_include_negatives:
  Default False follows Algorithm 2 (P = O): only positive item ids are
  uploaded when item_embeddings layout is "rows". True includes negatives
  sampled during local training. §3.5.2's blurring argument relies on
  negative-item updates being visible to the server.
"""

import logging
import time
from typing import List, Dict, Tuple, Union, Any

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
    flatten_grouped_state,
    restore_named_optimizer_state,
)
from src.fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering

logger = logging.getLogger(__name__)

_CLIENT_TENSOR_NAMES = (
    "gmf_user_embeddings.weight",
    "mlp_user_embeddings.weight",
)
_OPTIMIZER_STATE_SCOPES = ("client_resident", "none", "all")
_LABEL_MODES = ("binary", "rating")
_LOSSES = ("bce", "mse")
_ALL_SCOPE_WARN_BYTES = 1 << 30


class FedNCFClientAlgorithm(BaseClientAlgorithm):
    """Local training and scoring for Federated NCF.

    Defaults follow the paper: implicit binary labels, BCE, four negatives
    per positive. Pass ``label_mode="rating"`` and ``loss="mse"`` for the
    explicit-rating deviation.

    Parameters
    ----------
    predictive_factor : int, optional
        Latent width of the NeuMF skeleton. Default is 32.
    local_epochs : int, optional
        Local SGD epochs per round. Default is 5.
    lr : float, optional
        Adam learning rate. Default is ``5e-4``.
    batch_size : int, optional
        Mini-batch size. Default is 128.
    neg_samples : int, optional
        Unobserved items sampled per positive. Default is 4 (paper §4.2).
    label_mode : {"binary", "rating"}, optional
        ``"binary"`` (paper) sets every observed interaction to 1.
        ``"rating"`` keeps the raw rating as the positive target.
    loss : {"bce", "mse"}, optional
        ``"bce"`` (paper) is BCE-with-logits. ``"mse"`` is squared error
        on the scores (rating regression). ``loss="bce"`` requires
        ``label_mode="binary"``.
    optimizer_state_scope : {"client_resident", "none", "all"}, optional
        Which Adam moments to persist in ``ClientState``. Default is
        ``"client_resident"``.
    item_upload_include_negatives : bool, optional
        If True, row-layout item uploads include negatives sampled this
        round. Default is False (Algorithm 2, P = O).

    Raises
    ------
    ValueError
        If ``optimizer_state_scope``, ``label_mode``, or ``loss`` is
        unknown, or if BCE is paired with rating-valued labels.

    Examples
    --------
    >>> client = FedNCFClientAlgorithm()
    >>> client.label_mode, client.loss, client.neg_samples
    ('binary', 'bce', 4)
    >>> FedNCFClientAlgorithm(label_mode="rating", loss="mse").loss
    'mse'
    """

    def __init__(
        self,
        predictive_factor: int = 32,
        local_epochs: int = 5,
        lr: float = 5e-4,
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
        self.predictive_factor = predictive_factor
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.label_mode = label_mode
        self.loss = loss
        self.optimizer_state_scope = optimizer_state_scope
        self.item_upload_include_negatives = bool(item_upload_include_negatives)
        self.num_items = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._skeleton: NeuralCollaborativeFiltering | None = None
        self._warned_all_scope = False

    def init_model_skeleton(self):
        if self._skeleton is not None:
            return
        if self.num_items is None:
            raise RuntimeError("init_model_skeleton requires num_items to be set")
        self._skeleton = NeuralCollaborativeFiltering(
            num_items=self.num_items,
            predictive_factor=self.predictive_factor,
        ).to(self._device)
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
        if self.num_items is None:
            self.num_items = num_items
        self.init_model_skeleton()
        self._maybe_warn_all_scope(total_num_users)

        dim = 2 * self.predictive_factor
        gmf_user = torch.empty(num_users, dim)
        mlp_user = torch.empty(num_users, dim)
        nn.init.normal_(gmf_user, std=0.01)
        nn.init.normal_(mlp_user, std=0.01)

        user_id_to_local = {uid: idx for idx, uid in enumerate(users_in_client)}
        local_params = {
            "user_embeddings": {
                "gmf_user_embeddings.weight": gmf_user,
                "mlp_user_embeddings.weight": mlp_user,
            },
            "user_id_to_local": user_id_to_local,
        }
        return local_params, None

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        t0 = time.time()
        self.init_model_skeleton()
        model = self._skeleton
        device = self._device

        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data
        user_id_to_local = client_state.local_parameters["user_id_to_local"]

        model.load_server_weights(training_request.global_payload)
        model.train()

        named_params, gmf_user, mlp_user = self._inject_user_parameters(client_state)
        optimizer = torch.optim.Adam(list(model.parameters()) + [gmf_user, mlp_user], lr=self.lr)
        restore_named_optimizer_state(optimizer, named_params, client_state.optimizer_state)

        user_col, item_col = "user_id", "item_id"
        rating_col = "rating" if "rating" in training_data.columns else None

        pos_global_uids = training_data[user_col].values
        pos_item_ids = training_data[item_col].values
        pos_ratings = training_data[rating_col].values if rating_col else np.ones(len(training_data), dtype=np.float32)

        pos_local_uids = np.array([user_id_to_local[u] for u in pos_global_uids])

        user_pos_items: Dict[int, set] = {}
        for local_uid, iid in zip(pos_local_uids, pos_item_ids):
            user_pos_items.setdefault(local_uid, set()).add(iid)

        all_losses: List[float] = []
        num_training_examples = 0
        touched_items = set(int(i) for i in pos_item_ids)
        if training_request.seed is None:
            rng = None
        else:
            rng = np.random.default_rng(
                [training_request.seed, training_request.client_id, training_request.current_round]
            )

        for _ in range(self.local_epochs):
            samples_x, samples_y = self._build_epoch_samples(
                pos_local_uids, pos_item_ids, pos_ratings, user_pos_items, rng,
            )
            num_training_examples += len(samples_x)
            if self.item_upload_include_negatives:
                touched_items.update(int(i) for i in samples_x[:, 1])

            perm = np.random.permutation(len(samples_x)) if rng is None else rng.permutation(len(samples_x))
            samples_x = samples_x[perm]
            samples_y = samples_y[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(samples_x), self.batch_size):
                bx = torch.tensor(samples_x[start:start + self.batch_size], dtype=torch.long, device=device)
                by = torch.tensor(samples_y[start:start + self.batch_size], dtype=torch.float32, device=device)

                preds = model(bx, gmf_user, mlp_user)
                loss = self._compute_loss(preds, by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
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
                "user_embeddings": {
                    "gmf_user_embeddings.weight": gmf_user.detach().cpu().clone(),
                    "mlp_user_embeddings.weight": mlp_user.detach().cpu().clone(),
                },
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
        """Full tables if layout is dense; sparse {rows, values} otherwise."""
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
        """Build one epoch of training samples (positives + negatives).

        For each positive interaction, ``self.neg_samples`` random items that
        the user has not interacted with are sampled as negatives (label 0).
        Positive labels are 1 under ``label_mode="binary"`` and the raw
        rating under ``label_mode="rating"``.

        Parameters
        ----------
        pos_local_uids : ndarray
            Local user indices of observed interactions, shape ``(n_pos,)``.
        pos_item_ids : ndarray
            Item ids of observed interactions, shape ``(n_pos,)``.
        pos_ratings : ndarray
            Raw ratings aligned with the positives. Ignored when
            ``label_mode="binary"``.
        user_pos_items : dict of int to set
            Observed item ids per local user, used to reject sampled
            negatives that are actually positives.
        rng : numpy.random.Generator or None
            Seeded generator. ``None`` falls back to global ``np.random``.

        Returns
        -------
        samples_x : ndarray
            User–item index pairs, shape ``(n_pos * (1 + neg_samples), 2)``.
        samples_y : ndarray
            Training targets, shape ``(n_pos * (1 + neg_samples),)``.

        Examples
        --------
        >>> client = FedNCFClientAlgorithm(neg_samples=0, predictive_factor=4)
        >>> client.num_items = 5
        >>> x, y = client._build_epoch_samples(
        ...     np.array([0]), np.array([1]), np.array([4.0]), {0: {1}}, None)
        >>> y
        array([1.], dtype=float32)
        """
        n_pos = len(pos_local_uids)
        n_total = n_pos * (1 + self.neg_samples)

        samples_x = np.empty((n_total, 2), dtype=np.int64)
        samples_y = np.empty(n_total, dtype=np.float32)

        idx = 0
        for i in range(n_pos):
            local_uid = pos_local_uids[i]
            pos_item = pos_item_ids[i]
            # TODO: once a second implicit-feedback method needs the same
            # transform, binarise on InteractionDataset rather than here.
            pos_label = 1.0 if self.label_mode == "binary" else float(pos_ratings[i])

            samples_x[idx] = [local_uid, pos_item]
            samples_y[idx] = pos_label
            idx += 1

            excl = user_pos_items.get(local_uid, set())
            draw = np.random.randint if rng is None else rng.integers
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
            Model scores, shape ``(batch,)``. Logits under ``loss="bce"``;
            rating regressions under ``loss="mse"``.
        targets : torch.Tensor
            Training labels, shape ``(batch,)``. Under ``loss="bce"`` these
            are 0/1; under ``loss="mse"`` they may be ratings.

        Returns
        -------
        torch.Tensor
            Scalar mean loss.

        Raises
        ------
        RuntimeError
            If ``self.loss`` is not ``"bce"`` or ``"mse"``. The constructor
            already rejects unknown names, so this is a guard.

        Examples
        --------
        >>> client._compute_loss(logits, labels)  # doctest: +SKIP
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
        """Score candidate items with the assembled FedNCF model.

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
            Server-resident NCF groups for this round.
        client_state : ClientStateBase
            Holds the two client-resident user embedding tables.
        extra_payload : dict
            Unused by FedNCF scoring.

        Returns
        -------
        dict of int to ndarray
            Scores per user, aligned with that user's candidates (logits
            under BCE; rating regressions under MSE). Ranking order is
            unchanged by a sigmoid.

        Examples
        --------
        >>> scores = client.compute_scores(0, {0: np.array([1, 2])}, payload, state, {})  # doctest: +SKIP
        >>> scores[0].shape  # doctest: +SKIP
        (2,)
        """
        self.init_model_skeleton()
        model = self._skeleton
        device = self._device
        user_id_to_local = client_state.local_parameters["user_id_to_local"]

        model.load_server_weights(global_parameters)
        _, gmf_user, mlp_user = self._inject_user_parameters(client_state)
        model.eval()

        scores: Dict[int, np.ndarray] = {}

        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                local_uid = user_id_to_local[user_id]
                n_items = len(candidate_items)

                user_ids = torch.full((n_items,), local_uid, dtype=torch.long, device=device)
                item_ids = torch.tensor(candidate_items, dtype=torch.long, device=device)
                x = torch.stack([user_ids, item_ids], dim=1)

                scores[user_id] = model(x, gmf_user, mlp_user).cpu().numpy()

        return scores

    def _inject_user_parameters(self, client_state: ClientStateBase):
        user_state = client_state.local_parameters["user_embeddings"]
        gmf_user = nn.Parameter(
            torch.as_tensor(user_state["gmf_user_embeddings.weight"], device=self._device).clone()
        )
        mlp_user = nn.Parameter(
            torch.as_tensor(user_state["mlp_user_embeddings.weight"], device=self._device).clone()
        )
        named_params = {n: p for n, p in self._skeleton.named_parameters()}
        named_params["gmf_user_embeddings.weight"] = gmf_user
        named_params["mlp_user_embeddings.weight"] = mlp_user
        return named_params, gmf_user, mlp_user

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
