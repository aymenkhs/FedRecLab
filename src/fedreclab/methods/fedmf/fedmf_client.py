"""FedMF client: local user-factor updates and item-factor gradient uploads.

Source: Chai, Wang, Chen and Yang, "Secure Federated Matrix Factorization",
IEEE Intelligent Systems 2021, and the authors' ``FedMF_Full.py`` /
``FedMF_Part.py``.

Defaults follow the reference code: one sequential pass over the user's
ratings in stored order, per-rating regularisation, learning rate applied
on the client, no ``1/M`` scaling. Paper alternatives are constructor
flags. Homomorphic encryption is omitted (see the class docstring).
"""

import time
from typing import Any

import numpy as np
import torch

from src.fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from src.fedreclab.federated.utils import assert_skeleton_server_coverage
from src.fedreclab.methods.fedmf.fedmf_model import MatrixFactorization

_LOCAL_UPDATES = ("sgd_sequential", "full_gradient")
_U_REGS = ("per_rating", "once")
_V_REGS = ("per_rating", "once_rated", "none")
_LR_PLACEMENTS = ("client", "server")
_INIT_MODES = ("constant", "normal")
_LOSS_SCALES = ("none", "global_M")
_ITEM_WEIGHT = "item_embeddings.weight"


class FedMFClientAlgorithm(BaseClientAlgorithm):
    """Local training and scoring for plaintext FedMF.

    Each client holds its users' factor vectors. The server item table is
    loaded from the payload each call. The uploaded group is a gradient
    for ``item_embeddings``; FullText vs PartText is
    ``request.param_ios["item_embeddings"].layout``, not a client flag.

    Parameters
    ----------
    latent_dim : int, optional
        Width of each factor vector. Default is 100.
    lr : float, optional
        Step size ``γ``. Default is ``1e-3``.
    reg_u : float, optional
        User-factor regularisation weight. Default is ``1e-4``.
    reg_v : float, optional
        Item-factor regularisation weight. Default is ``1e-4``.
    local_epochs : int, optional
        Number of local passes per round. Default is 1.
    local_update : {"sgd_sequential", "full_gradient"}, optional
        ``"sgd_sequential"`` (default) is the reference ``user_update``:
        one pass in stored order, ``u`` is updated after each rating, the
        item gradient uses the already-updated ``u``, and item rows are
        assigned. ``"full_gradient"`` is Algorithm 1: both gradients at
        ``(u^{t-1}, V^{t-1})``, contributions summed.
    u_regularisation : {"per_rating", "once"}, optional
        ``"per_rating"`` (default) adds ``2 λ u`` at every rating.
        ``"once"`` adds it once per user step.
    v_regularisation : {"per_rating", "once_rated", "none"}, optional
        ``"per_rating"`` (default) adds ``2 μ v_j`` at every rating of
        ``j``. ``"once_rated"`` adds it once per rated item.
        ``"none"`` is the data term only.
    lr_placement : {"client", "server"}, optional
        ``"client"`` (default) multiplies the uploaded item gradient by
        ``γ``. ``"server"`` uploads the unscaled gradient; the caller
        must set ``server.server_learning_rate`` to ``γ``. Local ``u``
        updates always use ``lr``.
    init_mode : {"constant", "normal"}, optional
        User-factor initialisation. Default is ``"constant"``.
    init_value : float, optional
        Fill value for ``init_mode="constant"``. Default is 0.01.
    init_std : float, optional
        Standard deviation for ``init_mode="normal"``. Default is 0.01.
    init_seed : int or None, optional
        Seed for ``init_mode="normal"``. The server copies its ``seed``
        here when unset. Required for normal initialisation.
    loss_scale : {"none", "global_M"}, optional
        ``"none"`` (default) leaves the data term unscaled.
        ``"global_M"`` divides the data term by
        ``extra_payload["n_train_ratings"]`` and leaves ``λ``/``μ``
        unscaled.

    Raises
    ------
    ValueError
        If a flag is unknown or ``local_epochs`` is not a positive int.

    Examples
    --------
    >>> client = FedMFClientAlgorithm()
    >>> client.local_update, client.loss_scale, client.lr
    ('sgd_sequential', 'none', 0.001)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 1e-3,
        reg_u: float = 1e-4,
        reg_v: float = 1e-4,
        local_epochs: int = 1,
        local_update: str = "sgd_sequential",
        u_regularisation: str = "per_rating",
        v_regularisation: str = "per_rating",
        lr_placement: str = "client",
        init_mode: str = "constant",
        init_value: float = 0.01,
        init_std: float = 0.01,
        init_seed: int | None = None,
        loss_scale: str = "none",
    ):
        super().__init__()
        if local_update not in _LOCAL_UPDATES:
            raise ValueError(
                f"local_update must be one of {_LOCAL_UPDATES}, got {local_update!r}"
            )
        if u_regularisation not in _U_REGS:
            raise ValueError(
                f"u_regularisation must be one of {_U_REGS}, got {u_regularisation!r}"
            )
        if v_regularisation not in _V_REGS:
            raise ValueError(
                f"v_regularisation must be one of {_V_REGS}, got {v_regularisation!r}"
            )
        if lr_placement not in _LR_PLACEMENTS:
            raise ValueError(
                f"lr_placement must be one of {_LR_PLACEMENTS}, got {lr_placement!r}"
            )
        if init_mode not in _INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {_INIT_MODES}, got {init_mode!r}"
            )
        if loss_scale not in _LOSS_SCALES:
            raise ValueError(
                f"loss_scale must be one of {_LOSS_SCALES}, got {loss_scale!r}"
            )
        if not isinstance(local_epochs, int) or isinstance(local_epochs, bool) or local_epochs < 1:
            raise ValueError(
                f"local_epochs must be a positive int, got {local_epochs!r}"
            )
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.reg_u = float(reg_u)
        self.reg_v = float(reg_v)
        self.local_epochs = local_epochs
        self.local_update = local_update
        self.u_regularisation = u_regularisation
        self.v_regularisation = v_regularisation
        self.lr_placement = lr_placement
        self.init_mode = init_mode
        self.init_value = float(init_value)
        self.init_std = float(init_std)
        self.init_seed = init_seed
        self.loss_scale = loss_scale
        self.num_items = None
        self._skeleton: MatrixFactorization | None = None

    def init_model_skeleton(self) -> None:
        """Build the shared item-table module once.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If ``num_items`` or ``device`` has not been set.
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
        self._skeleton = MatrixFactorization(
            num_items=self.num_items,
            latent_dim=self.latent_dim,
            init_mode=self.init_mode,
            init_value=self.init_value,
            init_std=self.init_std,
        ).to(self.device)
        server_names = {
            "item_embeddings.weight",
        }
        assert_skeleton_server_coverage(self._skeleton, server_names)

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: Any,
        users_in_client: list[int],
    ):
        """Create this client's user-factor table.

        Parameters
        ----------
        client_id : int
            Client being initialised.
        total_num_users : int
            Unused; present for the base signature.
        num_users : int
            Number of users on this client.
        num_items : int
            Catalogue size after remapping.
        client_train_data_view : any
            Unused at init; ratings are required at train time.
        users_in_client : list of int
            Remapped user ids owned by this client.

        Returns
        -------
        dict
            ``user_embeddings`` (shape ``(num_users, latent_dim)``) and
            ``user_id_to_local``.

        Raises
        ------
        ValueError
            If ``init_mode="normal"`` and ``init_seed`` is None.
        """
        if self.num_items is None:
            self.num_items = num_items
        self.init_model_skeleton()

        table = torch.empty(num_users, self.latent_dim)
        if self.init_mode == "constant":
            table.fill_(self.init_value)
        else:
            if self.init_seed is None:
                raise ValueError(
                    "init_mode='normal' requires a seed; pass init_seed on "
                    "the client or seed= on the server."
                )
            generator = torch.Generator()
            generator.manual_seed(int(self.init_seed) + int(client_id) + 1)
            table.normal_(0.0, self.init_std, generator=generator)

        user_id_to_local = {int(uid): idx for idx, uid in enumerate(users_in_client)}
        return {
            "user_embeddings": table,
            "user_id_to_local": user_id_to_local,
        }

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """Update local user factors and upload an item-factor gradient.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Payload, client state, training rows, and effective ParamIO.

        Returns
        -------
        ClientTrainingResponse
            Updated user factors and an ``item_embeddings`` gradient
            package (dense table or ``{rows, values}``).

        Raises
        ------
        ValueError
            If the training table has no ``rating`` column, a gradient is
            non-finite, or ``loss_scale="global_M"`` is missing ``M``.
        """
        t0 = time.time()
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device

        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data
        user_id_to_local = dict(client_state.local_parameters["user_id_to_local"])

        if "rating" not in training_data.columns:
            raise ValueError(
                "FedMF requires a 'rating' column; implicit all-ones "
                "targets are not used."
            )

        model.load_server_weights(training_request.global_payload)
        item_table = model.item_embeddings.weight.detach()
        user_table = torch.as_tensor(
            client_state.local_parameters["user_embeddings"],
            device=device,
            dtype=item_table.dtype,
        ).clone()

        data_scale = self._data_scale(training_request)
        all_losses: list[float] = []
        num_training_examples = 0
        grad_total = torch.zeros_like(item_table)
        rated_items: set[int] = set()

        for _ in range(self.local_epochs):
            epoch_sq = []
            epoch_grad = torch.zeros_like(item_table)
            n_this = 0
            if len(training_data) > 0:
                grouped = training_data.groupby("user_id", sort=False)
                for user_id, group in grouped:
                    local_uid = user_id_to_local[int(user_id)]
                    u = user_table[local_uid]
                    items = group["item_id"].to_numpy()
                    ratings = group["rating"].to_numpy(dtype=np.float64)
                    if self.local_update == "sgd_sequential":
                        u, item_grad, sq, touched = self._sgd_sequential_user(
                            u, item_table, items, ratings, data_scale
                        )
                    else:
                        u, item_grad, sq, touched = self._full_gradient_user(
                            u, item_table, items, ratings, data_scale
                        )
                    user_table[local_uid] = u
                    epoch_grad = epoch_grad + item_grad
                    epoch_sq.extend(sq)
                    n_this += len(items)
                    rated_items.update(touched)
            grad_total = grad_total + epoch_grad
            num_training_examples += n_this
            if epoch_sq:
                all_losses.append(float(np.mean(epoch_sq)))
            self._require_finite_tensor(user_table, "user factors")
            self._require_finite_tensor(grad_total, "item-factor gradient")

        upload = self._item_gradient_upload(
            training_request, grad_total, rated_items
        )
        updates_packages = {
            "item_embeddings": self._update_package(
                training_request, "item_embeddings", upload
            )
        }
        updated_state = ClientStateBase(
            client_id=client_id,
            local_parameters={
                "user_embeddings": user_table.detach().cpu().clone(),
                "user_id_to_local": user_id_to_local,
            },
            optimizer_state=None,
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
        candidate_items_per_user: dict[int, np.ndarray],
        global_parameters: dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: dict[str, Any],
    ) -> dict[int, np.ndarray]:
        """Score candidates as inner products ``⟨u_i, v_j⟩``.

        See ``BaseClientAlgorithm.compute_scores`` for the ranking
        contract. This method returns scores only.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Server item table for this round.
        client_state : ClientStateBase
            Holds the client-resident user-factor table.
        extra_payload : dict
            Unused by FedMF scoring.

        Returns
        -------
        dict of int to ndarray
            Scores per user, aligned with that user's candidates.

        Examples
        --------
        >>> scores = client.compute_scores(0, {0: np.array([1, 2])}, payload, state, {})  # doctest: +SKIP
        >>> scores[0].shape  # doctest: +SKIP
        (2,)
        """
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device
        user_id_to_local = client_state.local_parameters["user_id_to_local"]
        model.load_server_weights(global_parameters)
        user_table = torch.as_tensor(
            client_state.local_parameters["user_embeddings"],
            device=device,
            dtype=model.item_embeddings.weight.dtype,
        )

        scores: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                local_uid = user_id_to_local[int(user_id)]
                n_items = len(candidate_items)
                user_vectors = user_table[local_uid].unsqueeze(0).expand(n_items, -1)
                item_ids = torch.as_tensor(
                    candidate_items, dtype=torch.long, device=device
                )
                scores[int(user_id)] = model(user_vectors, item_ids).detach().cpu().numpy()
        return scores

    def _data_scale(self, training_request: ClientTrainingRequest) -> float:
        """Return the multiplier for the squared-error data term.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Read for ``extra_payload["n_train_ratings"]`` when
            ``loss_scale="global_M"``.

        Returns
        -------
        float
            ``1.0`` or ``1/M``.

        Raises
        ------
        ValueError
            If ``global_M`` is set and ``M`` is missing or not positive.
        """
        if self.loss_scale == "none":
            return 1.0
        extra = training_request.extra_payload or {}
        m = extra.get("n_train_ratings")
        if m is None or int(m) <= 0:
            raise ValueError(
                "loss_scale='global_M' requires extra_payload['n_train_ratings'] "
                "to be a positive integer (training-split size)."
            )
        return 1.0 / float(m)

    def _sgd_sequential_user(
        self,
        u: torch.Tensor,
        item_table: torch.Tensor,
        items: np.ndarray,
        ratings: np.ndarray,
        data_scale: float,
    ):
        """Reference-code pass: assign per rated item, update ``u`` in place.

        Parameters
        ----------
        u : torch.Tensor
            Current user vector, shape ``(d,)``.
        item_table : torch.Tensor
            Downloaded ``V``, shape ``(n_items, d)``. Not written.
        items : ndarray
            Item ids in stored order.
        ratings : ndarray
            Ratings aligned with ``items``.
        data_scale : float
            Multiplier for the data term.

        Returns
        -------
        u : torch.Tensor
            Updated user vector.
        item_grad : torch.Tensor
            Assigned item gradients, shape ``(n_items, d)``.
        sq_errors : list of float
            Squared residuals at each rating, before the ``u`` step.
        touched : set of int
            Rated item ids.
        """
        item_grad = torch.zeros_like(item_table)
        sq_errors: list[float] = []
        touched: set[int] = set()
        u = u.clone()
        for item_id, rate in zip(items, ratings):
            j = int(item_id)
            v = item_table[j]
            error = float(rate) - torch.dot(u, v)
            sq_errors.append(float(error.detach().cpu() ** 2))
            grad_u = data_scale * (-2.0 * error * v)
            if self.u_regularisation == "per_rating":
                grad_u = grad_u + 2.0 * self.reg_u * u
            u = u - self.lr * grad_u
            grad_v = data_scale * (-2.0 * error * u)
            if self.v_regularisation != "none":
                grad_v = grad_v + 2.0 * self.reg_v * v
            if self.lr_placement == "client":
                grad_v = self.lr * grad_v
            item_grad[j] = grad_v
            touched.add(j)
        if self.u_regularisation == "once":
            u = u - self.lr * (2.0 * self.reg_u * u)
        return u, item_grad, sq_errors, touched

    def _full_gradient_user(
        self,
        u: torch.Tensor,
        item_table: torch.Tensor,
        items: np.ndarray,
        ratings: np.ndarray,
        data_scale: float,
    ):
        """Paper full-gradient step at ``(u^{t-1}, V^{t-1})``.

        Parameters
        ----------
        u : torch.Tensor
            Current user vector, shape ``(d,)``.
        item_table : torch.Tensor
            Downloaded ``V``, shape ``(n_items, d)``.
        items : ndarray
            Item ids.
        ratings : ndarray
            Ratings aligned with ``items``.
        data_scale : float
            Multiplier for the data term.

        Returns
        -------
        u : torch.Tensor
            Updated user vector.
        item_grad : torch.Tensor
            Accumulated item gradients, shape ``(n_items, d)``.
        sq_errors : list of float
            Squared residuals at ``(u^{t-1}, V)``.
        touched : set of int
            Rated item ids.
        """
        if len(items) == 0:
            return u, torch.zeros_like(item_table), [], set()
        item_ids = torch.as_tensor(items, dtype=torch.long, device=u.device)
        rates = torch.as_tensor(ratings, dtype=u.dtype, device=u.device)
        v_rows = item_table[item_ids]
        preds = (u * v_rows).sum(dim=-1)
        errors = rates - preds
        sq_errors = [float(x) for x in (errors.detach().cpu() ** 2).tolist()]
        data_u = (data_scale * (-2.0 * errors)).unsqueeze(-1) * v_rows
        data_v = (data_scale * (-2.0 * errors)).unsqueeze(-1) * u
        grad_u = data_u.sum(dim=0)
        if self.u_regularisation == "per_rating":
            grad_u = grad_u + (2.0 * self.reg_u * u) * len(items)
        else:
            grad_u = grad_u + 2.0 * self.reg_u * u
        item_grad = torch.zeros_like(item_table)
        item_grad.index_add_(0, item_ids, data_v)
        if self.v_regularisation == "per_rating":
            item_grad.index_add_(0, item_ids, (2.0 * self.reg_v) * v_rows)
        elif self.v_regularisation == "once_rated":
            unique = torch.unique(item_ids)
            item_grad[unique] = item_grad[unique] + 2.0 * self.reg_v * item_table[unique]
        if self.lr_placement == "client":
            item_grad = self.lr * item_grad
        u = u - self.lr * grad_u
        touched = {int(j) for j in items}
        return u, item_grad, sq_errors, touched

    def _item_gradient_upload(
        self,
        training_request: ClientTrainingRequest,
        dense_grad: torch.Tensor,
        rated_items: set[int],
    ):
        """Dense table or sparse ``{rows, values}`` from the effective layout.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Read for ``param_ios["item_embeddings"].layout``.
        dense_grad : torch.Tensor
            Full item-gradient table, shape ``(n_items, d)``.
        rated_items : set of int
            Item ids this client rated (PartText support).

        Returns
        -------
        dict
            Either ``{item_embeddings.weight: table}`` or
            ``{"rows": ids, "values": {item_embeddings.weight: rows}}``.
        """
        param_ios = training_request.param_ios or {}
        layout = getattr(param_ios.get("item_embeddings"), "layout", "rows")
        table = dense_grad.detach()
        if layout == "dense":
            return {_ITEM_WEIGHT: table.cpu().clone()}
        rows = np.array(sorted(rated_items), dtype=np.int64)
        if rows.size == 0:
            values = table.new_zeros((0, table.shape[1]))
        else:
            idx = torch.as_tensor(rows, dtype=torch.long, device=table.device)
            values = table[idx]
        return {
            "rows": rows,
            "values": {_ITEM_WEIGHT: values.detach().cpu().clone()},
        }

    @staticmethod
    def _require_finite_tensor(tensor: torch.Tensor, where: str) -> None:
        """Raise if ``tensor`` contains NaN or Inf.

        Parameters
        ----------
        tensor : torch.Tensor
            Values to check.
        where : str
            Label used in the error message.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any value is non-finite.
        """
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in {where}")
