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
from src.fedreclab.federated.utils import (
    assert_skeleton_server_coverage,
    flatten_grouped_state,
)
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

    Each client holds its users' factor vectors. The uploaded group is a
    gradient for ``item_embeddings``; FullText vs PartText is
    ``request.param_ios["item_embeddings"].layout``, not a client flag.

    ``local_train`` runs on CPU in NumPy float64: per-rating SGD is serial
    in ``u``, so an accelerator only adds launch and sync overhead. It
    gathers only the rows of items the client rated from the payload
    table, and casts uploads back to that table's dtype. ``compute_scores``
    uses the item-table skeleton on ``device``.

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
        KeyError
            If the payload does not hold exactly ``item_embeddings.weight``.
        ValueError
            If the training table has no ``rating`` column, the item table
            has the wrong shape, a value is non-finite, or
            ``loss_scale="global_M"`` is missing ``M``.
        """
        t0 = time.time()
        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data
        user_id_to_local = dict(client_state.local_parameters["user_id_to_local"])

        if "rating" not in training_data.columns:
            raise ValueError(
                "FedMF requires a 'rating' column; implicit all-ones "
                "targets are not used."
            )

        item_table = self._server_item_table(training_request.global_payload)
        num_items = int(item_table.shape[0])
        item_ids = training_data["item_id"].to_numpy(dtype=np.int64)
        ratings = training_data["rating"].to_numpy(dtype=np.float64)
        rows, local_items = np.unique(item_ids, return_inverse=True)
        item_rows, upload_dtype = self._gather_item_rows(item_table, rows)

        stored_users = torch.as_tensor(client_state.local_parameters["user_embeddings"])
        user_table = np.array(
            stored_users.detach().cpu().to(torch.float64).numpy()
        )

        data_scale = self._data_scale(training_request)
        user_step = (
            self._sgd_sequential_user
            if self.local_update == "sgd_sequential"
            else self._full_gradient_user
        )
        user_positions = (
            training_data.groupby("user_id", sort=False).indices
            if len(training_data) > 0
            else {}
        )
        all_losses: list[float] = []
        num_training_examples = 0
        grad_rows = np.zeros_like(item_rows)

        for _ in range(self.local_epochs):
            epoch_sq: list[float] = []
            for user_id, positions in user_positions.items():
                local_uid = user_id_to_local[int(user_id)]
                u, touched, row_grads, sq = user_step(
                    user_table[local_uid],
                    item_rows,
                    local_items[positions],
                    ratings[positions],
                    data_scale,
                )
                user_table[local_uid] = u
                grad_rows[touched] += row_grads
                epoch_sq.extend(sq)
            num_training_examples += len(item_ids)
            if epoch_sq:
                all_losses.append(float(np.mean(epoch_sq)))
            self._require_finite(user_table, "user factors")
            self._require_finite(grad_rows, "item-factor gradient")

        upload = self._item_gradient_upload(
            training_request, rows, grad_rows, num_items, upload_dtype
        )
        updates_packages = {
            "item_embeddings": self._update_package(
                training_request, "item_embeddings", upload
            )
        }
        updated_state = ClientStateBase(
            client_id=client_id,
            local_parameters={
                "user_embeddings": torch.from_numpy(user_table).to(stored_users.dtype),
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

    def _server_item_table(self, global_payload: dict[str, Any]) -> Any:
        """Return the payload's item table after checking names and shape.

        Parameters
        ----------
        global_payload : dict
            ParamIO groups from the server; must hold exactly
            ``item_embeddings.weight``.

        Returns
        -------
        torch.Tensor or ndarray
            The item table ``V`` as sent, shape ``(num_items, latent_dim)``.
            Not copied.

        Raises
        ------
        KeyError
            If the payload holds other or missing parameter names.
        ValueError
            If the table is not 2-D, its width is not ``latent_dim``, or
            its height disagrees with ``num_items``.

        Examples
        --------
        >>> client = FedMFClientAlgorithm(latent_dim=3)
        >>> payload = {"item_embeddings": {"item_embeddings.weight": torch.zeros(5, 3)}}
        >>> tuple(client._server_item_table(payload).shape)
        (5, 3)
        """
        flat = flatten_grouped_state(global_payload)
        if set(flat) != {_ITEM_WEIGHT}:
            raise KeyError(
                f"FedMF payload must hold exactly {_ITEM_WEIGHT!r}, "
                f"got {sorted(flat)}"
            )
        table = flat[_ITEM_WEIGHT]
        shape = tuple(table.shape)
        if len(shape) != 2 or shape[1] != self.latent_dim:
            raise ValueError(
                f"Item table must have shape (num_items, {self.latent_dim}), "
                f"got {shape}"
            )
        if self.num_items is not None and shape[0] != self.num_items:
            raise ValueError(
                f"Item table has {shape[0]} rows, expected num_items={self.num_items}"
            )
        return table

    @staticmethod
    def _gather_item_rows(item_table: Any, rows: np.ndarray) -> tuple[np.ndarray, torch.dtype]:
        """Copy selected item rows to a float64 CPU array in one transfer.

        Parameters
        ----------
        item_table : torch.Tensor or ndarray
            Item table ``V``, shape ``(num_items, d)``, on any device.
        rows : ndarray of int64
            Item ids to gather.

        Returns
        -------
        item_rows : ndarray of float64
            ``V[rows]``, shape ``(len(rows), d)``.
        table_dtype : torch.dtype
            Dtype of ``item_table``, used to cast uploads back.

        Examples
        --------
        >>> v = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        >>> item_rows, dtype = FedMFClientAlgorithm._gather_item_rows(v, np.array([2, 0]))
        >>> item_rows.tolist(), dtype
        ([[4.0, 5.0], [0.0, 1.0]], torch.float32)
        """
        if isinstance(item_table, torch.Tensor):
            idx = torch.as_tensor(rows, dtype=torch.long, device=item_table.device)
            gathered = item_table.detach()[idx].cpu()
        else:
            gathered = torch.as_tensor(np.asarray(item_table)[rows])
        return gathered.to(torch.float64).numpy(), gathered.dtype

    def _sgd_sequential_user(
        self,
        u: np.ndarray,
        item_rows: np.ndarray,
        local_items: np.ndarray,
        ratings: np.ndarray,
        data_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Reference-code pass: update ``u`` after each rating, assign item rows.

        If an item occurs more than once, its last gradient wins, as in the
        reference ``user_update``.

        Parameters
        ----------
        u : ndarray of float64
            Current user vector, shape ``(d,)``. Not written.
        item_rows : ndarray of float64
            Downloaded ``V`` restricted to the client's rated items, shape
            ``(n_rated, d)``. Not written.
        local_items : ndarray of int
            Row indices into ``item_rows``, in stored rating order.
        ratings : ndarray of float64
            Ratings aligned with ``local_items``.
        data_scale : float
            Multiplier for the data term.

        Returns
        -------
        u : ndarray of float64
            Updated user vector.
        touched : ndarray of int64
            Unique row indices into ``item_rows`` that received a gradient.
        row_grads : ndarray of float64
            Item gradients aligned with ``touched``, shape ``(len(touched), d)``.
        sq_errors : list of float
            Squared residual at each rating, before the ``u`` step.

        Examples
        --------
        >>> client = FedMFClientAlgorithm(lr=0.1, reg_u=0.0, reg_v=0.0)
        >>> u, touched, g, sq = client._sgd_sequential_user(
        ...     np.ones(2), np.eye(2), np.array([0]), np.array([3.0]), 1.0)
        >>> u.tolist(), touched.tolist(), sq
        ([1.4, 1.0], [0], [4.0])
        """
        data_coef = -2.0 * data_scale
        reg_u_coef = 2.0 * self.reg_u if self.u_regularisation == "per_rating" else 0.0
        reg_v_coef = 2.0 * self.reg_v if self.v_regularisation != "none" else 0.0
        v_scale = self.lr if self.lr_placement == "client" else 1.0
        assigned: dict[int, np.ndarray] = {}
        sq_errors: list[float] = []
        for j, rate in zip(local_items.tolist(), ratings.tolist()):
            v = item_rows[j]
            error = rate - float(u @ v)
            sq_errors.append(error * error)
            u = u - self.lr * (data_coef * error * v + reg_u_coef * u)
            assigned[j] = v_scale * (data_coef * error * u + reg_v_coef * v)
        if self.u_regularisation == "once":
            u = u - self.lr * (2.0 * self.reg_u * u)
        touched = np.fromiter(assigned, dtype=np.int64, count=len(assigned))
        if assigned:
            row_grads = np.stack(list(assigned.values()))
        else:
            row_grads = np.zeros((0, item_rows.shape[1]))
        return u, touched, row_grads, sq_errors

    def _full_gradient_user(
        self,
        u: np.ndarray,
        item_rows: np.ndarray,
        local_items: np.ndarray,
        ratings: np.ndarray,
        data_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Paper full-gradient step at ``(u^{t-1}, V^{t-1})``.

        Parameters
        ----------
        u : ndarray of float64
            Current user vector, shape ``(d,)``. Not written.
        item_rows : ndarray of float64
            Downloaded ``V`` restricted to the client's rated items, shape
            ``(n_rated, d)``. Not written.
        local_items : ndarray of int
            Row indices into ``item_rows``.
        ratings : ndarray of float64
            Ratings aligned with ``local_items``.
        data_scale : float
            Multiplier for the data term.

        Returns
        -------
        u : ndarray of float64
            Updated user vector.
        touched : ndarray of int64
            Unique row indices into ``item_rows`` that received a gradient.
        row_grads : ndarray of float64
            Summed item gradients aligned with ``touched``.
        sq_errors : list of float
            Squared residuals at ``(u^{t-1}, V)``.

        Examples
        --------
        >>> client = FedMFClientAlgorithm(lr=0.1, reg_u=0.0, reg_v=0.0,
        ...                               local_update="full_gradient")
        >>> u, touched, g, sq = client._full_gradient_user(
        ...     np.ones(2), np.eye(2), np.array([0, 1]), np.array([3.0, 1.0]), 1.0)
        >>> u.tolist(), sq
        ([1.4, 1.0], [4.0, 0.0])
        """
        if len(local_items) == 0:
            return u, np.zeros(0, dtype=np.int64), np.zeros((0, item_rows.shape[1])), []
        v_rows = item_rows[local_items]
        errors = ratings - v_rows @ u
        sq_errors = (errors ** 2).tolist()
        coef = data_scale * (-2.0 * errors)
        grad_u = coef @ v_rows
        if self.u_regularisation == "per_rating":
            grad_u = grad_u + (2.0 * self.reg_u * u) * len(local_items)
        else:
            grad_u = grad_u + 2.0 * self.reg_u * u
        touched, inverse = np.unique(local_items, return_inverse=True)
        row_grads = np.zeros((len(touched), item_rows.shape[1]))
        np.add.at(row_grads, inverse, coef[:, None] * u)
        if self.v_regularisation == "per_rating":
            np.add.at(row_grads, inverse, (2.0 * self.reg_v) * v_rows)
        elif self.v_regularisation == "once_rated":
            row_grads += 2.0 * self.reg_v * item_rows[touched]
        if self.lr_placement == "client":
            row_grads = self.lr * row_grads
        u = u - self.lr * grad_u
        return u, touched.astype(np.int64), row_grads, sq_errors

    def _item_gradient_upload(
        self,
        training_request: ClientTrainingRequest,
        rows: np.ndarray,
        grad_rows: np.ndarray,
        num_items: int,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Dense table or sparse ``{rows, values}`` from the effective layout.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Read for ``param_ios["item_embeddings"].layout``.
        rows : ndarray of int64
            Sorted item ids this client rated (PartText support).
        grad_rows : ndarray of float64
            Item gradients aligned with ``rows``, shape ``(len(rows), d)``.
        num_items : int
            Height of the dense table for FullText.
        dtype : torch.dtype
            Dtype of the server item table; uploads are cast to it.

        Returns
        -------
        dict
            Either ``{item_embeddings.weight: table}`` or
            ``{"rows": ids, "values": {item_embeddings.weight: rows}}``,
            with CPU tensors of ``dtype``.

        Examples
        --------
        >>> up = client._item_gradient_upload(  # doctest: +SKIP
        ...     request, np.array([1]), np.ones((1, 4)), 6, torch.float32)
        >>> up["rows"].tolist()  # doctest: +SKIP
        [1]
        """
        param_ios = training_request.param_ios or {}
        layout = getattr(param_ios.get("item_embeddings"), "layout", "rows")
        values = torch.from_numpy(grad_rows).to(dtype)
        if layout == "dense":
            table = torch.zeros((num_items, grad_rows.shape[1]), dtype=dtype)
            table[torch.from_numpy(rows)] = values
            return {_ITEM_WEIGHT: table}
        return {"rows": rows, "values": {_ITEM_WEIGHT: values}}

    @staticmethod
    def _require_finite(array: np.ndarray, where: str) -> None:
        """Raise if ``array`` contains NaN or Inf.

        Parameters
        ----------
        array : ndarray
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

        Examples
        --------
        >>> FedMFClientAlgorithm._require_finite(np.array([np.nan]), "u")
        Traceback (most recent call last):
        ...
        ValueError: Non-finite values in u
        """
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {where}")
