"""FedPerGNN client algorithm.

The shared GAT skeleton lives on this object and is loaded from the server
payload each call. ClientState holds the owner's user embedding and a
name-keyed optimiser dict; that embedding is the source of truth for the
owner's trajectory. The server user table is a mirror updated by the sparse
upload each round (expansion / neighbours). With neighbor_embeddings_trainable,
other clients can still write the owner's row in the mirror, but that no
longer becomes this client's next starting point. The flag covers neighbour
rows only; the anchor user's embedding is always trainable. Those other-user
rows are parameters of this round's Adam, initialised from the expansion
payload. Repeated slots of one user share one vector, and the upload is
that vector after the local epochs. Their moments are not kept: the
mirror, not this client's optimiser, is the next round's starting point.

``FedPerGNN`` copies ``hidden_dim``, ``history_len`` and ``neighbor_len``
from this object at construction. ``neighbor_embeddings_trainable`` is
read only by this algorithm.

Training task (``task``):
  ``"rating"`` (default) follows Wu et al.: MSE on observed ratings only,
  no negative sampling. Scores are unbounded inner products. The library
  evaluator is still top-K ranking; that pairing is a documented deviation
  from the paper's RMSE protocol, not a second training mode.

  ``"ranking"`` is the library ranking protocol: every remaining observed
  row is label 1 (raw ratings and play counts are ignored), unobserved
  items are sampled as label 0, and the client optimises BCE-with-logits.
  Drop low-count implicit rows with ``rating_exclusion_threshold`` at load
  time before this binarisation.   Row-layout item uploads include every
  item that entered the local loss, so ranking negatives are visible to
  the server. Pseudo-interacted items are not those negatives: they are
  extra rows added by ``apply_privacy``, with gradients drawn to match
  the observed-item deltas, and they never enter the loss.

optimizer_state_scope:
  - "client_resident" (default): persist Adam moments for the owner row
  - "none": no optimiser state is kept
  - "all": persist moments for every assembled parameter (memory-heavy).
    Neighbour rows are not assembled parameters. They are re-initialised
    from the payload every round, so their moments are never stored.
"""

import logging
import time
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
from src.fedreclab.federated.mechanisms import ClipLaplaceLDP, PseudoItemSampling
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
_TASKS = ("rating", "ranking")
_ALL_SCOPE_WARN_BYTES = 1 << 30
_DEFAULT_RANKING_NEGATIVES = 4
# Fourth seed term so privacy draws do not share a stream with local shuffling.
_PRIVACY_RNG_SALT = 17


class FedPerGNNClientAlgorithm(BaseClientAlgorithm):
    """Local training and scoring for FedPerGNN.

    Defaults follow Wu et al.: rating-valued MSE on observed interactions
    only. Pass ``task="ranking"`` for implicit binary labels, negative
    sampling, and BCE-with-logits under the library ranking evaluator.

    Parameters
    ----------
    hidden_dim : int, optional
        Width of embeddings and GAT hidden states. Default is 64.
    history_len : int, optional
        Padded history length. Default is 20.
    neighbor_len : int, optional
        Padded neighbour count per history item. Default is 10.
    local_epochs : int, optional
        Local Adam epochs per round. Default is 1.
    lr : float, optional
        Adam learning rate. Default is ``1e-3``.
    batch_size : int, optional
        Mini-batch size. Default is 128.
    neighbor_embeddings_trainable : bool, optional
        If True, each distinct other user in the expansion is an
        ``nn.Parameter`` for this round. It is initialised from the
        payload and updated by the same Adam as the owner embedding and
        the GAT, on every local batch. The resulting rows are uploaded
        to the server mirror. Default is False. The owner row is always
        trainable and is not one of these parameters. Neighbour Adam
        moments are not stored; the next round starts from the mirror.
    optimizer_state_scope : {"client_resident", "none", "all"}, optional
        Which Adam moments to persist in ``ClientState``. Default is
        ``"client_resident"``.
    task : {"rating", "ranking"}, optional
        ``"rating"`` (default) trains MSE on raw ratings with no
        negatives. ``"ranking"`` binarises observed rows to 1, samples
        unobserved items as 0, and trains BCE-with-logits.
    neg_samples : int or None, optional
        Unobserved items sampled per positive. ``None`` (default) is 0
        under ``task="rating"`` and 4 under ``task="ranking"``. Must be
        0 when ``task="rating"``.
    num_pseudo_items : int, optional
        Pseudo-interacted items appended to the item-embedding upload
        by :meth:`apply_privacy`. ``0`` (default) disables them. Wu et
        al. use 1000. These rows never enter the loss.
    ldp_clip : float or None, optional
        Shared L1 clip ``δ`` on the item, user and model deltas.
        ``None`` (default) disables clipping. Wu et al. use 0.1.
    ldp_scale : float, optional
        Laplace scale ``λ`` added after the clip. ``0`` (default)
        disables noise. Wu et al. use 0.2, and bound the privacy
        budget by ``2 * δ * e / λ`` with ``e`` local epochs.

    Raises
    ------
    ValueError
        If ``optimizer_state_scope`` or ``task`` is unknown, if
        ``neg_samples`` is negative, or if rating mode is paired with
        a non-zero ``neg_samples``. Also if ``num_pseudo_items`` is
        negative, ``ldp_clip`` is not a positive finite number, or
        ``ldp_scale`` is negative.

    Examples
    --------
    >>> FedPerGNNClientAlgorithm().task, FedPerGNNClientAlgorithm().neg_samples
    ('rating', 0)
    >>> FedPerGNNClientAlgorithm(task="ranking").neg_samples
    4
    """

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
        task: str = "rating",
        neg_samples: int | None = None,
        num_pseudo_items: int = 0,
        ldp_clip: float | None = None,
        ldp_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if optimizer_state_scope not in _OPTIMIZER_STATE_SCOPES:
            raise ValueError(
                f"optimizer_state_scope must be one of {_OPTIMIZER_STATE_SCOPES}, "
                f"got {optimizer_state_scope!r}"
            )
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        if neg_samples is not None and (not isinstance(neg_samples, int) or isinstance(neg_samples, bool) or neg_samples < 0):
            raise ValueError(
                f"neg_samples must be a non-negative integer or None, got {neg_samples!r}"
            )
        if task == "rating":
            resolved_neg = 0 if neg_samples is None else neg_samples
            if resolved_neg != 0:
                raise ValueError(
                    "task='rating' uses observed items only (Wu et al. L_i). "
                    "Set task='ranking' to sample negatives."
                )
        else:
            resolved_neg = (
                _DEFAULT_RANKING_NEGATIVES if neg_samples is None else neg_samples
            )
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.neighbor_embeddings_trainable = neighbor_embeddings_trainable
        self.optimizer_state_scope = optimizer_state_scope
        self.task = task
        self.neg_samples = resolved_neg
        # Constructors validate count, clip and scale and reject bools.
        self.pseudo_items = PseudoItemSampling(num_pseudo_items)
        self.clip_laplace = ClipLaplaceLDP(clip=ldp_clip, scale=ldp_scale)
        self.num_pseudo_items = self.pseudo_items.count
        self.ldp_clip = self.clip_laplace.clip
        self.ldp_scale = self.clip_laplace.scale

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

        users_in_client = training_request.users_in_client
        central_user_id = int(users_in_client[0]) if users_in_client else 0

        item_col = "item_id"
        rating_col = "rating" if "rating" in training_data.columns else None

        pos_item_ids = training_data[item_col].values.astype(np.int64)
        if rating_col is not None:
            pos_ratings = training_data[rating_col].values.astype(np.float32)
        else:
            pos_ratings = np.ones(len(training_data), dtype=np.float32)

        expanded_for_client = self._require_expansion(training_request.extra_payload, client_id)
        history_items_np = expanded_for_client.get("history_item_ids")
        neighbor_emb_np = expanded_for_client.get("neighbor_embeddings")
        neighbor_user_ids_np = expanded_for_client.get("neighbor_user_ids")
        hist_items = torch.as_tensor(history_items_np, dtype=torch.long, device=device)
        if neighbor_emb_np is not None:
            neighbor_emb = torch.as_tensor(
                neighbor_emb_np, dtype=torch.float32, device=device
            )
        else:
            neighbor_emb = None

        # Round-local rows. Moments are not restored: the payload, not
        # the previous round's optimiser, is the starting point.
        neighbour_param = None
        neighbour_user_ids: List[int] = []
        neighbour_index = None
        frozen_slots = None
        if (
            self.neighbor_embeddings_trainable
            and neighbor_emb_np is not None
            and neighbor_user_ids_np is not None
        ):
            (
                neighbour_param,
                neighbour_user_ids,
                neighbour_index,
                frozen_slots,
            ) = self._bind_neighbour_parameter(
                neighbor_emb_np,
                neighbor_user_ids_np,
                central_user_id,
                device,
            )

        optim_params = list(model.parameters()) + [user_emb]
        if neighbour_param is not None:
            optim_params.append(neighbour_param)
        optimizer = torch.optim.Adam(optim_params, lr=self.lr)
        restore_named_optimizer_state(
            optimizer, named_params, client_state.optimizer_state
        )

        all_losses: List[float] = []
        num_training_examples = 0
        trained_item_ids: set[int] = set(int(i) for i in pos_item_ids)
        if training_request.seed is None:
            rng = None
        else:
            rng = np.random.default_rng(
                [training_request.seed, training_request.client_id, training_request.current_round]
            )

        for _ in range(self.local_epochs):
            item_ids_np, labels_np = self._build_epoch_samples(
                pos_item_ids, pos_ratings, rng
            )
            trained_item_ids.update(int(i) for i in item_ids_np)
            num_examples = len(item_ids_np)
            indices = np.arange(num_examples)
            if rng is None:
                np.random.shuffle(indices)
            else:
                rng.shuffle(indices)

            item_ids_np_shuffled = item_ids_np[indices]
            labels_np_shuffled = labels_np[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_examples, self.batch_size):
                end = min(start + self.batch_size, num_examples)
                if start >= end:
                    break

                b_item_ids = torch.as_tensor(
                    item_ids_np_shuffled[start:end], dtype=torch.long, device=device
                )
                b_labels = torch.as_tensor(
                    labels_np_shuffled[start:end], dtype=torch.float32, device=device
                )

                if neighbour_param is not None:
                    b_neighbor_emb = self._neighbor_embeddings_from_parameter(
                        neighbour_param, neighbour_index, frozen_slots
                    )
                else:
                    b_neighbor_emb = neighbor_emb

                # History and neighbours do not depend on the target item.
                # One user state is shared by every row; mean loss sends
                # the batch-mean gradient back through that state.
                optimizer.zero_grad()
                user_repr = model.user_representation(
                    history_item_ids=hist_items,
                    user_embedding=user_emb,
                    neighbor_embeddings=b_neighbor_emb,
                )
                preds = model.score(user_repr, b_item_ids)
                loss = self._compute_loss(preds, b_labels)
                loss.backward()
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

        item_rows = set(trained_item_ids)
        if history_items_np is not None:
            # Negative ids are empty slots, not catalogue rows.
            item_rows.update(
                int(item_id)
                for item_id in np.asarray(history_items_np).reshape(-1)
                if int(item_id) >= 0
            )
        item_upload = self._embedding_upload(
            training_request,
            "item_embeddings",
            model.get_item_embeddings(),
            item_rows,
        )

        user_embedding_row = user_emb.detach().cpu().numpy()
        user_ids_list: List[int] = [central_user_id]
        embeddings_list: List[np.ndarray] = [np.asarray(user_embedding_row.squeeze(0), dtype=np.float32)]
        # No local batch means Adam never saw these rows. Uploading the
        # payload again would vote the mirror value back into the average.
        if neighbour_param is not None and optimizer.state.get(neighbour_param):
            updated_rows = neighbour_param.detach().cpu().numpy()
            for uid, row in zip(neighbour_user_ids, updated_rows):
                user_ids_list.append(int(uid))
                embeddings_list.append(np.asarray(row, dtype=np.float32))

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
            extra_response={"task": self.task},
        )

    def apply_privacy(self, training_request, training_response):
        """Append pseudo-item rows and protect the upload with LDP.

        The server calls this after :meth:`local_train`. With the
        defaults (no pseudo items, no clip, no noise) the response is
        returned untouched.

        Pseudo rows are catalogue ids outside the observed items and
        outside the rows already uploaded. Their parameter value is the
        downloaded embedding minus a diagonal-Gaussian vector matched
        to the observed items' deltas (downloaded row minus uploaded
        row). They are not training samples.

        When clipping or Laplace noise is enabled, one L1 budget covers
        the item-row deltas (existing and pseudo), the user-row deltas
        and every model-state tensor. Protected values are written back
        as parameters: ``downloaded - protected_delta``. The owner user
        delta is taken from the client-state row the step started from;
        neighbour rows, when present, use the server user table. This
        write-back is a stand-in until the groups are exchanged as
        gradients.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Carries the downloaded tables, the training items and the seed.
        training_response : ClientTrainingResponse
            Result of :meth:`local_train`. Packages are updated in place.

        Returns
        -------
        ClientTrainingResponse
            ``training_response``.

        Raises
        ------
        RuntimeError
            If privacy is enabled and ``training_request.seed`` is None,
            or an observed item is missing from the item upload.
        TypeError
            If the item upload is not a sparse ``{rows, values}`` dict
            while privacy is enabled.
        ValueError
            If fewer than ``num_pseudo_items`` candidates remain, or a
            protected array is non-finite.
        KeyError
            If Laplace or clipping is enabled and a non-owner user row
            has no server user table to difference against.

        Examples
        --------
        >>> client = FedPerGNNClientAlgorithm(num_pseudo_items=0)
        >>> client.apply_privacy(None, "response")
        'response'
        """
        if self.num_pseudo_items == 0 and not self.clip_laplace.enabled:
            return training_response
        rng = self._privacy_rng(training_request)
        self._protect_item_rows(training_request, training_response, rng)
        return training_response

    def _privacy_rng(self, training_request: ClientTrainingRequest):
        """Seeded generator for pseudo items and Laplace noise.

        The stream is ``[seed, client_id, current_round, salt]`` so it
        does not follow the shuffle generator inside :meth:`local_train`.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Must carry a non-None ``seed``.

        Returns
        -------
        numpy.random.Generator

        Raises
        ------
        RuntimeError
            If ``seed`` is None.

        Examples
        --------
        >>> client = FedPerGNNClientAlgorithm(num_pseudo_items=1)
        >>> req = ClientTrainingRequest(
        ...     client_id=1, current_round=2, users_in_client=[1],
        ...     client_state=None, global_payload={}, training_data=None,
        ...     seed=3)
        >>> client._privacy_rng(req).integers(1_000_000) == (
        ...     np.random.default_rng([3, 1, 2, 17]).integers(1_000_000))
        True
        """
        if training_request.seed is None:
            raise RuntimeError(
                "FedPerGNN privacy sampling needs ClientTrainingRequest.seed. "
                "Pass a seed on the server so pseudo items and Laplace noise "
                "are reproducible."
            )
        return np.random.default_rng(
            [
                int(training_request.seed),
                int(training_request.client_id),
                int(training_request.current_round),
                _PRIVACY_RNG_SALT,
            ]
        )

    def _protect_item_rows(self, training_request, training_response, rng):
        """Write pseudo-item rows and, when enabled, LDP into the packages.

        Draw order on ``rng`` is fixed: pseudo ids, matched gradients,
        then Laplace noise over item deltas, pseudo deltas, user deltas
        and model tensors. Existing item rows keep their values when
        LDP is off.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Downloaded parameters and the local interaction table.
        training_response : ClientTrainingResponse
            Packages produced by :meth:`local_train`.
        rng : numpy.random.Generator
            Generator from :meth:`_privacy_rng`.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``item_embeddings`` is not a row-layout upload.
        RuntimeError
            If an observed item has no uploaded row.
        KeyError
            If a neighbour user row needs the server user table and it
            is absent.
        ValueError
            If the pseudo-item pool is too small or a value is non-finite.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from src.fedreclab.federated.base_client import (
        ...     ClientStateBase, ClientTrainingRequest, ClientTrainingResponse)
        >>> from src.fedreclab.federated.utils import ParamIO, UpdatePackage
        >>> client = FedPerGNNClientAlgorithm(num_pseudo_items=1)
        >>> table = np.zeros((3, 1), dtype=np.float32)
        >>> state = ClientStateBase(0, {})
        >>> req = ClientTrainingRequest(
        ...     client_id=0, current_round=0, users_in_client=[0],
        ...     client_state=state,
        ...     global_payload={"item_embeddings": table},
        ...     training_data=pd.DataFrame({"item_id": [0]}),
        ...     seed=0,
        ... )
        >>> resp = ClientTrainingResponse(
        ...     client_id=0, updated_client_state=state, updates_packages={
        ...         "item_embeddings": UpdatePackage(
        ...             "item_embeddings",
        ...             ParamIO("server", "params", "server", layout="rows"),
        ...             {"rows": np.array([0]),
        ...              "values": np.array([[0.5]], dtype=np.float32)},
        ...             {},
        ...         ),
        ...     },
        ...     training_losses=[], training_duration=0.0, num_training_examples=1,
        ... )
        >>> client._protect_item_rows(req, resp, client._privacy_rng(req))
        >>> rows = resp.updates_packages["item_embeddings"].param_value["rows"]
        >>> int(rows[0]) == 0 and len(rows) == 2
        True
        """
        item_package = training_response.updates_packages["item_embeddings"]
        upload = item_package.param_value
        if not isinstance(upload, dict) or "rows" not in upload or "values" not in upload:
            raise TypeError(
                "FedPerGNN privacy expects a sparse item upload "
                "{'rows', 'values'}. A dense item table cannot take "
                "extra pseudo-item rows."
            )
        downloaded_items = np.asarray(
            training_request.global_payload["item_embeddings"], dtype=np.float32
        )
        rows = np.asarray(upload["rows"], dtype=np.int64)
        values = np.asarray(upload["values"], dtype=np.float32)
        observed = {
            int(item_id)
            for item_id in training_request.training_data["item_id"].to_numpy()
        }
        real_idx = np.array([int(row) in observed for row in rows], dtype=bool)
        if self.num_pseudo_items > 0:
            missing = observed.difference(int(row) for row in rows)
            if missing:
                raise RuntimeError(
                    "Observed items "
                    f"{sorted(missing)} are missing from the item upload, "
                    "so pseudo-item gradients cannot be matched to them."
                )
            if not real_idx.any():
                raise RuntimeError(
                    "Cannot match pseudo-item gradients: the upload has "
                    "no observed-item rows."
                )
            real_deltas = downloaded_items[rows[real_idx]] - values[real_idx]
            excluded = observed.union(int(row) for row in rows)
            pseudo_ids = self.pseudo_items.sample_ids(
                int(downloaded_items.shape[0]), excluded, rng
            )
            pseudo_deltas = self.pseudo_items.matched_gradients(real_deltas, rng)
        else:
            pseudo_ids = np.zeros((0,), dtype=np.int64)
            pseudo_deltas = np.zeros((0, values.shape[1]), dtype=np.float32)

        new_user = None
        new_model = None
        if self.clip_laplace.enabled:
            existing_deltas = downloaded_items[rows] - values
            user_starts, user_deltas = self._user_deltas(
                training_request, training_response
            )
            model_keys, model_starts, model_deltas = self._model_deltas(
                training_request, training_response
            )
            protected = self.clip_laplace.apply(
                [existing_deltas, pseudo_deltas, user_deltas, *model_deltas],
                rng,
            )
            new_values = downloaded_items[rows] - protected[0]
            pseudo_values = downloaded_items[pseudo_ids] - protected[1]
            new_user = (user_starts - protected[2]).astype(np.float32, copy=False)
            new_model = [
                (start - delta).astype(np.float32, copy=False)
                for start, delta in zip(model_starts, protected[3:])
            ]
        else:
            new_values = values
            pseudo_values = downloaded_items[pseudo_ids] - pseudo_deltas

        protected_parts = [new_values, pseudo_values]
        if new_user is not None:
            protected_parts.append(new_user)
            protected_parts.extend(new_model)
        if any(not np.isfinite(part).all() for part in protected_parts):
            raise ValueError("Protected upload contains non-finite values")
        if new_user is not None:
            self._write_user_values(training_response, new_user)
            self._write_model_values(training_response, model_keys, new_model)
        item_package.param_value = {
            "rows": np.concatenate([rows, pseudo_ids]).astype(np.int64, copy=False),
            "values": np.concatenate([new_values, pseudo_values], axis=0).astype(
                np.float32, copy=False
            ),
        }

    def _user_deltas(self, training_request, training_response):
        """Deltas of the uploaded user rows from the vector each step started at.

        The owner row starts from ``ClientState``, which can differ from
        the server mirror. Neighbour rows start from that mirror.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Original client state and, when a neighbour row is present,
            ``global_payload['user_embeddings']``.
        training_response : ClientTrainingResponse
            Must contain a sparse ``user_embeddings`` package.

        Returns
        -------
        starts : ndarray
            Shape ``(n, d)``, the pre-step rows.
        deltas : ndarray
            ``starts - uploaded values``, shape ``(n, d)``.

        Raises
        ------
        KeyError
            If a neighbour row is uploaded and the server user table
            is missing.
        TypeError
            If the user upload is not a sparse row dict.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from src.fedreclab.federated.base_client import (
        ...     ClientStateBase, ClientTrainingRequest, ClientTrainingResponse)
        >>> from src.fedreclab.federated.utils import ParamIO, UpdatePackage
        >>> state = ClientStateBase(
        ...     0, {"user_embedding.weight": np.array([[1.0, 0.0]], dtype=np.float32)})
        >>> req = ClientTrainingRequest(
        ...     client_id=0, current_round=0, users_in_client=[0],
        ...     client_state=state, global_payload={}, training_data=pd.DataFrame(),
        ... )
        >>> resp = ClientTrainingResponse(
        ...     client_id=0, updated_client_state=state, updates_packages={
        ...         "user_embeddings": UpdatePackage(
        ...             "user_embeddings",
        ...             ParamIO("server", "params", "server", layout="rows"),
        ...             {"rows": np.array([0]), "values": np.array([[0.0, 0.0]], dtype=np.float32)},
        ...             {},
        ...         ),
        ...     },
        ...     training_losses=[], training_duration=0.0, num_training_examples=1,
        ... )
        >>> starts, deltas = FedPerGNNClientAlgorithm()._user_deltas(req, resp)
        >>> np.allclose(deltas, [[1.0, 0.0]])
        True
        """
        package = training_response.updates_packages["user_embeddings"]
        upload = package.param_value
        if not isinstance(upload, dict) or "rows" not in upload or "values" not in upload:
            raise TypeError(
                "FedPerGNN LDP expects a sparse user upload {'rows', 'values'}."
            )
        user_rows = np.asarray(upload["rows"], dtype=np.int64)
        user_values = np.asarray(upload["values"], dtype=np.float32)
        owner_id = int(training_request.users_in_client[0])
        owner_start = np.asarray(
            training_request.client_state.local_parameters["user_embedding.weight"],
            dtype=np.float32,
        ).reshape(-1)
        starts = np.empty_like(user_values)
        table = training_request.global_payload.get("user_embeddings")
        table_np = None if table is None else np.asarray(table, dtype=np.float32)
        for index, user_id in enumerate(user_rows):
            if int(user_id) == owner_id:
                starts[index] = owner_start
                continue
            if table_np is None:
                raise KeyError(
                    "user_embeddings is missing from global_payload, so the "
                    f"neighbour row {int(user_id)} has no pre-step vector to clip."
                )
            starts[index] = table_np[int(user_id)]
        return starts, starts - user_values

    def _model_deltas(self, training_request, training_response):
        """Per-tensor deltas of the uploaded model state.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            ``global_payload['model_state']`` is the downloaded dict.
        training_response : ClientTrainingResponse
            ``model_state`` package is the locally updated dict.

        Returns
        -------
        keys : list of str
            Tensor names in upload order.
        starts : list of ndarray
            Downloaded tensors as float32.
        deltas : list of ndarray
            ``start - uploaded`` for each key.

        Raises
        ------
        ValueError
            If the two dicts have different keys or shapes.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from src.fedreclab.federated.base_client import (
        ...     ClientStateBase, ClientTrainingRequest, ClientTrainingResponse)
        >>> from src.fedreclab.federated.utils import ParamIO, UpdatePackage
        >>> state = ClientStateBase(0, {})
        >>> req = ClientTrainingRequest(
        ...     client_id=0, current_round=0, users_in_client=[0],
        ...     client_state=state,
        ...     global_payload={"model_state": {"w": np.array([1.0], dtype=np.float32)}},
        ...     training_data=pd.DataFrame(),
        ... )
        >>> resp = ClientTrainingResponse(
        ...     client_id=0, updated_client_state=state, updates_packages={
        ...         "model_state": UpdatePackage(
        ...             "model_state", ParamIO("server", "params", "server"),
        ...             {"w": np.array([0.25], dtype=np.float32)}, {},
        ...         ),
        ...     },
        ...     training_losses=[], training_duration=0.0, num_training_examples=1,
        ... )
        >>> keys, _, deltas = FedPerGNNClientAlgorithm()._model_deltas(req, resp)
        >>> keys == ["w"] and np.allclose(deltas[0], [0.75])
        True
        """
        uploaded = training_response.updates_packages["model_state"].param_value
        downloaded = training_request.global_payload["model_state"]
        if set(uploaded) != set(downloaded):
            raise ValueError(
                "model_state keys in the upload "
                f"{sorted(uploaded)} do not match the downloaded keys "
                f"{sorted(downloaded)}"
            )
        keys = list(uploaded)
        starts = []
        deltas = []
        for key in keys:
            start = np.asarray(downloaded[key], dtype=np.float32)
            current = np.asarray(uploaded[key], dtype=np.float32)
            if start.shape != current.shape:
                raise ValueError(
                    f"model_state[{key!r}] shape changed from {start.shape} "
                    f"to {current.shape}"
                )
            starts.append(start)
            deltas.append(start - current)
        return keys, starts, deltas

    @staticmethod
    def _write_user_values(training_response, values):
        """Store protected user-row values on the existing row ids.

        Parameters
        ----------
        training_response : ClientTrainingResponse
            Response whose ``user_embeddings`` package is updated in place.
        values : ndarray
            New row values, shape ``(n, d)``, already equal to
            ``start - protected_delta``.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> from src.fedreclab.federated.base_client import (
        ...     ClientStateBase, ClientTrainingResponse)
        >>> from src.fedreclab.federated.utils import ParamIO, UpdatePackage
        >>> resp = ClientTrainingResponse(
        ...     client_id=0,
        ...     updated_client_state=ClientStateBase(0, {}),
        ...     updates_packages={
        ...         "user_embeddings": UpdatePackage(
        ...             "user_embeddings",
        ...             ParamIO("server", "params", "server", layout="rows"),
        ...             {"rows": np.array([3]), "values": np.zeros((1, 1), dtype=np.float32)},
        ...             {},
        ...         ),
        ...     },
        ...     training_losses=[], training_duration=0.0, num_training_examples=0,
        ... )
        >>> FedPerGNNClientAlgorithm._write_user_values(
        ...     resp, np.array([[0.5]], dtype=np.float32))
        >>> int(resp.updates_packages["user_embeddings"].param_value["rows"][0])
        3
        """
        package = training_response.updates_packages["user_embeddings"]
        upload = package.param_value
        package.param_value = {
            "rows": np.asarray(upload["rows"], dtype=np.int64),
            "values": np.asarray(values, dtype=np.float32),
        }

    @staticmethod
    def _write_model_values(training_response, keys, values):
        """Store protected model tensors under the same names.

        Parameters
        ----------
        training_response : ClientTrainingResponse
            Response whose ``model_state`` dict is updated in place.
        keys : sequence of str
            Tensor names, aligned with ``values``.
        values : sequence of ndarray
            Protected tensors, one per key.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> from src.fedreclab.federated.base_client import (
        ...     ClientStateBase, ClientTrainingResponse)
        >>> from src.fedreclab.federated.utils import ParamIO, UpdatePackage
        >>> resp = ClientTrainingResponse(
        ...     client_id=0,
        ...     updated_client_state=ClientStateBase(0, {}),
        ...     updates_packages={
        ...         "model_state": UpdatePackage(
        ...             "model_state", ParamIO("server", "params", "server"),
        ...             {"w": np.zeros(1, dtype=np.float32)}, {},
        ...         ),
        ...     },
        ...     training_losses=[], training_duration=0.0, num_training_examples=0,
        ... )
        >>> FedPerGNNClientAlgorithm._write_model_values(
        ...     resp, ["w"], [np.array([0.25], dtype=np.float32)])
        >>> float(resp.updates_packages["model_state"].param_value["w"][0])
        0.25
        """
        state = training_response.updates_packages["model_state"].param_value
        for key, value in zip(keys, values):
            state[key] = np.asarray(value, dtype=np.float32)

    def _build_epoch_samples(
        self,
        pos_item_ids: np.ndarray,
        pos_ratings: np.ndarray,
        rng: np.random.Generator | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build one epoch of item ids and labels for the owner user.

        Under ``task="rating"`` the observed rows are kept with their raw
        ratings and no negatives are added. Under ``task="ranking"`` each
        observed row is label 1 and ``neg_samples`` items the user has not
        interacted with are sampled as label 0. Paper pseudo-items are not
        sampled here: they never enter the loss. ``apply_privacy`` appends
        them to the item upload afterwards.

        Parameters
        ----------
        pos_item_ids : ndarray
            Observed item ids, shape ``(n_pos,)``.
        pos_ratings : ndarray
            Raw ratings aligned with the positives. Ignored when
            ``task="ranking"``.
        rng : numpy.random.Generator or None
            Seeded generator. ``None`` falls back to global ``np.random``.

        Returns
        -------
        item_ids : ndarray
            Item ids, shape ``(n_pos * (1 + neg_samples),)``.
        labels : ndarray
            Targets aligned with ``item_ids``. Ratings under
            ``task="rating"``; 1/0 under ``task="ranking"``.

        Raises
        ------
        RuntimeError
            If ranking negatives are requested and the user has interacted
            with every catalogue item, so no unobserved item exists.

        Examples
        --------
        >>> client = FedPerGNNClientAlgorithm(task="ranking", neg_samples=0)
        >>> client._num_items = 5
        >>> items, y = client._build_epoch_samples(
        ...     np.array([1]), np.array([4.0], dtype=np.float32), None)
        >>> y
        array([1.], dtype=float32)
        """
        n_pos = len(pos_item_ids)
        n_neg = self.neg_samples if self.task == "ranking" else 0
        n_total = n_pos * (1 + n_neg)
        item_ids = np.empty(n_total, dtype=np.int64)
        labels = np.empty(n_total, dtype=np.float32)
        excl = {int(i) for i in pos_item_ids}
        if n_neg > 0:
            n_items = self._num_items
            if n_items is None:
                raise RuntimeError("ranking negatives require num_items to be set")
            if n_items <= len(excl):
                raise RuntimeError(
                    "Cannot sample ranking negatives: the user has interacted "
                    "with every catalogue item."
                )

        idx = 0
        draw = np.random.randint if rng is None else rng.integers
        for i in range(n_pos):
            item_ids[idx] = int(pos_item_ids[i])
            if self.task == "ranking":
                labels[idx] = 1.0
            else:
                labels[idx] = float(pos_ratings[i])
            idx += 1
            for _ in range(n_neg):
                neg_item = int(draw(0, n_items))
                while neg_item in excl:
                    neg_item = int(draw(0, n_items))
                item_ids[idx] = neg_item
                labels[idx] = 0.0
                idx += 1
        return item_ids[:idx], labels[:idx]

    def _compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the loss for the configured task.

        Parameters
        ----------
        preds : torch.Tensor
            Unbounded inner-product scores, shape ``(batch,)``.
        targets : torch.Tensor
            Ratings under ``task="rating"``; 0/1 under ``task="ranking"``.

        Returns
        -------
        torch.Tensor
            Scalar mean loss. MSE for rating, BCE-with-logits for ranking.

        Raises
        ------
        RuntimeError
            If ``self.task`` is not ``"rating"`` or ``"ranking"``. The
            constructor already rejects unknown names, so this is a guard.

        Examples
        --------
        >>> client = FedPerGNNClientAlgorithm()
        >>> client._compute_loss(torch.zeros(2), torch.ones(2)).item()
        1.0
        """
        if self.task == "ranking":
            return torch.nn.functional.binary_cross_entropy_with_logits(preds, targets)
        if self.task == "rating":
            return torch.nn.functional.mse_loss(preds, targets)
        raise RuntimeError(f"unrecognised task {self.task!r}")

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
        validation loss.         Both tasks return unbounded inner products, which
        ``FullRankingEvaluator`` ranks as higher-is-better. The user
        tower is evaluated once for the client and reused for every
        candidate.

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
        if not candidate_items_per_user:
            return scores

        with torch.no_grad():
            hist_items = torch.as_tensor(
                history_items_np, dtype=torch.long, device=device
            )
            if neighbor_emb_np is not None:
                neigh = torch.as_tensor(
                    neighbor_emb_np, dtype=torch.float32, device=device
                )
            else:
                neigh = None
            user_repr = model.user_representation(
                history_item_ids=hist_items,
                user_embedding=user_emb,
                neighbor_embeddings=neigh,
            )
            for user_id, cand_items in candidate_items_per_user.items():
                item_ids = torch.as_tensor(cand_items, dtype=torch.long, device=device)
                scores[user_id] = model.score(user_repr, item_ids).cpu().numpy()

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

    def _bind_neighbour_parameter(
        self,
        neighbor_embeddings,
        neighbor_user_ids,
        owner_id: int,
        device,
    ):
        """Build one Adam parameter per distinct neighbour user.

        The server stores one row per user, so every slot that names the
        same other user shares one vector. That vector starts at the
        first payload row for the id. The owner is not included: that
        trajectory is ``user_embedding.weight``. Negative ids are empty
        slots. Both stay as constant payload vectors, added back in
        :meth:`_neighbor_embeddings_from_parameter`. Index 0 of the
        lookup is a zero row and is not a parameter, so padding has no
        gradient.

        The parameter is not registered on the skeleton. The caller
        passes it to this round's optimiser and does not store its
        moments.

        Parameters
        ----------
        neighbor_embeddings : array_like
            Payload rows, shape ``(L, N, H)``.
        neighbor_user_ids : array_like
            User ids, shape ``(L, N)``. Negative entries are empty slots.
        owner_id : int
            This client's user id. Slots with this id are not parameters.
        device : torch.device
            Device for the parameter and the lookup.

        Returns
        -------
        param : torch.nn.Parameter or None
            Shape ``(U, H)`` in first-seen order. ``None`` when no other
            user's id is present.
        user_ids : list of int
            The ``U`` ids, aligned with ``param``.
        index : torch.Tensor or None
            Long tensor of shape ``(L, N)``. Entry ``0`` is the zero row;
            entry ``k + 1`` reads ``param[k]``.
        frozen_slots : torch.Tensor or None
            Shape ``(L, N, H)``. Payload values for the owner and for
            empty slots. ``None`` when every such slot is zero.

        Raises
        ------
        ValueError
            If the arrays are not ``(L, N, H)`` and ``(L, N)``, the width
            is not ``hidden_dim``, ``device`` is missing, or one user id
            is paired with two different payload rows.

        Examples
        --------
        >>> client = FedPerGNNClientAlgorithm(hidden_dim=2)
        >>> emb = np.zeros((2, 2, 2), dtype=np.float32)
        >>> emb[0, 0] = (0.5, -0.25)
        >>> emb[1, 0] = (0.1, 0.2)
        >>> emb[1, 1] = (0.5, -0.25)
        >>> ids = np.array([[4, -1], [0, 4]], dtype=np.int64)
        >>> param, uids, index, frozen = client._bind_neighbour_parameter(
        ...     emb, ids, owner_id=0, device=torch.device("cpu"))
        >>> uids
        [4]
        >>> index.tolist()
        [[1, 0], [0, 1]]
        >>> round(float(frozen[1, 0, 0]), 4)
        0.1
        >>> emb2 = np.zeros((1, 2, 2), dtype=np.float32)
        >>> emb2[0, 0] = (0.5, 0.0)
        >>> emb2[0, 1] = (0.25, 0.0)
        >>> client._bind_neighbour_parameter(
        ...     emb2, np.array([[7, 7]]), 0, torch.device("cpu"))
        Traceback (most recent call last):
            ...
        ValueError: neighbour user 7 has two different payload rows
        """
        embeddings = np.asarray(neighbor_embeddings, dtype=np.float32)
        user_ids = np.asarray(neighbor_user_ids)
        if embeddings.ndim != 3 or user_ids.ndim != 2:
            raise ValueError(
                "neighbour embeddings must have shape (L, N, H) and "
                "neighbour user ids shape (L, N); "
                f"got {embeddings.shape} and {user_ids.shape}"
            )
        if embeddings.shape[:2] != user_ids.shape:
            raise ValueError(
                "neighbour embeddings and user ids disagree on (L, N): "
                f"{embeddings.shape[:2]} vs {tuple(user_ids.shape)}"
            )
        if embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"neighbour embedding width {embeddings.shape[-1]} does not "
                f"match hidden_dim {self.hidden_dim}"
            )
        if device is None:
            raise ValueError("device is required to build neighbour parameters")

        history_len, neighbor_len, _ = embeddings.shape
        lookup = np.zeros((history_len, neighbor_len), dtype=np.int64)
        frozen = np.array(embeddings, dtype=np.float32, copy=True)
        order: List[int] = []
        index_of: Dict[int, int] = {}
        init_rows: List[np.ndarray] = []
        owner = int(owner_id)

        for ell in range(history_len):
            for n in range(neighbor_len):
                uid = int(user_ids[ell, n])
                if uid < 0 or uid == owner:
                    continue
                row = np.asarray(embeddings[ell, n], dtype=np.float32).copy()
                slot = index_of.get(uid)
                if slot is None:
                    slot = len(order)
                    index_of[uid] = slot
                    order.append(uid)
                    init_rows.append(row)
                elif not np.array_equal(init_rows[slot], row):
                    raise ValueError(
                        f"neighbour user {uid} has two different payload rows"
                    )
                lookup[ell, n] = slot + 1
                frozen[ell, n] = 0

        if not order:
            return None, [], None, None

        param = nn.Parameter(
            torch.tensor(np.stack(init_rows), dtype=torch.float32, device=device)
        )
        index = torch.as_tensor(lookup, dtype=torch.long, device=device)
        frozen_slots = None
        if np.any(frozen):
            frozen_slots = torch.tensor(frozen, dtype=torch.float32, device=device)
        return param, order, index, frozen_slots

    def _neighbor_embeddings_from_parameter(
        self,
        param: nn.Parameter,
        index: torch.Tensor,
        frozen_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gather shared neighbour rows into the ``(L, N, H)`` GAT input.

        Parameters
        ----------
        param : torch.nn.Parameter
            Distinct neighbour rows, shape ``(U, H)``.
        index : torch.Tensor
            Long tensor of shape ``(L, N)`` from
            :meth:`_bind_neighbour_parameter`. ``0`` selects a zero row
            that is not ``param``.
        frozen_slots : torch.Tensor or None
            Constant payload rows for the owner and for empty slots,
            shape ``(L, N, H)``. Added to the gathered tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(L, N, H)``. Trainable slots read ``param``.

        Examples
        --------
        >>> client = FedPerGNNClientAlgorithm(hidden_dim=2)
        >>> param = nn.Parameter(torch.tensor([[0.5, -0.25]]))
        >>> index = torch.tensor([[1, 0]])
        >>> built = client._neighbor_embeddings_from_parameter(param, index, None)
        >>> [round(v, 2) for v in built[0, 0].tolist()]
        [0.5, -0.25]
        >>> float(built[0, 1].detach().abs().sum())
        0.0
        """
        # Index 0 is padding. It is not a row of ``param``, so empty
        # slots do not receive a gradient.
        zero = param.new_zeros(1, self.hidden_dim)
        bank = torch.cat([zero, param], dim=0)
        gathered = nn.functional.embedding(index, bank)
        if frozen_slots is None:
            return gathered
        return gathered + frozen_slots

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
