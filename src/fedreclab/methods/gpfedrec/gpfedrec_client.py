"""GPFedRec client: local item table, user vectors and personal score function.

Source: Zhang, Long, Zhou, Zhang, Yan and Yang, "GPFedRec: Graph-Guided
Personalization for Federated Recommendation", KDD 2024; reference code
``GPFedRec-main`` (``engine.py``, ``mlp.py``, ``train.py``).

Each round a selected client:

1. starts its item table from the server's global table ``q_global``
   (payload key ``item_embeddings``);
2. trains user vectors, item table and score function with plain SGD on
   ``BCE + reg * MSE(item_table, r_i)``, where ``r_i`` is the
   user-specific item table the server computed for this client in its
   last participating round (payload key
   ``personalized_item_embeddings``; equal to ``q_global`` before then);
3. uploads the full trained item table, optionally with Laplace noise.

The user vectors, score function and the noise-free trained item table
stay on the client. The stored table is used for scoring. A client that
has never trained scores with ``q_global``.

Every training row is a positive. Negatives are drawn uniformly from the
items the user has no training interaction with.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from fedreclab.methods.gpfedrec.gpfedrec_model import (
    GPFedRecScoreFunction,
    build_client_template,
    predict_logits,
)


class GPFedRecClientAlgorithm(BaseClientAlgorithm):
    """Local training and scoring for GPFedRec.

    Learning rates follow the reference ``engine.py``: the score function
    uses ``lr``, the user vectors use
    ``lr / participation_ratio * lr_eta - lr`` and the item table uses
    ``lr * num_items * lr_eta - lr``. ``user_lr`` and ``item_lr`` replace
    these formulas when given.

    Parameters
    ----------
    latent_dim : int, optional
        Width of user and item vectors. Must match the server. Default 32.
    hidden_layers : sequence of int, optional
        Score-function widths; the first must be ``2 * latent_dim``.
        Default ``(64, 32, 16, 8)``.
    local_epochs : int, optional
        Passes over the local samples per round. Default 1.
    batch_size : int, optional
        Mini-batch size. Default 256.
    neg_samples : int, optional
        Negatives drawn per positive, redrawn every epoch. Default 4.
    lr : float, optional
        Base SGD learning rate (score function). Default 0.1.
    lr_eta : float, optional
        Multiplier in the user and item learning-rate formulas. Default 80.
    participation_ratio : float, optional
        Fraction of clients selected per round, used only by the user
        learning-rate formula. The GPFedRec server checks it against its
        selector. Default 1.0.
    user_lr : float or None, optional
        Explicit user-vector learning rate. ``None`` uses the formula.
    item_lr : float or None, optional
        Explicit item-table learning rate. ``None`` uses the formula.
    reg : float, optional
        Weight of ``MSE(item_table, r_i)``. Default 1.0.
    ldp_laplace_scale : float, optional
        Scale of the Laplace noise added to the uploaded table. ``0``
        (default) disables noise.
    init_seed : int or None, optional
        Seed of the shared initial user vector and score function. The
        GPFedRec server sets it from its own ``seed`` when left ``None``.

    Raises
    ------
    ValueError
        If a hyper-parameter is out of range, or the formula gives a
        non-positive user learning rate.

    Examples
    --------
    >>> client = GPFedRecClientAlgorithm(latent_dim=4, hidden_layers=(8, 4))
    >>> round(client.effective_user_lr(), 6)
    7.9
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_layers: Sequence[int] = (64, 32, 16, 8),
        local_epochs: int = 1,
        batch_size: int = 256,
        neg_samples: int = 4,
        lr: float = 0.1,
        lr_eta: float = 80.0,
        participation_ratio: float = 1.0,
        user_lr: float | None = None,
        item_lr: float | None = None,
        reg: float = 1.0,
        ldp_laplace_scale: float = 0.0,
        init_seed: int | None = None,
    ):
        super().__init__()
        hidden_layers = tuple(int(h) for h in hidden_layers)
        if len(hidden_layers) < 1 or hidden_layers[0] != 2 * int(latent_dim):
            raise ValueError(
                f"hidden_layers must start with 2 * latent_dim = {2 * int(latent_dim)}, "
                f"got {hidden_layers}"
            )
        for name, value in (
            ("local_epochs", local_epochs),
            ("batch_size", batch_size),
            ("neg_samples", neg_samples),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if float(lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 < float(participation_ratio) <= 1.0:
            raise ValueError(
                f"participation_ratio must be in (0, 1], got {participation_ratio}"
            )
        for name, value in (("user_lr", user_lr), ("item_lr", item_lr)):
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0 when given, got {value}")
        if float(reg) < 0.0:
            raise ValueError(f"reg must be >= 0, got {reg}")
        if float(ldp_laplace_scale) < 0.0:
            raise ValueError(f"ldp_laplace_scale must be >= 0, got {ldp_laplace_scale}")

        self.latent_dim = int(latent_dim)
        self.hidden_layers = hidden_layers
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.neg_samples = int(neg_samples)
        self.lr = float(lr)
        self.lr_eta = float(lr_eta)
        self.participation_ratio = float(participation_ratio)
        self.user_lr = None if user_lr is None else float(user_lr)
        self.item_lr = None if item_lr is None else float(item_lr)
        self.reg = float(reg)
        self.ldp_laplace_scale = float(ldp_laplace_scale)
        self.init_seed = init_seed
        self.num_items: int | None = None
        self._template_user_vector: torch.Tensor | None = None
        self._template_score_state: dict[str, torch.Tensor] | None = None
        self._skeleton: GPFedRecScoreFunction | None = None
        self.effective_user_lr()

    def user_lr_for_ratio(self, participation_ratio: float) -> float:
        """Return the reference user learning rate for a participation ratio.

        Parameters
        ----------
        participation_ratio : float
            Fraction of clients selected per round.

        Returns
        -------
        float
            ``lr / participation_ratio * lr_eta - lr``.

        Examples
        --------
        >>> GPFedRecClientAlgorithm(latent_dim=4, hidden_layers=(8,)).user_lr_for_ratio(0.5)
        15.9
        """
        return self.lr / float(participation_ratio) * self.lr_eta - self.lr

    def effective_user_lr(self) -> float:
        """Return the user-vector learning rate used in local training.

        Returns
        -------
        float
            ``user_lr`` if set, else :meth:`user_lr_for_ratio` at
            ``participation_ratio``.

        Raises
        ------
        ValueError
            If the formula gives a non-positive value.
        """
        if self.user_lr is not None:
            return self.user_lr
        value = self.user_lr_for_ratio(self.participation_ratio)
        if value <= 0.0:
            raise ValueError(
                f"user learning rate lr / participation_ratio * lr_eta - lr = {value} "
                "is not positive; increase lr_eta or pass user_lr"
            )
        return value

    def effective_item_lr(self) -> float:
        """Return the item-table learning rate used in local training.

        Returns
        -------
        float
            ``item_lr`` if set, else ``lr * num_items * lr_eta - lr``.

        Raises
        ------
        RuntimeError
            If the formula is needed and ``num_items`` is not known yet.
        ValueError
            If the formula gives a non-positive value.
        """
        if self.item_lr is not None:
            return self.item_lr
        if self.num_items is None:
            raise RuntimeError(
                "num_items is unknown; call initialize_local_parameters first"
            )
        value = self.lr * self.num_items * self.lr_eta - self.lr
        if value <= 0.0:
            raise ValueError(
                f"item learning rate lr * num_items * lr_eta - lr = {value} "
                "is not positive; increase lr_eta or pass item_lr"
            )
        return value

    def init_model_skeleton(self) -> None:
        """Build the shared template and the score-function skeleton once.

        The template comes from :func:`build_client_template` with
        ``init_seed``. The skeleton is a copy of the template module on
        ``device``; its weights are overwritten from client state on every
        use, and building it draws no random numbers.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``init_seed`` is ``None``.
        RuntimeError
            If ``device`` has not been set.
        """
        if self._skeleton is not None:
            return
        if self.init_seed is None:
            raise ValueError(
                "GPFedRecClientAlgorithm.init_seed is None; pass seed= on the "
                "GPFedRec server or init_seed= on the client"
            )
        if self.device is None:
            raise RuntimeError(
                "device is not set. The server assigns client.device from "
                "server.device before initialise; standalone callers must set it."
            )
        user_vector, score_function = build_client_template(
            self.latent_dim, self.hidden_layers, int(self.init_seed)
        )
        self._template_user_vector = user_vector
        self._template_score_state = score_function.get_state()
        self._skeleton = copy.deepcopy(score_function).to(self.device)

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: Any,
        users_in_client: list[int],
    ) -> tuple[dict[str, Any], None]:
        """Give the client copies of the shared template. No item table yet.

        Parameters
        ----------
        client_id : int
            Client being initialised. Unused beyond the shared template.
        total_num_users : int
            Unused; present for the base signature.
        num_users : int
            Number of users on this client.
        num_items : int
            Catalogue size after remapping.
        client_train_data_view : pandas.DataFrame
            Unused; negatives are built from each round's training rows.
        users_in_client : list of int
            Remapped user ids owned by this client.

        Returns
        -------
        local_parameters : dict
            ``user_embeddings`` (CPU tensor ``(num_users, latent_dim)``,
            every row the template user vector), ``score_function`` (CPU
            state dict), ``local_item_embeddings`` (``None`` until the first
            training) and ``user_id_to_local`` (user id to row index).
        optimizer_state : None
            Plain SGD keeps no state between rounds.

        Raises
        ------
        ValueError
            If ``init_seed`` is ``None``, or ``num_items`` differs from an
            earlier call.
        """
        if self.num_items is None:
            self.num_items = int(num_items)
        elif self.num_items != int(num_items):
            raise ValueError(
                f"num_items changed from {self.num_items} to {num_items} "
                "between clients"
            )
        self.init_model_skeleton()
        user_ids = [int(u) for u in users_in_client]
        local_parameters = {
            "user_embeddings": self._template_user_vector.unsqueeze(0)
            .repeat(len(user_ids), 1)
            .clone(),
            "score_function": {
                name: value.clone() for name, value in self._template_score_state.items()
            },
            "local_item_embeddings": None,
            "user_id_to_local": {uid: row for row, uid in enumerate(user_ids)},
        }
        return local_parameters, None

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """Train on ``BCE + reg * MSE(item_table, r_i)`` and upload the table.

        The payload tensors are cloned before training, so the server's
        ``q_global`` and ``r_i`` are never modified.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            ``global_payload`` must hold ``item_embeddings`` (``q_global``)
            and ``personalized_item_embeddings`` (``r_i``), both of shape
            ``(num_items, latent_dim)``. ``seed`` must be set.

        Returns
        -------
        ClientTrainingResponse
            Updated state (user vectors, score function, noise-free item
            table) and a dense ``item_embeddings`` package holding a CPU
            float32 table, with Laplace noise when
            ``ldp_laplace_scale > 0``. ``training_losses`` holds the mean
            loss of each epoch; ``num_training_examples`` counts the
            positive and negative samples over all epochs.

        Raises
        ------
        KeyError
            If a payload entry is missing.
        ValueError
            If ``seed`` is ``None``, a payload table has the wrong shape, a
            training user is not on this client, a user has interacted with
            every item (no negatives exist), or the loss is non-finite.
        """
        t0 = time.time()
        self.init_model_skeleton()
        device = self.device
        seed = training_request.seed
        if seed is None:
            raise ValueError(
                "GPFedRec local training needs training_request.seed for "
                "negative sampling and noise"
            )
        client_id = int(training_request.client_id)
        current_round = int(training_request.current_round)
        payload = training_request.global_payload
        for key in ("item_embeddings", "personalized_item_embeddings"):
            if key not in payload:
                raise KeyError(f"GPFedRec payload is missing '{key}'")
        state = training_request.client_state
        local = state.local_parameters
        user_id_to_local = local["user_id_to_local"]

        item_table = nn.Parameter(
            self._payload_table(payload["item_embeddings"], "item_embeddings")
        )
        target = self._payload_table(
            payload["personalized_item_embeddings"], "personalized_item_embeddings"
        )
        user_table = nn.Parameter(
            torch.as_tensor(local["user_embeddings"], dtype=torch.float32)
            .detach()
            .clone()
            .to(device)
        )
        model = self._skeleton
        model.load_state(local["score_function"])
        model.train()

        optimizer = torch.optim.SGD(
            [
                {"params": list(model.parameters()), "lr": self.lr},
                {"params": [user_table], "lr": self.effective_user_lr()},
                {"params": [item_table], "lr": self.effective_item_lr()},
            ]
        )

        rng = np.random.default_rng([int(seed), client_id, current_round])
        positives = self._positives_per_local_user(
            training_request.training_data, user_id_to_local
        )
        epoch_losses: list[float] = []
        num_examples = 0
        for _ in range(self.local_epochs):
            users, items, labels = self._build_epoch_samples(positives, rng)
            num_examples += int(users.size)
            if users.size == 0:
                continue
            users_t = torch.as_tensor(users, dtype=torch.long, device=device)
            items_t = torch.as_tensor(items, dtype=torch.long, device=device)
            labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
            loss_sum = 0.0
            for start in range(0, users.size, self.batch_size):
                stop = start + self.batch_size
                optimizer.zero_grad(set_to_none=True)
                logits = predict_logits(
                    model, users_t[start:stop], items_t[start:stop], user_table, item_table
                )
                loss = F.binary_cross_entropy_with_logits(logits, labels_t[start:stop])
                loss = loss + self.reg * F.mse_loss(item_table, target)
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Non-finite GPFedRec loss on client {client_id} in round "
                        f"{current_round}: {float(loss.detach())}"
                    )
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * (min(stop, users.size) - start)
            epoch_losses.append(loss_sum / users.size)

        trained_table = item_table.detach().cpu().clone()
        upload = trained_table.clone()
        if self.ldp_laplace_scale > 0.0:
            noise_rng = np.random.default_rng([int(seed), client_id, current_round, 1])
            noise = noise_rng.laplace(0.0, self.ldp_laplace_scale, size=tuple(upload.shape))
            upload = upload + torch.from_numpy(noise.astype(np.float32))

        new_local = dict(local)
        new_local["user_embeddings"] = user_table.detach().cpu().clone()
        new_local["score_function"] = model.get_state()
        new_local["local_item_embeddings"] = trained_table
        updated_state = ClientStateBase(
            client_id=state.client_id,
            local_parameters=new_local,
            optimizer_state=None,
            metadata=state.metadata,
        )
        return ClientTrainingResponse(
            client_id=client_id,
            updated_client_state=updated_state,
            updates_packages={
                "item_embeddings": self._update_package(
                    training_request, "item_embeddings", upload
                )
            },
            training_losses=epoch_losses,
            training_duration=time.time() - t0,
            num_training_examples=num_examples,
        )

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: dict[int, np.ndarray],
        global_parameters: dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: dict[str, Any],
    ) -> dict[int, np.ndarray]:
        """Score candidates with the client's own model and item table.

        Uses the stored trained item table; a client that has never
        trained uses ``global_parameters["item_embeddings"]``. Scores are
        logits.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Holds ``item_embeddings`` (``q_global``), used as fallback.
        client_state : ClientStateBase
            The client's user vectors, score function and item table.
        extra_payload : dict
            Unused by GPFedRec scoring.

        Returns
        -------
        dict of int to ndarray
            One float score array per user, aligned with its candidates.

        Raises
        ------
        KeyError
            If a user is not on this client.

        Examples
        --------
        >>> scores = client.compute_scores(0, {0: np.array([1, 2])}, params, state, {})  # doctest: +SKIP
        >>> scores[0].shape  # doctest: +SKIP
        (2,)
        """
        self.init_model_skeleton()
        device = self.device
        local = client_state.local_parameters
        table = local.get("local_item_embeddings")
        if table is None:
            table = global_parameters["item_embeddings"]
        item_table = torch.as_tensor(table, dtype=torch.float32).to(device)
        user_table = torch.as_tensor(local["user_embeddings"], dtype=torch.float32).to(device)
        user_id_to_local = local["user_id_to_local"]
        model = self._skeleton
        model.load_state(local["score_function"])
        model.eval()
        scores: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                if int(user_id) not in user_id_to_local:
                    raise KeyError(f"User {user_id} is not on client {client_id}")
                items = torch.as_tensor(
                    np.asarray(candidate_items, dtype=np.int64), device=device
                )
                users = torch.full_like(items, user_id_to_local[int(user_id)])
                logits = predict_logits(model, users, items, user_table, item_table)
                scores[int(user_id)] = logits.cpu().numpy()
        return scores

    def _payload_table(self, value: Any, name: str) -> torch.Tensor:
        """Return an owned float32 copy of a payload table on ``device``.

        Parameters
        ----------
        value : torch.Tensor or array-like
            Payload table.
        name : str
            Payload key, used in error messages.

        Returns
        -------
        torch.Tensor
            Copy of shape ``(num_items, latent_dim)`` that shares no memory
            with ``value``.

        Raises
        ------
        ValueError
            If the shape is not ``(num_items, latent_dim)``.
        """
        table = torch.as_tensor(value, dtype=torch.float32).detach().to(self.device).clone()
        expected = (self.num_items, self.latent_dim)
        if tuple(table.shape) != expected:
            raise ValueError(
                f"Payload '{name}' has shape {tuple(table.shape)}, expected {expected}"
            )
        return table

    def _positives_per_local_user(
        self,
        training_data: pd.DataFrame | None,
        user_id_to_local: dict[int, int],
    ) -> dict[int, np.ndarray]:
        """Group training item ids by local user row.

        Parameters
        ----------
        training_data : pandas.DataFrame or None
            Training rows with ``user_id`` and ``item_id``.
        user_id_to_local : dict of int to int
            Remapped user id to row of the client's user table.

        Returns
        -------
        dict of int to ndarray
            Local row to an int64 array of positive item ids, in data order.
            Users without training rows are absent.

        Raises
        ------
        ValueError
            If a training row belongs to a user not on this client.
        """
        if training_data is None or len(training_data) == 0:
            return {}
        positives: dict[int, np.ndarray] = {}
        for user_id, group in training_data.groupby("user_id", sort=True):
            if int(user_id) not in user_id_to_local:
                raise ValueError(f"Training row for user {user_id} not on this client")
            positives[user_id_to_local[int(user_id)]] = group["item_id"].to_numpy(
                dtype=np.int64
            )
        return positives

    def _build_epoch_samples(
        self,
        positives: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw negatives and return one shuffled epoch of samples.

        Each positive gets ``neg_samples`` negatives drawn uniformly, with
        replacement, from the items the user has no training row for.

        Parameters
        ----------
        positives : dict of int to ndarray
            Output of :meth:`_positives_per_local_user`.
        rng : numpy.random.Generator
            Seeded generator for negatives and shuffling.

        Returns
        -------
        users, items : ndarray
            int64 arrays of local user rows and item ids.
        labels : ndarray
            float32 array, 1 for positives and 0 for negatives.

        Raises
        ------
        ValueError
            If a user has a training row for every item.
        """
        users, items, labels = [], [], []
        for row, pos_items in positives.items():
            excluded = np.unique(pos_items)
            if excluded.size >= self.num_items:
                raise ValueError(
                    f"Local user row {row} has interacted with every item; "
                    "no negatives can be drawn"
                )
            n_neg = pos_items.size * self.neg_samples
            # Draw a rank among the allowed items and map it to the item id v
            # with v = rank + #{excluded <= v} (least fixed point, reached by
            # iterating from v = rank). Uniform over allowed items, no retries.
            ranks = rng.integers(0, self.num_items - excluded.size, size=n_neg)
            negatives = ranks
            while True:
                shifted = ranks + np.searchsorted(excluded, negatives, side="right")
                if np.array_equal(shifted, negatives):
                    break
                negatives = shifted
            users.append(np.full(pos_items.size + n_neg, row, dtype=np.int64))
            items.append(np.concatenate([pos_items, negatives]).astype(np.int64))
            labels.append(
                np.concatenate(
                    [np.ones(pos_items.size, np.float32), np.zeros(n_neg, np.float32)]
                )
            )
        if not users:
            empty = np.array([], dtype=np.int64)
            return empty, empty, np.array([], dtype=np.float32)
        users_arr = np.concatenate(users)
        items_arr = np.concatenate(items)
        labels_arr = np.concatenate(labels)
        order = rng.permutation(users_arr.size)
        return users_arr[order], items_arr[order], labels_arr[order]
