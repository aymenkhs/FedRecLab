"""FedVAE client: one multinomial-VAE gradient, no local parameters.

Source: Polato, "Federated Variational Autoencoder for Collaborative
Filtering", IJCNN 2021.

Each client holds only the training item ids of its users (needed at
scoring time; ``compute_scores`` is not given interactions). The VAE
weights arrive in ``global_payload`` every round. Local training is one
forward/backward pass on the binary user vectors and uploads the raw
gradient of the whole model. There is no local optimiser and no local
step.

Every training row is treated as a positive. Paper binarisation (for
example ml-1m ratings of 4 or more) is a load-time filter, not a client
flag.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from fedreclab.federated.utils import assert_skeleton_server_coverage
from fedreclab.methods.fedvae.fedvae_model import MultVAE


class FedVAEClientAlgorithm(BaseClientAlgorithm):
    """Local gradient and scoring for FedVAE.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Encoder hidden widths. Must match the server. Default is
        ``(600,)``.
    latent_dim : int, optional
        Size of ``mu`` and ``logvar``. Must match the server. Default is
        200.
    dropout : float, optional
        Input dropout rate. Default is 0.5.
    beta : float, optional
        KL weight in the Mult-VAE ELBO. Default is 0.2 (paper section
        V-A, constant, no annealing).

    Raises
    ------
    ValueError
        If ``beta`` is negative.

    Examples
    --------
    >>> client = FedVAEClientAlgorithm(hidden_dims=(4,), latent_dim=2, beta=0.2)
    >>> client.beta
    0.2
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (600,),
        latent_dim: int = 200,
        dropout: float = 0.5,
        beta: float = 0.2,
    ):
        super().__init__()
        beta = float(beta)
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.beta = beta
        self.num_items = None
        self._skeleton: MultVAE | None = None

    def init_model_skeleton(self) -> None:
        """Build the shared Mult-VAE module once.

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
        self._skeleton = MultVAE(
            num_items=self.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)
        server_names = set(self._skeleton.get_server_state_groups()["vae"])
        assert_skeleton_server_coverage(self._skeleton, server_names)

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: Any,
        users_in_client: list[int],
    ) -> dict[str, Any]:
        """Store each user's training item ids. No model parameters.

        Parameters
        ----------
        client_id : int
            Client being initialised. Unused beyond the shared skeleton.
        total_num_users : int
            Unused; present for the base signature.
        num_users : int
            Unused; present for the base signature.
        num_items : int
            Catalogue size after remapping.
        client_train_data_view : pandas.DataFrame
            Training interactions for this client. Used only to record
            item ids per user.
        users_in_client : list of int
            Remapped user ids owned by this client.

        Returns
        -------
        dict
            ``train_items_per_user`` mapping each user id to a 1-d
            integer array of training item ids (empty if the user has
            none).
        """
        if self.num_items is None:
            self.num_items = num_items
        self.init_model_skeleton()

        train_items_per_user: dict[int, np.ndarray] = {
            int(uid): np.array([], dtype=np.int64) for uid in users_in_client
        }
        if client_train_data_view is not None and len(client_train_data_view) > 0:
            grouped = client_train_data_view.groupby("user_id", sort=False)
            for user_id, group in grouped:
                train_items_per_user[int(user_id)] = (
                    group["item_id"].to_numpy(dtype=np.int64)
                )
        return {"train_items_per_user": train_items_per_user}

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """Compute one Mult-VAE gradient on the client's training users.

        The payload is loaded onto the shared skeleton and not written.
        Dropout and ``eps`` use a generator seeded from
        ``(seed, client_id, round)`` when ``seed`` is set.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Payload, client state, training rows, and effective ParamIO.

        Returns
        -------
        ClientTrainingResponse
            Unchanged client state and a dense ``vae`` gradient package.
            ``num_training_examples`` is the number of training
            interaction rows.

        Raises
        ------
        ValueError
            If the loss or any gradient is non-finite.
        """
        t0 = time.time()
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device
        client_id = training_request.client_id
        training_data = training_request.training_data
        users_in_client = list(training_request.users_in_client)

        model.load_server_weights(training_request.global_payload)
        model.train()

        batch = self._user_item_batch(
            users_in_client, training_data, device=device
        )
        generator = self._round_generator(
            training_request.seed, client_id, training_request.current_round
        )
        model.zero_grad(set_to_none=True)
        logits, mu, logvar = model(batch, generator=generator)
        loss = self._mult_vae_loss(logits, batch, mu, logvar)
        self._require_finite_tensor(loss, "Mult-VAE loss")
        loss.backward()

        grad = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                value = torch.zeros_like(param, device="cpu")
            else:
                self._require_finite_tensor(param.grad, f"gradient of {name}")
                value = param.grad.detach().cpu().clone()
            grad[name] = value

        updates_packages = {
            "vae": self._update_package(training_request, "vae", grad)
        }
        return ClientTrainingResponse(
            client_id=client_id,
            updated_client_state=training_request.client_state,
            updates_packages=updates_packages,
            training_losses=[float(loss.detach().cpu())],
            training_duration=time.time() - t0,
            num_training_examples=int(len(training_data)),
        )

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: dict[int, np.ndarray],
        global_parameters: dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: dict[str, Any],
    ) -> dict[int, np.ndarray]:
        """Score candidates with decoder logits at ``z = mu``.

        The input history is the stored training items of each user. An
        empty history is a zero vector and still yields finite scores.
        Training items that appear in the candidate list are scored, not
        masked; exclusion is the evaluator's job.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Server-resident VAE group for this round.
        client_state : ClientStateBase
            Holds ``train_items_per_user``.
        extra_payload : dict
            Unused by FedVAE scoring.

        Returns
        -------
        dict of int to ndarray
            One score array per user, aligned with that user's candidates.

        Examples
        --------
        >>> scores = client.compute_scores(0, {0: np.array([1, 2])}, payload, state, {})  # doctest: +SKIP
        >>> scores[0].shape  # doctest: +SKIP
        (2,)
        """
        self.init_model_skeleton()
        model = self._skeleton
        device = self.device
        model.load_server_weights(global_parameters)
        model.eval()
        histories = client_state.local_parameters["train_items_per_user"]
        scores: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                items = histories.get(int(user_id), np.array([], dtype=np.int64))
                x = self._binary_vector(items, device)
                logits, _, _ = model(x.unsqueeze(0))
                idx = torch.as_tensor(candidate_items, dtype=torch.long, device=device)
                scores[int(user_id)] = logits[0].index_select(0, idx).detach().cpu().numpy()
        return scores

    def _mult_vae_loss(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Mean multinomial reconstruction plus ``beta`` times mean KL.

        The reconstruction target is the undropped ``x``. Matches the
        authors' rectorch loss (mean over users of the per-user sums).

        Parameters
        ----------
        logits : torch.Tensor
            Decoder output, shape ``(batch, num_items)``.
        x : torch.Tensor
            Binary user vectors, same shape.
        mu, logvar : torch.Tensor
            Variational parameters, shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        recon = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * x, dim=1))
        kld = -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        return recon + self.beta * kld

    def _user_item_batch(
        self,
        users_in_client: list[int],
        training_data: Any,
        device: Any,
    ) -> torch.Tensor:
        """Stack binary user vectors for every user on this client.

        Users missing from ``training_data`` get a zero vector.

        Parameters
        ----------
        users_in_client : list of int
            Remapped user ids.
        training_data : pandas.DataFrame
            Training rows with ``user_id`` and ``item_id``.
        device : torch.device or str
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(n_users, num_items)``.
        """
        items_per_user: dict[int, np.ndarray] = {
            int(uid): np.array([], dtype=np.int64) for uid in users_in_client
        }
        if training_data is not None and len(training_data) > 0:
            grouped = training_data.groupby("user_id", sort=False)
            for user_id, group in grouped:
                items_per_user[int(user_id)] = group["item_id"].to_numpy(dtype=np.int64)
        rows = [
            self._binary_vector(items_per_user[int(uid)], device)
            for uid in users_in_client
        ]
        if not rows:
            return torch.zeros((0, self.num_items), device=device)
        return torch.stack(rows, dim=0)

    def _binary_vector(self, item_ids: np.ndarray, device: Any) -> torch.Tensor:
        """Return a length-``num_items`` vector with ones at ``item_ids``.

        Parameters
        ----------
        item_ids : ndarray
            Item ids to mark. Duplicates are still ones.
        device : torch.device or str
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(num_items,)``.
        """
        x = torch.zeros(self.num_items, device=device)
        if item_ids is None:
            return x
        ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return x
        idx = torch.as_tensor(ids, dtype=torch.long, device=device)
        x[idx] = 1.0
        return x

    @staticmethod
    def _round_generator(
        seed: int | None,
        client_id: int,
        current_round: int,
    ) -> torch.Generator | None:
        """CPU generator seeded from ``(seed, client_id, round)``.

        Parameters
        ----------
        seed : int or None
            Experiment seed. ``None`` returns ``None`` (default RNG).
        client_id, current_round : int
            Folded into the seed sequence.

        Returns
        -------
        torch.Generator or None
        """
        if seed is None:
            return None
        rng = np.random.default_rng([int(seed), int(client_id), int(current_round)])
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**63 - 1)))
        return generator

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
