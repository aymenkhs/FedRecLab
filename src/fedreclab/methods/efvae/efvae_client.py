"""EFVAE client algorithms: FCIS base and MultVAE concrete subclass.

Source: Zhang et al., CIKM 2024. The FCIS client base builds hybrid item
sets (``rho=1``), runs a backbone-specific ``compute_fcis_loss``, and
uploads sparse encoder/decoder rows plus a dense middle gradient.
``EFVAEClientAlgorithm`` is the MultVAE objective (sampled softmax +
annealed KL).

Intended later subclasses: MultDAE (CE only) and RecVAE (different
encode / prior). Override ``compute_fcis_loss`` and, if needed,
``compute_scores`` / ``init_model_skeleton``.

Every training row is a positive. Paper binarisation is a load-time
filter, not a client flag.
"""

from __future__ import annotations

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
from src.fedreclab.federated.utils import assert_skeleton_server_coverage, flatten_grouped_state
from src.fedreclab.methods.efvae.efvae_fcis import (
    build_hybrid_items,
    sampled_softmax_ce,
    sparse_row_upload_named,
)
from src.fedreclab.methods.efvae.efvae_model import EFMultVAE
from src.fedreclab.methods.efvae.efvae_sampler import FIMSUniSampler


class EFVAEFCISClientAlgorithm(BaseClientAlgorithm):
    """Shared FCIS client: hybrid items, sparse packs, score skeleton.

    Subclasses implement ``compute_fcis_loss`` and typically
    ``init_model_skeleton`` / ``compute_scores``.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Encoder hidden widths. Default is ``(120,)``.
    latent_dim : int, optional
        Latent size for MultVAE-style backbones. Default is 40.
    dropout : float, optional
        Dropout rate. Default is 0.5.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (120,),
        latent_dim: int = 40,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.num_items: int | None = None
        self._skeleton: EFMultVAE | None = None

    def init_model_skeleton(self) -> None:
        """Build the shared EF-MultVAE module once.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If ``num_items`` or ``device`` is unset.
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
        self._skeleton = EFMultVAE(
            num_items=self.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)
        flat = flatten_grouped_state(self._skeleton.get_server_state_groups())
        assert_skeleton_server_coverage(self._skeleton, set(flat))

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: Any,
        users_in_client: list[int],
    ) -> dict[str, Any]:
        """Store training item ids per user. No model parameters.

        Parameters
        ----------
        client_id : int
            Client id (unused beyond shared skeleton).
        total_num_users, num_users : int
            Unused; present for the base signature.
        num_items : int
            Catalogue size.
        client_train_data_view : pandas.DataFrame
            Training interactions for this client.
        users_in_client : list of int
            Remapped user ids on this client.

        Returns
        -------
        dict
            ``train_items_per_user`` mapping user id to int64 item arrays.
        """
        del client_id, total_num_users, num_users
        if self.num_items is None:
            self.num_items = int(num_items)
        self.init_model_skeleton()
        train_items_per_user: dict[int, np.ndarray] = {
            int(uid): np.array([], dtype=np.int64) for uid in users_in_client
        }
        if client_train_data_view is not None and len(client_train_data_view) > 0:
            grouped = client_train_data_view.groupby("user_id", sort=False)
            for user_id, group in grouped:
                train_items_per_user[int(user_id)] = group["item_id"].to_numpy(
                    dtype=np.int64
                )
        return {"train_items_per_user": train_items_per_user}

    def compute_fcis_loss(
        self,
        model: EFMultVAE,
        pos_items: torch.Tensor,
        lengths: torch.Tensor,
        sampler: FIMSUniSampler,
        anneal: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss, sampled_neg_items)`` for one padded batch.

        Parameters
        ----------
        model : EFMultVAE
            Shared skeleton with server weights loaded.
        pos_items : torch.Tensor
            Padded positives, shape ``(batch, max_len)``.
        lengths : torch.Tensor
            True lengths, shape ``(batch,)``.
        sampler : FIMSUniSampler
            Proposal sampler for this round.
        anneal : float
            KL weight (ignored by MultDAE subclasses).
        generator : torch.Generator or None
            RNG for dropout / reparameterisation.

        Returns
        -------
        loss : torch.Tensor
            Scalar.
        neg_items : torch.Tensor
            Sampled negatives, shape ``(batch, num_neg)``.
        """
        raise NotImplementedError("Subclasses must implement compute_fcis_loss")

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """One FCIS gradient: hybrid items, backbone loss, sparse uploads.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Payload, state, training rows, ParamIO, extras.

        Returns
        -------
        ClientTrainingResponse
            Unchanged client state and update packages for the four
            FCIS groups.

        Raises
        ------
        ValueError
            If sampler state / anneal is missing or a tensor is non-finite.
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

        extra = training_request.extra_payload or {}
        if "fcis_sampler_state" not in extra:
            raise ValueError(
                "extra_payload['fcis_sampler_state'] is required; the server "
                "must rebuild FIMS on round start"
            )
        sampler = FIMSUniSampler(
            num_items=self.num_items,
            num_cluster=1,
            num_neg=1,
        )
        sampler.load_state_dict(extra["fcis_sampler_state"], device=device)
        anneal = float(extra.get("anneal", 1.0))

        items_per_user = self._items_per_user(users_in_client, training_data)
        generator = self._round_generator(
            training_request.seed, client_id, training_request.current_round
        )
        rng = self._numpy_rng(
            training_request.seed, client_id, training_request.current_round
        )

        # One user at a time avoids item-id 0 padding (0 is a real id).
        losses: list[torch.Tensor] = []
        hybrid_all: list[np.ndarray] = []
        sampled_all: list[np.ndarray] = []
        model.zero_grad(set_to_none=True)
        n_users_with_data = 0
        for uid in users_in_client:
            pos_np = items_per_user[int(uid)]
            if pos_np.size == 0:
                continue
            hybrid = build_hybrid_items(pos_np, self.num_items, rng)
            hybrid_all.append(hybrid)
            pos = torch.as_tensor(pos_np, dtype=torch.long, device=device).view(1, -1)
            lengths = torch.tensor([pos_np.size], dtype=torch.long, device=device)
            loss_u, neg = self.compute_fcis_loss(
                model, pos, lengths, sampler, anneal, generator
            )
            self._require_finite_tensor(loss_u, "FCIS loss")
            losses.append(loss_u)
            sampled_all.append(neg.detach().cpu().numpy().reshape(-1))
            n_users_with_data += 1

        if n_users_with_data == 0:
            # No training interactions: upload empty sparse + zero dense.
            loss = torch.zeros((), device=device, requires_grad=True)
            loss.backward()
            mean_loss = 0.0
            row_support = np.array([], dtype=np.int64)
        else:
            loss = torch.stack(losses).mean()
            self._require_finite_tensor(loss, "FCIS mean loss")
            loss.backward()
            mean_loss = float(loss.detach().cpu())
            row_support = np.unique(
                np.concatenate(
                    [*(hybrid_all or [np.array([], dtype=np.int64)]),
                     *(sampled_all or [np.array([], dtype=np.int64)])]
                )
            )

        updates_packages = self._pack_updates(
            training_request, model, row_support
        )
        return ClientTrainingResponse(
            client_id=client_id,
            updated_client_state=training_request.client_state,
            updates_packages=updates_packages,
            training_losses=[mean_loss],
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
        """Score candidates with full decoder logits at ``z = mu``.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Candidate item ids per user.
        global_parameters : dict
            Server groups for this round.
        client_state : ClientStateBase
            Holds ``train_items_per_user``.
        extra_payload : dict
            Unused by MultVAE scoring.

        Returns
        -------
        dict of int to ndarray
            Scores aligned with each user's candidates.
        """
        del client_id, extra_payload
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
                scores[int(user_id)] = (
                    logits[0].index_select(0, idx).detach().cpu().numpy()
                )
        return scores

    def _pack_updates(
        self,
        training_request: ClientTrainingRequest,
        model: EFMultVAE,
        row_support: np.ndarray,
    ) -> dict[str, Any]:
        """Build sparse embedding + dense middle update packages."""
        enc_g = model.enc_embeddings.weight.grad
        dec_g = model.dec_embeddings.weight.grad
        bias_g = model.dec_bias.weight.grad
        if enc_g is None:
            enc_g = torch.zeros_like(model.enc_embeddings.weight)
        if dec_g is None:
            dec_g = torch.zeros_like(model.dec_embeddings.weight)
        if bias_g is None:
            bias_g = torch.zeros_like(model.dec_bias.weight)
        self._require_finite_tensor(enc_g, "enc_embeddings grad")
        self._require_finite_tensor(dec_g, "dec_embeddings grad")
        self._require_finite_tensor(bias_g, "dec_bias grad")

        middle_grad: dict[str, torch.Tensor] = {}
        if model.enc_bias.grad is None:
            middle_grad["enc_bias"] = torch.zeros_like(model.enc_bias).cpu()
        else:
            self._require_finite_tensor(model.enc_bias.grad, "enc_bias grad")
            middle_grad["enc_bias"] = model.enc_bias.grad.detach().cpu().clone()
        for name, param in model.q_layers.named_parameters(prefix="q_layers"):
            if param.grad is None:
                middle_grad[name] = torch.zeros_like(param).cpu()
            else:
                self._require_finite_tensor(param.grad, f"gradient of {name}")
                middle_grad[name] = param.grad.detach().cpu().clone()
        for name, param in model.p_layers.named_parameters(prefix="p_layers"):
            if param.grad is None:
                middle_grad[name] = torch.zeros_like(param).cpu()
            else:
                self._require_finite_tensor(param.grad, f"gradient of {name}")
                middle_grad[name] = param.grad.detach().cpu().clone()

        return {
            "enc_embeddings": self._update_package(
                training_request,
                "enc_embeddings",
                sparse_row_upload_named(
                    enc_g, "enc_embeddings.weight", row_support
                ),
            ),
            "dec_embeddings": self._update_package(
                training_request,
                "dec_embeddings",
                sparse_row_upload_named(
                    dec_g, "dec_embeddings.weight", row_support
                ),
            ),
            "dec_bias": self._update_package(
                training_request,
                "dec_bias",
                sparse_row_upload_named(bias_g, "dec_bias.weight", row_support),
            ),
            "middle": self._update_package(training_request, "middle", middle_grad),
        }

    def _items_per_user(
        self,
        users_in_client: list[int],
        training_data: Any,
    ) -> dict[int, np.ndarray]:
        items_per_user: dict[int, np.ndarray] = {
            int(uid): np.array([], dtype=np.int64) for uid in users_in_client
        }
        if training_data is not None and len(training_data) > 0:
            grouped = training_data.groupby("user_id", sort=False)
            for user_id, group in grouped:
                items_per_user[int(user_id)] = group["item_id"].to_numpy(dtype=np.int64)
        return items_per_user

    def _binary_vector(self, item_ids: np.ndarray, device: Any) -> torch.Tensor:
        x = torch.zeros(self.num_items, device=device)
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
        if seed is None:
            return None
        rng = np.random.default_rng([int(seed), int(client_id), int(current_round)])
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**63 - 1)))
        return generator

    @staticmethod
    def _numpy_rng(
        seed: int | None,
        client_id: int,
        current_round: int,
    ) -> np.random.Generator:
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(
            [int(seed), int(client_id), int(current_round), 91]
        )

    @staticmethod
    def _require_finite_tensor(tensor: torch.Tensor, where: str) -> None:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in {where}")


class EFVAEClientAlgorithm(EFVAEFCISClientAlgorithm):
    """MultVAE FCIS client: sampled softmax + annealed KL.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Must match the server. Default is ``(120,)``.
    latent_dim : int, optional
        Must match the server. Default is 40.
    dropout : float, optional
        Default is 0.5.

    Examples
    --------
    >>> client = EFVAEClientAlgorithm(hidden_dims=(4,), latent_dim=2)
    >>> client.latent_dim
    2
    """

    def compute_fcis_loss(
        self,
        model: EFMultVAE,
        pos_items: torch.Tensor,
        lengths: torch.Tensor,
        sampler: FIMSUniSampler,
        anneal: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MultVAE FCIS loss for one user batch.

        Parameters
        ----------
        model : EFMultVAE
            Shared skeleton.
        pos_items : torch.Tensor
            Shape ``(batch, max_len)``.
        lengths : torch.Tensor
            Shape ``(batch,)``.
        sampler : FIMSUniSampler
            Round sampler.
        anneal : float
            KL weight.
        generator : torch.Generator or None
            Dropout / ``eps`` generator.

        Returns
        -------
        loss : torch.Tensor
            Scalar CE + ``anneal * KL``.
        neg_items : torch.Tensor
            Sampled negatives.
        """
        _, mu, logvar = model.encode_items(pos_items, lengths, generator)
        z = model._reparameterize(mu, logvar, generator)
        h = model.decode_hidden(z)
        max_len = int(pos_items.shape[1])
        pos_mask = (
            torch.arange(max_len, device=pos_items.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        ).to(dtype=h.dtype)
        with torch.no_grad():
            pos_log_prob, neg_items, neg_log_prob = sampler(h, pos_items)
        pos_logit = model.decoder_logits_for_items(h, pos_items)
        neg_logit = model.decoder_logits_for_items(h, neg_items)
        ce = sampled_softmax_ce(
            neg_logit, neg_log_prob, pos_logit, pos_log_prob, pos_mask=pos_mask
        )
        kld = -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )
        if not torch.isfinite(kld):
            raise ValueError("Non-finite KL term")
        return ce + float(anneal) * kld, neg_items
