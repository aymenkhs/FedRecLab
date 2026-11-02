"""MetaMF client: gradient of MSE on generated per-user parameters.

Source: Lin et al., "Meta Matrix Factorization for Federated Rating
Predictions", SIGIR 2020.

The device holds no trainable parameters. Each round it receives the
generated item factors and rating MLP for its users, computes one MSE
on a local rating batch, and uploads the gradient of that loss with
respect to the generated parameters of each user. Only the item-factor
rows of items that appeared in the batch are uploaded.

``compute_scores`` regenerates those parameters from the current Theta
on a shared skeleton (a simulation shortcut; Algorithm 1 regenerates
after every server step). The skeleton is algorithm-level scratch
space, not per-client state. It is reloaded only when the Theta dict
identity changes, and candidate subsets slice ``emb_layer_1`` instead
of generating the full catalogue.

Theta is never mutated in place: every server update assigns a new
dict. The cache holds a reference to the dict it last loaded so a
recycled ``id()`` cannot serve stale weights.
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
from src.fedreclab.federated.utils import assert_skeleton_server_coverage
from src.fedreclab.methods.metamf.metamf_model import (
    FACTOR_PHI_KEYS,
    FULL_PHI_KEYS,
    ITEM_PAYLOADS,
    MetaRecommender,
    apply_rating_mlp,
    predict_ratings,
    resolve_rp_hidden_dim,
    unbatch_phi,
)


class MetaMFClientAlgorithm(BaseClientAlgorithm):
    """Local gradient and scoring for MetaMF.

    Parameters
    ----------
    user_emb_dim : int, optional
        Width of ``e_u``. Must match the server. Default is 32.
    item_emb_dim : int, optional
        Width of generated item embeddings. Default is 32.
    item_mem_num : int, optional
        Low-dimensional item width ``s``. Default is 8.
    mem_size : int, optional
        Collaborative-vector width. Default is 128.
    hidden_size : int, optional
        Meta-recommender hidden width. Default is 512.
    rp_hidden_dim : int or None, optional
        Rating-MLP hidden width. ``None`` (default) is
        ``item_emb_dim // 4``.
    item_payload : {"factors", "full"}, optional
        Download and upload form of the item factors. Default is
        ``"factors"``.
    local_batch_size : int or None, optional
        Ratings drawn per user each round. ``None`` (default) uses every
        local training rating. An integer samples that many draws per
        user; users with fewer ratings and ``local_batch_replacement``
        False use all of theirs.
    local_batch_replacement : bool, optional
        If True, the per-user sample is with replacement. Default is
        False. A repeated item counts once per draw in the loss; the
        upload still has unique rows because autograd sums those draws.

    Raises
    ------
    ValueError
        If ``item_payload`` is unknown, ``local_batch_size`` is not a
        positive int or ``None``, or the resolved rating hidden width
        is below 1.

    Examples
    --------
    >>> client = MetaMFClientAlgorithm(item_emb_dim=8)
    >>> client.rp_hidden_dim
    2
    """

    def __init__(
        self,
        user_emb_dim: int = 32,
        item_emb_dim: int = 32,
        item_mem_num: int = 8,
        mem_size: int = 128,
        hidden_size: int = 512,
        rp_hidden_dim: int | None = None,
        item_payload: str = "factors",
        local_batch_size: int | None = None,
        local_batch_replacement: bool = False,
    ):
        super().__init__()
        if item_payload not in ITEM_PAYLOADS:
            raise ValueError(
                f"item_payload must be one of {ITEM_PAYLOADS}, got {item_payload!r}"
            )
        if local_batch_size is not None:
            if (
                not isinstance(local_batch_size, int)
                or isinstance(local_batch_size, bool)
                or local_batch_size < 1
            ):
                raise ValueError(
                    f"local_batch_size must be a positive int or None, "
                    f"got {local_batch_size!r}"
                )
        self.user_emb_dim = int(user_emb_dim)
        self.item_emb_dim = int(item_emb_dim)
        self.item_mem_num = int(item_mem_num)
        self.mem_size = int(mem_size)
        self.hidden_size = int(hidden_size)
        self.rp_hidden_dim = resolve_rp_hidden_dim(self.item_emb_dim, rp_hidden_dim)
        self.item_payload = item_payload
        self.local_batch_size = local_batch_size
        self.local_batch_replacement = bool(local_batch_replacement)
        self.num_users = None
        self.num_items = None
        self._skeleton: MetaRecommender | None = None
        self._loaded_theta: dict[str, Any] | None = None

    def init_model_skeleton(self) -> None:
        """Build the shared meta-network module once.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If ``num_users``, ``num_items``, or ``device`` has not been set.
        """
        if self._skeleton is not None:
            return
        if self.num_users is None or self.num_items is None:
            raise RuntimeError(
                "init_model_skeleton requires num_users and num_items to be set"
            )
        if self.device is None:
            raise RuntimeError(
                "device is not set. The server assigns client.device from "
                "server.device before initialise; standalone callers must set it."
            )
        self._skeleton = MetaRecommender(
            num_users=self.num_users,
            num_items=self.num_items,
            user_emb_dim=self.user_emb_dim,
            item_emb_dim=self.item_emb_dim,
            item_mem_num=self.item_mem_num,
            mem_size=self.mem_size,
            hidden_size=self.hidden_size,
            rp_hidden_dim=self.rp_hidden_dim,
        ).to(self.device)
        server_names = set(self._skeleton.get_server_state_groups()["meta_network"])
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
        """Record catalogue sizes and build the shared skeleton. No local tensors.

        Parameters
        ----------
        client_id : int
            Client being initialised. Unused beyond the shared skeleton.
        total_num_users : int
            Catalogue of users; the skeleton embedding covers all of them.
        num_users : int
            Unused; present for the base signature.
        num_items : int
            Catalogue size after remapping.
        client_train_data_view : pandas.DataFrame
            Unused at init; ratings are required at train time.
        users_in_client : list of int
            Unused; present for the base signature.

        Returns
        -------
        dict
            Empty. MetaMF has no client-resident trainable parameters.
        """
        del client_id, num_users, client_train_data_view, users_in_client
        if self.num_users is None:
            self.num_users = int(total_num_users)
        if self.num_items is None:
            self.num_items = int(num_items)
        self.init_model_skeleton()
        return {}

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        """Compute one MSE on generated parameters and upload per-user dPhi.

        Parameters
        ----------
        training_request : ClientTrainingRequest
            Payload with ``generated_parameters`` keyed by user id,
            training rows, and the effective ParamIO map.

        Returns
        -------
        ClientTrainingResponse
            Unchanged client state and a ``meta_network`` package whose
            value is ``{user_id: dPhi_u}``. ``num_training_examples`` is
            the number of ratings used.

        Raises
        ------
        ValueError
            If the training table has no ``rating`` column, the payload
            is missing a user, or the loss or a gradient is non-finite.
        """
        t0 = time.time()
        training_data = training_request.training_data
        if "rating" not in training_data.columns:
            raise ValueError(
                "MetaMF requires a 'rating' column; implicit all-ones "
                "targets are not used."
            )

        payload = training_request.global_payload.get("generated_parameters", {})
        generator = self._round_rng(
            training_request.seed,
            training_request.client_id,
            training_request.current_round,
        )
        device = self.device
        per_user_samples = self._samples_per_user(
            training_request.users_in_client, training_data, generator
        )

        leaves: dict[int, dict[str, torch.Tensor]] = {}
        pred_chunks: list[torch.Tensor] = []
        rating_chunks: list[torch.Tensor] = []
        n_examples = 0
        for user_id, items, ratings in per_user_samples:
            if user_id not in payload:
                raise ValueError(
                    f"generated_parameters is missing user_id={user_id}"
                )
            phi_leaves, preds = self._user_forward(payload[user_id], items, device)
            leaves[user_id] = phi_leaves
            pred_chunks.append(preds)
            rating_chunks.append(
                torch.as_tensor(ratings, dtype=preds.dtype, device=device)
            )
            n_examples += int(len(items))

        if n_examples == 0:
            loss_value = 0.0
            upload: dict[int, dict[str, Any]] = {}
        else:
            preds = torch.cat(pred_chunks, dim=0)
            ratings = torch.cat(rating_chunks, dim=0)
            loss = torch.mean((preds - ratings) ** 2)
            self._require_finite_tensor(loss, "MetaMF MSE")
            loss.backward()
            loss_value = float(loss.detach().cpu())
            upload = {
                user_id: self._pack_dphi(phi_leaves)
                for user_id, phi_leaves in leaves.items()
            }

        updates_packages = {
            "meta_network": self._update_package(
                training_request, "meta_network", upload
            )
        }
        return ClientTrainingResponse(
            client_id=training_request.client_id,
            updated_client_state=training_request.client_state,
            updates_packages=updates_packages,
            training_losses=[loss_value],
            training_duration=time.time() - t0,
            num_training_examples=n_examples,
        )

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: dict[int, np.ndarray],
        global_parameters: dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: dict[str, Any],
    ) -> dict[int, np.ndarray]:
        """Score candidates as predicted ratings from regenerated Phi.

        Empty candidate arrays skip generation. When a user has fewer
        than ``num_items // 2`` candidates, ``item_rows`` slices
        ``emb_layer_1``; otherwise the full catalogue is generated once
        and indexed. Both paths are numerically the same.

        Parameters
        ----------
        client_id : int
            Client being scored. Unused.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Server-resident ``meta_network`` group for this round. The
            cache reloads only when this dict is a different object from
            the last load.
        client_state : ClientStateBase
            Unused; MetaMF has no client-resident parameters.
        extra_payload : dict
            Unused.

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
        del client_id, client_state, extra_payload
        self._ensure_theta(global_parameters)
        model = self._skeleton
        device = self.device
        scores: dict[int, np.ndarray] = {}
        half = self.num_items // 2
        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                items = np.asarray(candidate_items).reshape(-1)
                if items.size == 0:
                    scores[int(user_id)] = np.array([], dtype=np.float64)
                    continue
                user_t = torch.tensor([int(user_id)], dtype=torch.long, device=device)
                item_t = torch.as_tensor(items, dtype=torch.long, device=device)
                if items.size < half:
                    phi = model.item_rows(
                        user_t, item_t, item_payload=self.item_payload
                    )
                    item_emb = self._candidate_item_emb(phi)
                    preds = apply_rating_mlp(
                        item_emb, phi["w1"][0], phi["b1"][0], phi["w2"][0], phi["b2"][0]
                    )
                else:
                    phi = model.generate(user_t, item_payload=self.item_payload)
                    preds = predict_ratings(unbatch_phi(phi, 0), item_t)
                scores[int(user_id)] = preds.detach().cpu().numpy()
        return scores

    def _ensure_theta(self, global_parameters: dict[str, Any]) -> None:
        """Load Theta onto the shared skeleton when the dict identity changes."""
        self.init_model_skeleton()
        theta = global_parameters["meta_network"]
        if theta is self._loaded_theta:
            return
        self._skeleton.load_server_weights({"meta_network": theta})
        self._loaded_theta = theta

    def _candidate_item_emb(self, phi: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``(n_sel, d)`` item embeddings aligned with the sliced items."""
        if "item_embeddings" in phi:
            return phi["item_embeddings"][0]
        return phi["item_left"][0] @ phi["item_right"][0]

    def _samples_per_user(
        self,
        users_in_client: list[int],
        training_data: Any,
        generator: np.random.Generator | None,
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Return ``(user_id, item_ids, ratings)`` for users with a non-empty batch."""
        grouped = {}
        if training_data is not None and len(training_data) > 0:
            for user_id, group in training_data.groupby("user_id", sort=False):
                grouped[int(user_id)] = (
                    group["item_id"].to_numpy(dtype=np.int64),
                    group["rating"].to_numpy(dtype=np.float64),
                )
        out: list[tuple[int, np.ndarray, np.ndarray]] = []
        for user_id in users_in_client:
            uid = int(user_id)
            items, ratings = grouped.get(
                uid,
                (np.array([], dtype=np.int64), np.array([], dtype=np.float64)),
            )
            items, ratings = self._sample_user_batch(items, ratings, generator)
            if items.size == 0:
                continue
            out.append((uid, items, ratings))
        return out

    def _sample_user_batch(
        self,
        items: np.ndarray,
        ratings: np.ndarray,
        generator: np.random.Generator | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optionally subsample one user's ratings."""
        n = int(items.size)
        if n == 0 or self.local_batch_size is None:
            return items, ratings
        size = int(self.local_batch_size)
        if not self.local_batch_replacement and size >= n:
            return items, ratings
        rng = generator if generator is not None else np.random.default_rng()
        idx = rng.choice(n, size=size, replace=self.local_batch_replacement)
        return items[idx], ratings[idx]

    def _user_forward(
        self,
        phi_payload: dict[str, Any],
        items: np.ndarray,
        device: Any,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Build gradient leaves for one user and predict the sampled ratings."""
        rows, inverse = np.unique(items, return_inverse=True)
        item_key = "item_left" if self.item_payload == "factors" else "item_embeddings"
        expected = FACTOR_PHI_KEYS if self.item_payload == "factors" else FULL_PHI_KEYS
        missing = [key for key in expected if key not in phi_payload]
        if missing:
            raise KeyError(
                f"generated_parameters is missing {missing} for item_payload="
                f"{self.item_payload!r}"
            )

        def _leaf(key: str, index_rows: bool = False) -> torch.Tensor:
            value = torch.as_tensor(phi_payload[key], device=device)
            if index_rows:
                idx = torch.as_tensor(rows, dtype=torch.long, device=device)
                value = value.index_select(0, idx)
            return value.detach().clone().requires_grad_(True)

        leaves: dict[str, torch.Tensor] = {
            "item_rows": torch.as_tensor(rows, dtype=torch.long, device=device),
            item_key: _leaf(item_key, index_rows=True),
            "w1": _leaf("w1"),
            "b1": _leaf("b1"),
            "w2": _leaf("w2"),
            "b2": _leaf("b2"),
        }
        if self.item_payload == "factors":
            leaves["item_right"] = _leaf("item_right")
            local_left = leaves["item_left"].index_select(
                0, torch.as_tensor(inverse, dtype=torch.long, device=device)
            )
            item_emb = local_left @ leaves["item_right"]
        else:
            item_emb = leaves["item_embeddings"].index_select(
                0, torch.as_tensor(inverse, dtype=torch.long, device=device)
            )
        preds = apply_rating_mlp(
            item_emb, leaves["w1"], leaves["b1"], leaves["w2"], leaves["b2"]
        )
        return leaves, preds

    def _pack_dphi(self, leaves: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Detach per-user gradients; skip the integer ``item_rows`` tensor."""
        packed: dict[str, Any] = {
            "item_rows": leaves["item_rows"].detach().cpu().numpy().astype(np.int64)
        }
        for key, tensor in leaves.items():
            if key == "item_rows":
                continue
            if tensor.grad is None:
                raise ValueError(f"Missing gradient for generated parameter {key!r}")
            self._require_finite_tensor(tensor.grad, f"gradient of {key}")
            packed[key] = tensor.grad.detach().cpu().clone()
        return packed

    @staticmethod
    def _round_rng(
        seed: int | None, client_id: int, current_round: int
    ) -> np.random.Generator | None:
        """Seeded generator from ``(seed, client_id, round)``, or ``None``."""
        if seed is None:
            return None
        return np.random.default_rng([int(seed), int(client_id), int(current_round)])

    @staticmethod
    def _require_finite_tensor(tensor: torch.Tensor, where: str) -> None:
        """Raise if ``tensor`` contains NaN or Inf."""
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in {where}")
