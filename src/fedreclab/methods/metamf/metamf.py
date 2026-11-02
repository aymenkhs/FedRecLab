"""Meta Matrix Factorization (MetaMF) server.

Source: Lin, Yujie, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Dongxiao Yu,
Jun Ma, Maarten de Rijke, and Xiuzhen Cheng. "Meta Matrix Factorization
for Federated Rating Predictions." SIGIR 2020.
https://doi.org/10.1145/3397271.3401081

A server-resident meta network Theta generates per-user item embeddings
and a small two-layer rating MLP (Phi_u). Selected clients download Phi_u
for their users, compute the gradient of a local MSE with respect to Phi_u,
and upload that gradient. This class translates each upload into a
gradient on Theta (the chain rule of Algorithm 1, lines 12-14), hands
the per-client Theta-gradients to the configured aggregator, and applies
a torch Adam step with coupled weight decay.

Task: explicit rating prediction. Loss is paper eq. (8), the MSE averaged
over training ratings, plus the L2 term on Theta only (eq. 9-10).
Generated parameters are outputs of Theta and are not trained on the
device. Score is the predicted rating of the generated MLP.

Faithful aggregation is ``FedAvg`` on ``meta_network``: each client
uploads the gradient of its own per-rating mean, so a
``num_training_examples``-weighted mean reproduces eq. (8) on the round's
sample. ``Sum`` is the literal "accumulate" of Algorithm 1 and scales
the data gradient by the round size; that scale interacts with
``weight_decay`` and ``max_grad_norm``. Aggregators stay experiment
configuration; this class does not import or special-case them.

Defaults (paper section 4.4 and the authors' ``MetaMF.py`` /
``Run.ipynb``): ``user_emb_dim=item_emb_dim=32``, ``item_mem_num=8``,
``mem_size=128``, ``hidden_size=512``, rating hidden width
``item_emb_dim // 4``, Adam ``lr=1e-4``, ``weight_decay=1e-3`` (torch
Adam's coupled L2, equal to lambda in eq. 10), gradient-norm clip 5.
``item_payload="factors"`` sends the rise-dimensional factors rather
than the full item matrix.

Bypass Adam with
``param_io_overrides={"meta_network": {"apply_to": "server"}}`` and set
``server.server_learning_rate`` (or
``server.server_learning_rates["meta_network"]``) for a plain SGD
ablation. The default rate of 1.0 diverges and is refused. That path
loses weight decay, clipping, and the Adam moments.

Wire format against ParamIO: the ``meta_network`` package carries dPhi_u
per user, while the ParamIO describes dTheta. ``_collect_updates``
translates one into the other after validating ownership, structure,
row range, and finiteness, and after summing duplicate item rows with
``index_add_``. Overrides that would make the base pipeline treat the
upload as something else are rejected at construction: ``layout`` must
be ``"dense"``; ``exchange`` must be ``"gradient"`` or
``"encrypted_gradient"``; ``residency`` must be ``"server"``;
``apply_to`` must be ``"custom"`` or ``"server"``.

Training budget: the released ``Run.ipynb`` is centralized minibatch
training. It shuffles all training ratings and takes one Adam step per
batch of 64 ratings drawn across users, clipping at 5, for 100 epochs.
The paper's numbers come from about 10^6 Adam steps of 64 ratings each
(on ml-1m, about 12.5k steps per epoch). Algorithm 1 is the only
federated specification; under it one round is one Adam step, which is
what this class implements. Round counts are therefore not comparable
to the code's epochs. Learning rate, number of rounds, clients per
round, and ``local_batch_size`` are experiment configuration. A tuned
``lr`` is a configuration deviation of that run, not a method deviation.

Simulation shortcuts:
  ``compute_scores`` regenerates Phi_u from Theta instead of reading a
  stored copy on the device. This is numerically identical, because
  Algorithm 1 lines 16-19 regenerate every user's copy after every
  update. Storing them would take m * n * d_i floats (about 3 GB on
  ml-1m). The broadcast to all users after each update is not
  simulated for the same reason: Phi_u depends only on the current
  Theta.

Deviations from the paper and the reference code:
  The default ``item_payload="factors"`` sends I^l_u and I^r_u rather
  than I_u. It is numerically identical and only changes
  communication; use ``"full"`` for the paper's section 6.1 accounting.
  Clients that hold several users: each user has its own Phi_u, and
  one loss is averaged over the client's ratings. The paper assumes
  one user per device.
  Initialisation and gradient clipping follow the reference code, not
  the paper text (Xavier-normal Linear weights, zero Linear biases,
  Xavier-normal memory, N(0, 1) user embeddings; clip at 5).
  Shared item-generation hidden layer: one ``hidden_layer_3`` feeds
  both ``emb_layer_1`` (I^l) and ``emb_layer_2`` (I^r), as in the
  reference code. The paper's eq. (3) gives I^l and I^r separate
  hidden layers.

Defaults not fixed by the paper: ``local_batch_size=None`` (all local
ratings); clients per round and number of rounds.

Privacy:
  What stays on the device is the raw ratings only. The server holds
  every user's embedding e_u and the whole meta network, so it can
  generate any user's private model and predict that user's rating for
  any item. The paper affirms this server residency (Algorithm 1
  input; section 3.2) and acknowledges the leakage in section 7.
  Uploaded item-row ids reveal each user's rated items. With small
  batches, the b2 gradient reveals prediction residuals. MetaMF is not
  compatible with secure aggregation of the uploads, because the
  server needs each user's individual Phi-gradient to run the chain
  rule.

Theta is never mutated in place. Every update assigns a new
``global_parameters["meta_network"]`` dict so the client's skeleton
cache can detect a change by object identity.

Known limitations:
  Robust aggregators and sparse per-client gradients. Each user's row
  of U is non-zero in exactly one client's Theta-gradient. So are the
  ``emb_layer_1`` rows of items that only one client rated in the
  round, and rows rated by few clients are non-zero in only a few
  uploads. Robust aggregators therefore mostly discard these
  coordinates:
    Median: a coordinate that is non-zero in only one of K >= 3
    uploads aggregates to zero. With K = 2 the median is the mean of
    the two values, so that coordinate gets half.
    TrimmedMean: whether such a coordinate is dropped depends on the
    trim fraction. With nothing trimmed it is the mean.
    Krum / Multi-Krum: these keep or drop whole client gradients, not
    individual rows. Only the selected clients' user rows move.
    Bulyan: it selects clients and then takes a coordinate-wise
    trimmed mean, so it behaves like TrimmedMean on those rows.
  Where a robust aggregator zeroes a user's row, that embedding learns
  only through weight decay and Adam momentum, and the item generator
  learns little. ``FedAvg`` and ``SimpleAvg`` are the aggregators that
  match eq. (8). This class does not react to which aggregator is
  configured. FedMF's sparse uploads have the same property.
  Server memory grows with clients-per-round times |Theta|.
  ``emb_layer_1`` (hidden_size * item_mem_num * n) dominates: about
  60 MB per client on ml-1m and about 650 MB on Douban.
  Generating parameters for very large catalogues is expensive, which
  the paper also notes in section 6.1. Full-catalogue ranking still
  costs O(n * s * o) per user; slicing helps only for candidate
  subsets.
  The client's cached skeleton holds one copy of Theta plus a
  reference to the last loaded dict until the next reload.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn

from src.fedreclab.federated.base_client import ClientTrainingResponse
from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.metamf.metamf_model import (
    FACTOR_PHI_KEYS,
    FULL_PHI_KEYS,
    ITEM_PAYLOADS,
    MetaRecommender,
    resolve_rp_hidden_dim,
    unbatch_phi,
)

_ALLOWED_EXCHANGES = frozenset({"gradient", "encrypted_gradient"})
_ALLOWED_APPLY_TO = frozenset({"custom", "server"})
_FACTOR_UPLOAD_KEYS = frozenset(("item_rows",) + FACTOR_PHI_KEYS)
_FULL_UPLOAD_KEYS = frozenset(("item_rows",) + FULL_PHI_KEYS)


def default_param_ios() -> dict[str, ParamIO]:
    """Return the MetaMF ParamIO map.

    ``meta_network`` is server-resident, exchanged as a dense gradient,
    and applied by the server's custom Adam hook. The package a client
    puts on the wire is a gradient on generated Phi_u; ``_collect_updates``
    translates it into a gradient on Theta.

    Returns
    -------
    dict of str to ParamIO
        One group, ``meta_network``.

    Examples
    --------
    >>> pio = default_param_ios()["meta_network"]
    >>> pio.exchange, pio.apply_to, pio.layout
    ('gradient', 'custom', 'dense')
    """
    return {
        "meta_network": ParamIO(
            residency="server",
            exchange="gradient",
            apply_to="custom",
            layout="dense",
        ),
    }


class MetaMF(BaseFederatedServer):
    """Federated MetaMF. Server holds Theta; devices hold only ratings.

    Parameters
    ----------
    user_emb_dim : int, optional
        Width of user embeddings. Default is 32.
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
        Form of generated item factors sent to the device. Default is
        ``"factors"``.
    lr : float, optional
        Adam step size. Default is ``1e-4``.
    weight_decay : float, optional
        Coupled L2 coefficient. Default is ``1e-3``.
    max_grad_norm : float or None, optional
        Clip threshold. Default is 5. ``None`` disables clipping.
    betas : tuple of float, optional
        Adam ``(beta1, beta2)``. Default is ``(0.9, 0.999)``.
    eps : float, optional
        Adam ``eps``. Default is ``1e-8``.

    Raises
    ------
    ValueError
        If a hyper-parameter is out of range, an override would break
        the Phi-to-Theta translation, or the client architecture
        disagrees with this server.

    Examples
    --------
    >>> server = MetaMF(interaction_dataset=data, ...)  # doctest: +SKIP
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
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        max_grad_norm: float | None = 5.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs,
    ):
        if item_payload not in ITEM_PAYLOADS:
            raise ValueError(
                f"item_payload must be one of {ITEM_PAYLOADS}, got {item_payload!r}"
            )
        lr = float(lr)
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        weight_decay = float(weight_decay)
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)
            if max_grad_norm <= 0.0:
                raise ValueError(
                    f"max_grad_norm must be > 0 or None, got {max_grad_norm}"
                )
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas!r}")
        eps = float(eps)
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.user_emb_dim = int(user_emb_dim)
        self.item_emb_dim = int(item_emb_dim)
        self.item_mem_num = int(item_mem_num)
        self.mem_size = int(mem_size)
        self.hidden_size = int(hidden_size)
        self.rp_hidden_dim = resolve_rp_hidden_dim(self.item_emb_dim, rp_hidden_dim)
        self.item_payload = item_payload
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.betas = (beta1, beta2)
        self.eps = eps
        self._meta_model: MetaRecommender | None = None
        self._meta_optimizer: torch.optim.Adam | None = None
        self._validate_param_io_overrides(kwargs.get("param_io_overrides"))
        super().__init__(param_ios=default_param_ios(), **kwargs)

    def _validate_param_io_overrides(self, overrides: dict[str, dict] | None) -> None:
        """Reject overrides that would break the Phi-to-Theta translation.

        Runs before ``super().__init__`` so a rejected configuration
        does not bind the client, selector, or aggregators.

        Parameters
        ----------
        overrides : dict or None
            ``param_io_overrides`` as passed to the constructor.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the ``meta_network`` override is not a dict.
        ValueError
            If ``layout``, ``exchange``, ``residency``, or ``apply_to``
            would make the base pipeline treat dPhi_u as dTheta.
        """
        fields = {
            "residency": "server",
            "exchange": "gradient",
            "apply_to": "custom",
            "layout": "dense",
        }
        if overrides and "meta_network" in overrides:
            extra = overrides["meta_network"]
            if not isinstance(extra, dict):
                raise TypeError(
                    "param_io_overrides['meta_network'] must be a dict of "
                    f"ParamIO fields, got {type(extra).__name__}"
                )
            fields.update(extra)

        layout = fields.get("layout", "dense")
        if layout != "dense":
            raise ValueError(
                f"ParamIO.layout for 'meta_network' must be 'dense' "
                f"(got {layout!r}); layout='rows' would pack per-user "
                "Phi-gradients as item-table rows, which breaks the "
                "Phi-to-Theta translation in _collect_updates"
            )
        exchange = fields.get("exchange")
        if exchange not in _ALLOWED_EXCHANGES:
            raise ValueError(
                f"ParamIO.exchange for 'meta_network' must be one of "
                f"{sorted(_ALLOWED_EXCHANGES)} (got {exchange!r}); the "
                "translated value is a Theta-gradient"
            )
        residency = fields.get("residency")
        if residency != "server":
            raise ValueError(
                f"ParamIO.residency for 'meta_network' must be 'server' "
                f"(got {residency!r}); client residency would read Theta "
                "from client states, which hold nothing"
            )
        apply_to = fields.get("apply_to")
        if apply_to not in _ALLOWED_APPLY_TO:
            raise ValueError(
                f"ParamIO.apply_to for 'meta_network' must be one of "
                f"{sorted(_ALLOWED_APPLY_TO)} (got {apply_to!r}); "
                "'clients' or 'both' would write Theta into client states"
            )

    def _initialize_server_parameters(self) -> dict[str, dict[str, torch.Tensor]]:
        """Create the meta network and a persistent Adam optimiser.

        Returns
        -------
        dict
            ``meta_network`` group of CPU clones.

        Raises
        ------
        ValueError
            If the client architecture disagrees with this server.
        """
        self._assert_client_architecture()
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
        model = MetaRecommender(
            num_users=self.interaction_dataset.num_users,
            num_items=self.interaction_dataset.num_items,
            user_emb_dim=self.user_emb_dim,
            item_emb_dim=self.item_emb_dim,
            item_mem_num=self.item_mem_num,
            mem_size=self.mem_size,
            hidden_size=self.hidden_size,
            rp_hidden_dim=self.rp_hidden_dim,
            generator=generator,
        )
        if self.device is not None:
            model = model.to(self.device)
        self._meta_model = model
        self._meta_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        return model.get_server_state_groups()

    def _build_server_payload(self, client_id: int | None = None) -> dict[str, Any]:
        """Generate Phi_u for this client's users; never send Theta.

        Parameters
        ----------
        client_id : int or None
            Client to generate for. ``None`` is rejected because the
            payload is per-client.

        Returns
        -------
        dict
            ``{"generated_parameters": {user_id: Phi_u}}``.

        Raises
        ------
        ValueError
            If ``client_id`` is None.
        """
        if client_id is None:
            raise ValueError("MetaMF payloads are per-client; client_id is required")
        users = list(
            self.interaction_dataset.client_partition.client_ids_2_user_ids[client_id]
        )
        if not users:
            return {"generated_parameters": {}}
        self._meta_model.load_server_weights(self.global_parameters)
        user_t = torch.as_tensor(users, dtype=torch.long, device=self.device)
        with torch.no_grad():
            batched = self._meta_model.generate(
                user_t, item_payload=self.item_payload
            )
        generated = {}
        for index, uid in enumerate(users):
            generated[int(uid)] = {
                key: tensor[index].detach().cpu().clone()
                for key, tensor in batched.items()
            }
        return {"generated_parameters": generated}

    def _collect_updates(
        self,
        param_name: str,
        param_io: ParamIO,
        training_responses: List[ClientTrainingResponse],
    ) -> Tuple[List[Any], List[int]]:
        """Validate Phi-gradients and translate them into Theta-gradients.

        Parameters
        ----------
        param_name : str
            ParamIO group to collect.
        param_io : ParamIO
            Declaration for ``param_name``.
        training_responses : list of ClientTrainingResponse
            Responses from the clients trained this round.

        Returns
        -------
        updates : list
            One dense Theta-gradient dict per participating client.
        participant_ids : list of int
            Client ids aligned with ``updates``.

        Raises
        ------
        ValueError
            If a user id is not owned by the uploading client, the
            per-user key set or shapes are wrong, or a value is
            non-finite.
        IndexError
            If an uploaded item row is outside ``[0, num_items)``.
        """
        raw_updates, participant_ids = super()._collect_updates(
            param_name, param_io, training_responses
        )
        if param_name != "meta_network":
            return raw_updates, participant_ids
        self._meta_model.load_server_weights(self.global_parameters)
        translated = [
            self._phi_grads_to_theta_grad(client_id, upload)
            for upload, client_id in zip(raw_updates, participant_ids)
        ]
        return translated, participant_ids

    def _phi_grads_to_theta_grad(
        self, client_id: int, upload: dict[Any, Any]
    ) -> dict[str, torch.Tensor]:
        """Chain-rule one client's dPhi uploads into a dense dTheta."""
        owned = {
            int(uid)
            for uid in self.interaction_dataset.client_partition.client_ids_2_user_ids[
                client_id
            ]
        }
        acc: dict[str, torch.Tensor] | None = None
        if not isinstance(upload, dict):
            raise ValueError(
                f"Client {client_id} meta_network upload must be a dict of "
                f"per-user Phi-gradients, got {type(upload).__name__}"
            )
        for raw_uid, dphi in upload.items():
            user_id = int(raw_uid)
            if user_id not in owned:
                raise ValueError(
                    f"Client {client_id} uploaded user_id={user_id} which "
                    f"is not in client_ids_2_user_ids[{client_id}]"
                )
            coalesced = self._validate_and_coalesce_dphi(client_id, user_id, dphi)
            theta_grad = self._vjp_one_user(user_id, coalesced)
            if acc is None:
                acc = theta_grad
            else:
                for name, value in theta_grad.items():
                    acc[name] = acc[name] + value
        if acc is None:
            return {
                name: torch.zeros_like(param, device="cpu")
                for name, param in self._meta_model.named_parameters()
            }
        return acc

    def _validate_and_coalesce_dphi(
        self, client_id: int, user_id: int, dphi: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Check keys, shapes, range, and finiteness; sum duplicate item rows."""
        if not isinstance(dphi, dict):
            raise ValueError(
                f"Client {client_id} user {user_id} Phi-gradient must be a "
                f"dict, got {type(dphi).__name__}"
            )
        expected = (
            _FACTOR_UPLOAD_KEYS if self.item_payload == "factors" else _FULL_UPLOAD_KEYS
        )
        got = set(dphi)
        if got != expected:
            raise ValueError(
                f"Client {client_id} user {user_id} Phi-gradient keys "
                f"{sorted(got)} do not match item_payload={self.item_payload!r} "
                f"keys {sorted(expected)}"
            )
        rows = np.asarray(dphi["item_rows"], dtype=np.int64).reshape(-1)
        n_items = self.interaction_dataset.num_items
        if rows.size and (np.any(rows < 0) or np.any(rows >= n_items)):
            raise IndexError(
                f"Client {client_id} user {user_id} has item_rows outside "
                f"[0, {n_items})"
            )
        item_key = "item_left" if self.item_payload == "factors" else "item_embeddings"
        item_width = (
            self.item_mem_num if self.item_payload == "factors" else self.item_emb_dim
        )
        item_values = torch.as_tensor(dphi[item_key], dtype=torch.float32)
        if item_values.ndim == 1:
            item_values = item_values.reshape(-1, item_width)
        if item_values.shape[0] != rows.size:
            raise ValueError(
                f"Client {client_id} user {user_id} {item_key} has "
                f"{item_values.shape[0]} rows, expected {rows.size}"
            )
        if item_values.shape[-1] != item_width:
            raise ValueError(
                f"Client {client_id} user {user_id} {item_key} last dim "
                f"{item_values.shape[-1]} != {item_width}"
            )
        self._require_finite(item_values, f"client {client_id} user {user_id} {item_key}")

        expected_shapes = {
            "w1": (self.item_emb_dim, self.rp_hidden_dim),
            "b1": (1, self.rp_hidden_dim),
            "w2": (self.rp_hidden_dim, 1),
            "b2": (1, 1),
        }
        if self.item_payload == "factors":
            expected_shapes["item_right"] = (self.item_mem_num, self.item_emb_dim)
        dense = {}
        for key, shape in expected_shapes.items():
            tensor = torch.as_tensor(dphi[key], dtype=torch.float32)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"Client {client_id} user {user_id} {key} has shape "
                    f"{tuple(tensor.shape)}, expected {shape}"
                )
            self._require_finite(tensor, f"client {client_id} user {user_id} {key}")
            dense[key] = tensor

        row_t = torch.as_tensor(rows, dtype=torch.long)
        if row_t.numel() == 0:
            coalesced_rows = row_t
            coalesced_values = item_values.new_zeros((0, item_width))
        else:
            unique, inverse = torch.unique(row_t, return_inverse=True, sorted=True)
            coalesced_values = item_values.new_zeros((unique.numel(), item_width))
            coalesced_values.index_add_(0, inverse, item_values)
            coalesced_rows = unique
        dense["item_rows"] = coalesced_rows
        dense[item_key] = coalesced_values
        return dense

    def _vjp_one_user(
        self, user_id: int, dphi: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Vector-Jacobian product of generate/item_rows at this user."""
        device = self.device
        user_t = torch.tensor([int(user_id)], dtype=torch.long, device=device)
        rows = dphi["item_rows"].to(device=device)
        phi = self._meta_model.item_rows(
            user_t, rows, item_payload=self.item_payload
        )
        one = unbatch_phi(phi, 0)
        outputs = []
        grad_outputs = []
        item_key = "item_left" if self.item_payload == "factors" else "item_embeddings"
        outputs.append(one[item_key])
        grad_outputs.append(dphi[item_key].to(device=device, dtype=one[item_key].dtype))
        for key in ("item_right", "w1", "b1", "w2", "b2"):
            if key not in one:
                continue
            outputs.append(one[key])
            grad_outputs.append(dphi[key].to(device=device, dtype=one[key].dtype))
        params = list(self._meta_model.parameters())
        grads = torch.autograd.grad(
            outputs,
            params,
            grad_outputs=grad_outputs,
            retain_graph=False,
            allow_unused=True,
        )
        named = {}
        for (name, param), grad in zip(self._meta_model.named_parameters(), grads):
            if grad is None:
                value = torch.zeros_like(param, device="cpu")
            else:
                self._require_finite(grad, f"Theta-gradient of {name}")
                value = grad.detach().cpu().clone()
            named[name] = value
        return named

    def _apply_aggregated_result_custom(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: list[int],
    ) -> None:
        """Clip, then Adam-update the persistent meta network.

        Loads ``global_parameters`` into the module in place so Adam
        moments stay attached to the same Parameter objects. Writes a
        new CPU-clone dict back so the client skeleton cache sees a
        new identity.

        Parameters
        ----------
        param_name : str
            Must be ``meta_network``.
        aggregated_result : dict
            Aggregated Theta-gradient, keyed like
            ``global_parameters["meta_network"]``.
        param_io : ParamIO
            Unused; present for the hook signature.
        selected_client_ids : list of int
            Unused.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``param_name`` is not ``meta_network``, the keys do not
            match, or a value is non-finite.
        """
        del param_io, selected_client_ids
        if param_name != "meta_network":
            raise ValueError(
                f"MetaMF custom apply is defined only for 'meta_network', "
                f"got {param_name!r}"
            )
        old = self.global_parameters[param_name]
        if set(old) != set(aggregated_result):
            raise ValueError(
                f"Aggregated keys {sorted(aggregated_result)} do not match "
                f"parameter keys {sorted(old)}"
            )
        self._meta_model.load_server_weights(self.global_parameters)
        for name, param in self._meta_model.named_parameters():
            grad = torch.as_tensor(
                aggregated_result[name], dtype=param.dtype, device=param.device
            )
            self._require_finite(grad, f"aggregated gradient of {name}")
            param.grad = grad
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self._meta_model.parameters(), self.max_grad_norm
            )
        self._meta_optimizer.step()
        self._meta_optimizer.zero_grad(set_to_none=True)
        for param in self._meta_model.parameters():
            self._require_finite(param, "updated meta-network parameter")
        self.global_parameters[param_name] = {
            name: param.detach().cpu().clone()
            for name, param in self._meta_model.named_parameters()
        }

    def _on_round_start_server(self) -> None:
        """Refuse the SGD ablation when no learning rate is set."""
        pio = self.param_ios["meta_network"]
        if pio.apply_to != "server":
            return
        has_named = (
            getattr(self, "server_learning_rates", None) is not None
            and "meta_network" in self.server_learning_rates
        )
        has_global = getattr(self, "server_learning_rate", None) is not None
        if not has_named and not has_global:
            raise ValueError(
                "param_io_overrides apply_to='server' requires "
                "server.server_learning_rate or "
                "server.server_learning_rates['meta_network']; the "
                "default of 1.0 diverges"
            )

    def _assert_client_architecture(self) -> None:
        """Raise if the client architecture disagrees with this server."""
        client = self.client_algorithm
        checks = (
            ("user_emb_dim", self.user_emb_dim, int),
            ("item_emb_dim", self.item_emb_dim, int),
            ("item_mem_num", self.item_mem_num, int),
            ("mem_size", self.mem_size, int),
            ("hidden_size", self.hidden_size, int),
            ("rp_hidden_dim", self.rp_hidden_dim, int),
        )
        for name, expected, caster in checks:
            got = getattr(client, name, None)
            if got is not None and caster(got) != expected:
                raise ValueError(
                    f"MetaMF {name}={expected} does not match client {name}={got}"
                )
        got_payload = getattr(client, "item_payload", None)
        if got_payload is not None and got_payload != self.item_payload:
            raise ValueError(
                f"MetaMF item_payload={self.item_payload!r} does not match "
                f"client item_payload={got_payload!r}"
            )

    @staticmethod
    def _require_finite(tensor: torch.Tensor, where: str) -> None:
        """Raise if ``tensor`` contains NaN or Inf."""
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values in {where}")
