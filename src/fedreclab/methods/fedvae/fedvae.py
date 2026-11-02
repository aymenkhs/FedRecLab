"""Federated Variational Autoencoder (FedVAE) server.

Source: Polato, "Federated Variational Autoencoder for Collaborative
Filtering", IJCNN 2021.

The whole Mult-VAE is server-resident. Selected clients download the
weights, compute one gradient of the Mult-VAE ELBO on their training
histories, and upload that dense gradient. The configured aggregator
combines the uploads (``SimpleAvg`` is the paper's round mean; Multi-Krum
is the paper's optional SecureFilter). This class then applies a
Kingma-Ba Adam step, with the optional FedVAEdec schedule, in
``_apply_aggregated_result_custom``. Aggregators stay experiment
configuration; this class does not import or special-case them.

The paper reports held-out-user (strong generalisation) evaluation. The
method is protocol-agnostic: it trains on whatever the dataset calls each
client's training interactions and scores the candidate ids the evaluator
passes in. Splits and evaluators are not implemented here. Every training
row is a positive; paper binarisation (ml-1m ratings of 4 or more) is
``rating_exclusion_threshold=4`` at load time.

Weights use Xavier-uniform and biases ``N(0, 1)``, as in the authors'
reference code (rectorch ``MultiVAE_net``); Algorithm 1's "N(0, 1)" is
read as the bias initialisation.

Defaults (paper section V-A): architecture ``|I| -> 600 -> 200``,
dropout 0.5, ``beta=0.2`` (constant; set on the client), Adam
``lr=1e-3``, ``beta1=0.9``, ``beta2=0.999``, ``eps=1e-8``. Plain FedVAE
is ``lr_multiplier=0``. FedVAEdec is opt-in (``lr_multiplier`` in
``{1, 3, 5}``, ``lr_decay=0.9``). There is exactly one local gradient;
there is no ``local_steps`` flag.

Faithful aggregation is ``SimpleAvg`` on ``vae``. The notebook divides
the summed gradient by the total number of training users; under Adam
that only rescales ``eps``. ``FedAvg`` weights clients by history length
and is a documented deviation from the paper's uniform user mean.

Parameter grouping: the whole VAE is one exchanged group, ``vae``, so
one aggregator applies to all of it. To use different aggregators for
the encoder and the decoder, subclass and split the group: override
``default_param_ios()`` and ``MultVAE.get_server_state_groups()`` so
that the parameter names are partitioned into, for example, ``encoder``
and ``decoder``. ``param_io_overrides`` cannot split a group; it only
changes the fields of existing groups.

Cost of splitting: averaging and coordinate-wise aggregators
(SimpleAvg, FedAvg, Sum, Median, TrimmedMean) give the same result
either way. Krum, Multi-Krum and Bulyan do not. They run once per group
and keep or drop clients separately in each, so one round can apply
client A's encoder gradient together with client C's decoder gradient.
That update is not any single client's gradient, and a client rejected as
an outlier in one group can still be accepted in the other. The paper's
SecureFilter keeps or drops whole client gradients, which only the
single group reproduces.

Known limitations: every selected client uploads a dense gradient of the
whole VAE, about 18 MB for 3.5k items with hidden width 600. The round
loop holds all uploads before aggregating, the server copies them to
``device``, and FedAvg/SimpleAvg build further float64 temporaries per
tensor. Peak memory therefore grows as ``clients_per_round * |I|``.
ml-1m with 250 clients per round needs on the order of 10 GB; catalogues
of tens of thousands of items are infeasible at that round size. Reduce
clients per round or run on CPU for large catalogues. A future fix
(accumulating aggregators, then streaming aggregation) is on the
roadmap.

Privacy: gradients travel in plaintext. The encoder's first-layer
gradient is non-zero exactly on the post-dropout positive items, so the
server sees about half of each user's positives in the clear. The
paper's "dropout as differential privacy" argument does not cover this.
Krum is not the default.

Deviations from the paper and the notebook:

- The decay exponent starts at 0, so ``lr_multiplier`` is the
  first-period multiplier (the notebook). The paper's indexing (first
  round ``t=1``) is ``lr_multiplier = gamma * delta``.
- Default ``decay_every_rounds`` is one epoch,
  ``ceil(num_clients / R)``, with ``R`` taken from the first round's
  selection and then frozen. Variable-size selectors therefore use the
  first round's count. Pass ``decay_every_rounds=1`` for Algorithm 1's
  per-round decay.
- ``extrapolate="all"`` (paper eq. 4) scales every tensor;
  ``extrapolate="weights"`` matches the notebook (biases keep the plain
  Adam step).
- Initialisation follows rectorch, not a literal ``N(0, 1)`` on weights.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedvae.fedvae_model import MultVAE

_EXTRAPOLATE = ("all", "weights")


def default_param_ios() -> dict[str, ParamIO]:
    """Return the FedVAE ParamIO map.

    ``vae`` is server-resident, exchanged as a dense gradient, and
    applied by the server's custom Adam hook.

    Returns
    -------
    dict of str to ParamIO
        One group, ``vae``.

    Examples
    --------
    >>> pio = default_param_ios()["vae"]
    >>> pio.exchange, pio.apply_to, pio.layout
    ('gradient', 'custom', 'dense')
    """
    return {
        "vae": ParamIO(
            residency="server",
            exchange="gradient",
            apply_to="custom",
            layout="dense",
        ),
    }


class FedVAE(BaseFederatedServer):
    """Federated Mult-VAE. Server holds the whole network and Adam state.

    Bypass Adam with
    ``param_io_overrides={"vae": {"apply_to": "server"}}`` and set
    ``server.server_learning_rate`` for a plain SGD ablation.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Encoder hidden widths. Default is ``(600,)``.
    latent_dim : int, optional
        Size of ``mu`` and ``logvar``. Default is 200.
    dropout : float, optional
        Input dropout rate stored on the skeleton. Default is 0.5.
    lr : float, optional
        Adam step size ``eta``. Default is ``1e-3``.
    betas : tuple of float, optional
        Adam ``(beta1, beta2)``. Default is ``(0.9, 0.999)``.
    eps : float, optional
        Adam ``eps``. Default is ``1e-8``.
    lr_multiplier : float, optional
        FedVAEdec ``gamma``. Default is 0 (plain FedVAE).
    lr_decay : float, optional
        FedVAEdec ``delta`` in ``[0, 1]``. Default is 0.9.
    decay_every_rounds : int or None, optional
        Period of the decay exponent. ``None`` (default) is one epoch,
        resolved after the first selection as
        ``ceil(num_clients / R)``. ``1`` counts rounds.
    extrapolate : {"all", "weights"}, optional
        Which tensors receive ``eta_t``. Default is ``"all"``.

    Raises
    ------
    ValueError
        If a hyper-parameter is out of range, or the client architecture
        disagrees with this server.

    Examples
    --------
    >>> server = FedVAE(hidden_dims=(4,), latent_dim=2, interaction_dataset=data, ...)  # doctest: +SKIP
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (600,),
        latent_dim: int = 200,
        dropout: float = 0.5,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lr_multiplier: float = 0.0,
        lr_decay: float = 0.9,
        decay_every_rounds: int | None = None,
        extrapolate: str = "all",
        **kwargs,
    ):
        if extrapolate not in _EXTRAPOLATE:
            raise ValueError(
                f"extrapolate must be one of {_EXTRAPOLATE}, got {extrapolate!r}"
            )
        lr_decay = float(lr_decay)
        if not 0.0 <= lr_decay <= 1.0:
            raise ValueError(f"lr_decay must be in [0, 1], got {lr_decay}")
        lr_multiplier = float(lr_multiplier)
        if lr_multiplier < 0.0:
            raise ValueError(f"lr_multiplier must be >= 0, got {lr_multiplier}")
        if decay_every_rounds is not None:
            if (
                not isinstance(decay_every_rounds, int)
                or isinstance(decay_every_rounds, bool)
                or decay_every_rounds < 1
            ):
                raise ValueError(
                    f"decay_every_rounds must be a positive int or None, "
                    f"got {decay_every_rounds!r}"
                )
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas!r}")
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.betas = (beta1, beta2)
        self.eps = float(eps)
        self.lr_multiplier = lr_multiplier
        self.lr_decay = lr_decay
        self.decay_every_rounds = decay_every_rounds
        self.extrapolate = extrapolate
        self.adam_m: dict[str, torch.Tensor] = {}
        self.adam_v: dict[str, torch.Tensor] = {}
        self.adam_step = 0
        super().__init__(param_ios=default_param_ios(), **kwargs)

    def _initialize_server_parameters(self) -> dict[str, dict[str, torch.Tensor]]:
        """Create the VAE weights and zero the Adam moments.

        Returns
        -------
        dict
            ``vae`` group of CPU clones.

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
        model = MultVAE(
            num_items=self.interaction_dataset.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            generator=generator,
        )
        groups = model.get_server_state_groups()
        self.adam_m = {
            name: torch.zeros_like(tensor) for name, tensor in groups["vae"].items()
        }
        self.adam_v = {
            name: torch.zeros_like(tensor) for name, tensor in groups["vae"].items()
        }
        self.adam_step = 0
        return groups

    def _post_selection_callback(self) -> None:
        """Resolve the default epoch length from the first round's ``R``.

        ``decay_every_rounds is None`` becomes
        ``ceil(num_clients / max(R, 1))`` and is then frozen. An explicit
        constructor value is left unchanged.

        Returns
        -------
        None
        """
        if self.decay_every_rounds is not None:
            return
        if self.current_round != 0:
            return
        n_selected = len(self.selected_client_ids or [])
        r = max(n_selected, 1)
        self.decay_every_rounds = int(math.ceil(self.num_clients / r))

    def _apply_aggregated_result_custom(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: list[int],
    ) -> None:
        """Adam-update ``global_parameters[param_name]`` from the aggregated gradient.

        Follows Algorithm 2 (bias-corrected Adam) with step size
        ``eta_t = lr * (1 + lr_multiplier * lr_decay ** e)``,
        ``e = current_round // decay_every_rounds``. Under
        ``extrapolate="weights"`` only tensors whose names end in
        ``.weight`` use ``eta_t``; the rest use ``lr``.

        Parameters
        ----------
        param_name : str
            Must be ``vae``.
        aggregated_result : dict
            Aggregated gradient, keyed like ``global_parameters["vae"]``.
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
            If ``param_name`` is not ``vae``, the keys do not match, a
            value is non-finite, or ``decay_every_rounds`` is still
            ``None``.
        """
        if param_name != "vae":
            raise ValueError(
                f"FedVAE custom apply is defined only for 'vae', got {param_name!r}"
            )
        old = self.global_parameters[param_name]
        if set(old) != set(aggregated_result):
            raise ValueError(
                f"Aggregated keys {sorted(aggregated_result)} do not match "
                f"parameter keys {sorted(old)}"
            )
        if self.decay_every_rounds is None:
            raise ValueError(
                "decay_every_rounds is still None; the first selection "
                "must run before an Adam step."
            )
        period = int(self.decay_every_rounds)
        epoch_idx = int(self.current_round) // period
        gamma_t = self.lr_multiplier * (self.lr_decay ** epoch_idx)
        eta_t = self.lr * (1.0 + gamma_t)
        beta1, beta2 = self.betas
        self.adam_step += 1
        t = self.adam_step
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        updated: dict[str, torch.Tensor] = {}
        for name, old_value in old.items():
            like = old_value if torch.is_tensor(old_value) else torch.as_tensor(old_value)
            g = torch.as_tensor(
                aggregated_result[name], dtype=like.dtype, device=like.device
            )
            if not torch.isfinite(g).all():
                raise ValueError(f"Non-finite aggregated gradient for {name}")
            m = self.adam_m[name].to(device=like.device, dtype=like.dtype)
            v = self.adam_v[name].to(device=like.device, dtype=like.dtype)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / bc1
            v_hat = v / bc2
            eta = eta_t if self._uses_schedule(name) else self.lr
            new_value = like - eta * m_hat / (torch.sqrt(v_hat) + self.eps)
            if not torch.isfinite(new_value).all():
                raise ValueError(f"Non-finite Adam update for {name}")
            self.adam_m[name] = m
            self.adam_v[name] = v
            updated[name] = new_value
        self.global_parameters[param_name] = updated

    def _uses_schedule(self, tensor_name: str) -> bool:
        """Return whether ``tensor_name`` receives the FedVAEdec step size.

        Parameters
        ----------
        tensor_name : str
            Fully qualified parameter name.

        Returns
        -------
        bool
            Always True for ``extrapolate="all"``. For ``"weights"``,
            True iff the name ends with ``.weight``.
        """
        if self.extrapolate == "all":
            return True
        return tensor_name.endswith(".weight")

    def _assert_client_architecture(self) -> None:
        """Raise if the client widths disagree with this server.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``hidden_dims``, ``latent_dim``, or ``dropout`` differ.
        """
        client = self.client_algorithm
        client_hidden = getattr(client, "hidden_dims", None)
        if client_hidden is not None and tuple(client_hidden) != self.hidden_dims:
            raise ValueError(
                f"FedVAE hidden_dims={self.hidden_dims} does not match "
                f"client hidden_dims={tuple(client_hidden)}"
            )
        client_dim = getattr(client, "latent_dim", None)
        if client_dim is not None and int(client_dim) != self.latent_dim:
            raise ValueError(
                f"FedVAE latent_dim={self.latent_dim} does not match "
                f"client latent_dim={client_dim}"
            )
        client_dropout = getattr(client, "dropout", None)
        if client_dropout is not None and float(client_dropout) != float(self.dropout):
            raise ValueError(
                f"FedVAE dropout={self.dropout} does not match "
                f"client dropout={client_dropout}"
            )
