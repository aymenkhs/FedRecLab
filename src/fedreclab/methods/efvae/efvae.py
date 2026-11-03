"""Efficient Federated Variational Autoencoder (EFVAE) server.

Source: Zhang et al., "EFVAE: Efficient Federated Variational Autoencoder
for Collaborative Filtering", CIKM 2024.

This concrete server is the MultVAE + FCIS instantiation. Shared FCIS
machinery (ParamIO, FIMS rebuild, custom Adam including row groups,
full-download simulation) lives on ``EFVAEFCISServer``. MultDAE and
RecVAE are intended later subclasses of that base.

Selected clients download the full MultVAE weights and the round's FIMS
sampler (simulation shortcut matching the authors' reference), compute
one importance-sampled MultVAE gradient with ``rho=1`` fake items, and
upload sparse encoder/decoder row gradients plus a dense middle
gradient. Aggregators stay experiment configuration (faithful default:
``SimpleAvg`` on every group).

Defaults (paper §4.1.4 / reference ``ef-multi_vae/main.py``): architecture
``|I| -> 120 -> 40``, dropout 0.5, ``anneal_cap=0.2``,
``total_anneal_steps=200000``, Adam ``lr=1e-3``, FIMS-Uni
``num_cluster=16``, ``num_sample=800``.

Simulation shortcuts
--------------------
Full parameter tables are distributed every round (the reference code
also ``load_state_dict``s the full model and only accounts sparse
communication). Secret sharing / MPC is omitted. Communication-cost
meters are not implemented.

Deviations
----------
Paper Alg. 1–2 multi-step sparse download is not enforced (same as the
zip). Item padding never uses id ``0`` as a sentinel (AGENTS rule;
clients train one user at a time). If a user has more positives than
remaining non-train items, fake-item count is clamped to the remainder
(paper assumes ``rho=1`` is feasible). Splits and evaluators are not
hard-coded; the method only implements ``compute_scores``.
"""

from __future__ import annotations

import torch

from fedreclab.methods.efvae.efvae_fcis import EFVAEFCISServer
from fedreclab.methods.efvae.efvae_model import EFMultVAE


class EFVAE(EFVAEFCISServer):
    """Federated MultVAE with FCIS. Server holds the network and Adam state.

    Parameters
    ----------
    hidden_dims : sequence of int, optional
        Encoder hidden widths. Default is ``(120,)``.
    latent_dim : int, optional
        Size of ``mu`` / ``logvar``. Default is 40.
    dropout : float, optional
        Input / embedding dropout. Default is 0.5.
    anneal_cap : float, optional
        Maximum KL weight. Default is 0.2.
    total_anneal_steps : int, optional
        Linear anneal length in Adam steps (one per federated round).
        Default is 200000. ``0`` uses ``anneal_cap`` immediately.
    lr, betas, eps, num_cluster, num_sample
        Forwarded to ``EFVAEFCISServer``.

    Raises
    ------
    ValueError
        If hyper-parameters are out of range, or the client architecture
        disagrees with this server.

    Examples
    --------
    >>> server = EFVAE(hidden_dims=(4,), latent_dim=2, interaction_dataset=data, ...)  # doctest: +SKIP
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (120,),
        latent_dim: int = 40,
        dropout: float = 0.5,
        anneal_cap: float = 0.2,
        total_anneal_steps: int = 200000,
        **kwargs,
    ):
        anneal_cap = float(anneal_cap)
        if anneal_cap < 0.0:
            raise ValueError(f"anneal_cap must be >= 0, got {anneal_cap}")
        if (
            not isinstance(total_anneal_steps, int)
            or isinstance(total_anneal_steps, bool)
            or total_anneal_steps < 0
        ):
            raise ValueError(
                f"total_anneal_steps must be a non-negative int, "
                f"got {total_anneal_steps!r}"
            )
        dropout = float(dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.dropout = dropout
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = int(total_anneal_steps)
        self.update_count = 0
        super().__init__(**kwargs)

    def _initialize_server_parameters(self) -> dict[str, dict[str, torch.Tensor]]:
        """Create EF-MultVAE weights and zero Adam moments.

        Returns
        -------
        dict
            FCIS ParamIO groups of CPU clones.

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
        model = EFMultVAE(
            num_items=self.interaction_dataset.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            generator=generator,
        )
        groups = model.get_server_state_groups()
        self._zero_adam_moments(groups)
        self.update_count = 0
        return groups

    def _on_round_start_server(self) -> None:
        """Rebuild FIMS and publish anneal for MultVAE clients.

        Returns
        -------
        None
        """
        super()._on_round_start_server()
        # Match reference: anneal = min(cap, update_count / total_steps).
        if self.total_anneal_steps > 0:
            anneal = min(
                self.anneal_cap,
                1.0 * float(self.update_count) / float(self.total_anneal_steps),
            )
        else:
            anneal = self.anneal_cap
        if self.extra_payload is None:
            self.extra_payload = {}
        self.extra_payload["anneal"] = float(anneal)

    def _on_round_end_server(self) -> None:
        """Advance the MultVAE anneal counter (one step per round).

        Returns
        -------
        None
        """
        self.update_count += 1

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
                f"EFVAE hidden_dims={self.hidden_dims} does not match "
                f"client hidden_dims={tuple(client_hidden)}"
            )
        client_dim = getattr(client, "latent_dim", None)
        if client_dim is not None and int(client_dim) != self.latent_dim:
            raise ValueError(
                f"EFVAE latent_dim={self.latent_dim} does not match "
                f"client latent_dim={client_dim}"
            )
        client_dropout = getattr(client, "dropout", None)
        if client_dropout is not None and float(client_dropout) != float(self.dropout):
            raise ValueError(
                f"EFVAE dropout={self.dropout} does not match "
                f"client dropout={client_dropout}"
            )
