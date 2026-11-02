"""Federated Matrix Factorization (FedMF) server.

Source: Chai, Di, Leye Wang, Kai Chen, and Qiang Yang. "Secure Federated
Matrix Factorization." IEEE Intelligent Systems 36, no. 5 (2021): 11–20.
https://doi.org/10.1109/MIS.2020.3014880

Minimal ``BaseFederatedServer`` subclass. The only override is
``_initialize_server_parameters``, which creates the item-factor table
``V`` and stores the training-set size on ``extra_payload`` for the
optional ``loss_scale="global_M"`` client flag.

Task: explicit rating prediction. Score is ``⟨u_i, v_j⟩``. Loss is
regularised least squares (paper eqs. (2), (5), (6)). Each selected
client downloads ``V``, updates its user factors locally, and uploads a
gradient for ``V``. The server applies ``V ← V − aggregated`` when
``ParamIO.exchange="gradient"`` and the server learning rate is 1.0
(the framework default).

Faithful aggregation is ``Sum`` on ``item_embeddings`` (plain sum of
uploaded gradients; Algorithm 1 / the authors' ``FedMF_Full.py`` server
loop). Aggregators stay experiment configuration; this class does not
import or special-case ``Sum``.

Defaults (reference ``shared_parameter.py`` and ``user_update``):
``latent_dim=100``, constant init ``0.01``. Client defaults are one
sequential pass, per-rating regularisation, client-side ``γ=1e-3``,
``reg_u=reg_v=1e-4``, ``loss_scale="none"``. PartText is the ParamIO
default (``layout="rows"``); FullText is
``param_io_overrides={"item_embeddings": {"layout": "dense"}}``.

Simulation shortcuts:
  Homomorphic encryption, key generation, ciphertext handling, and
  communication-cost simulation are omitted. Gradients travel in
  plaintext, so the Section 4 inversion attack applies. Plaintext
  FullText still reveals the rated-item support because unrated rows
  are exact zeros. ``extra_payload["n_train_ratings"]`` is the global
  training-set size; clients that set ``loss_scale="global_M"`` see it.

Deviations from the paper and the reference code:
  Local update, regularisation placement, learning-rate placement,
  initialisation, and ``1/M`` scaling are constructor flags; defaults
  follow the reference code, not a centralized reading of eq. (2).
  Eq. (6) prints ``λ`` for ``V`` while (2) prints ``μ``; ``reg_u`` and
  ``reg_v`` are separate. There is no server-side once-per-round
  ``2μ V`` term. RegularMF's per-user sequential apply of ``V`` is not
  implemented; the round loop waits for all selected clients, matching
  DistributedMF/FedMF. Client selection, aggregation, evaluation, and
  the split are not hard-coded. Ranking evaluation uses the library
  evaluator; the paper's RMSE is not a library metric.
"""

import torch

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedmf.fedmf_model import MatrixFactorization

_INIT_MODES = ("constant", "normal")


def default_param_ios() -> dict[str, ParamIO]:
    """Return the FedMF ParamIO map.

    ``item_embeddings`` is server-resident, exchanged as a gradient,
    applied on the server, and row-layout (PartText) by default.

    Returns
    -------
    dict of str to ParamIO
        One group, ``item_embeddings``.

    Examples
    --------
    >>> pio = default_param_ios()["item_embeddings"]
    >>> pio.exchange, pio.layout
    ('gradient', 'rows')
    """
    return {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="gradient",
            apply_to="server",
            layout="rows",
        ),
    }


class FedMF(BaseFederatedServer):
    """Federated MF. Server holds the item table; users hold ``u_i``.

    Override item-upload density with
    ``param_io_overrides={"item_embeddings": {"layout": "dense"}}``
    for FullText.

    Parameters
    ----------
    latent_dim : int, optional
        Width of each factor vector. Default is 100.
    init_mode : {"constant", "normal"}, optional
        Item-table initialisation. Default is ``"constant"``.
    init_value : float, optional
        Fill value for ``init_mode="constant"``. Default is 0.01.
    init_std : float, optional
        Standard deviation for ``init_mode="normal"``. Default is 0.01.

    Raises
    ------
    ValueError
        If ``init_mode`` is unknown, or ``init_mode="normal"`` and
        ``seed`` is None.

    Examples
    --------
    >>> server = FedMF(latent_dim=8, interaction_dataset=data, ...)  # doctest: +SKIP
    """

    def __init__(
        self,
        latent_dim: int = 100,
        init_mode: str = "constant",
        init_value: float = 0.01,
        init_std: float = 0.01,
        **kwargs,
    ):
        if init_mode not in _INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {_INIT_MODES}, got {init_mode!r}"
            )
        self.latent_dim = int(latent_dim)
        self.init_mode = init_mode
        self.init_value = float(init_value)
        self.init_std = float(init_std)
        client = kwargs.get("client_algorithm")
        if client is not None and getattr(client, "init_seed", None) is None:
            client.init_seed = kwargs.get("seed")
        super().__init__(param_ios=default_param_ios(), **kwargs)

    def _initialize_server_parameters(self):
        """Create ``V`` on ``self.device`` and publish training-set ``M``.

        ``M`` is the number of training rows only. Clients read it from
        ``extra_payload["n_train_ratings"]`` when ``loss_scale="global_M"``.

        Returns
        -------
        dict
            ``item_embeddings`` group with a CPU clone of ``V``.

        Raises
        ------
        ValueError
            If ``init_mode="normal"`` and ``self.seed`` is None, or if
            the client ``latent_dim`` disagrees with this server.
        """
        client_dim = getattr(self.client_algorithm, "latent_dim", None)
        if client_dim is not None and int(client_dim) != self.latent_dim:
            raise ValueError(
                f"FedMF latent_dim={self.latent_dim} does not match "
                f"client latent_dim={client_dim}"
            )
        self.extra_payload["n_train_ratings"] = int(
            len(self.interaction_dataset.train_interactions)
        )
        generator = None
        if self.init_mode == "normal":
            if self.seed is None:
                raise ValueError(
                    "init_mode='normal' requires seed= on the server."
                )
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
        model = MatrixFactorization(
            num_items=self.interaction_dataset.num_items,
            latent_dim=self.latent_dim,
            init_mode=self.init_mode,
            init_value=self.init_value,
            init_std=self.init_std,
            generator=generator,
        )
        if self.device is not None:
            model = model.to(self.device)
        return model.get_server_state_groups()
