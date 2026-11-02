"""Multinomial variational autoencoder skeleton for FedVAE.

Source: Polato, "Federated Variational Autoencoder for Collaborative
Filtering", IJCNN 2021, and Liang et al., "Variational Autoencoders for
Collaborative Filtering", WWW 2018.

The architecture matches rectorch ``MultiVAE_net``: L2-normalised input,
inverted dropout on the input in training only, tanh hidden layers, a
linear encoder head split into ``mu`` and ``logvar``, and a linear
decoder output. Weights use Xavier-uniform and biases ``N(0, 1)``, as in
that reference code. Algorithm 1's "N(0, 1)" is read as the bias
initialisation.

Scoring uses ``z = mu`` (no sampling). Dropout and the reparameterisation
noise are drawn from an explicit ``torch.Generator`` when one is passed,
so they do not depend on the global RNG.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.fedreclab.federated.utils import flatten_grouped_state


def _xavier_uniform_(tensor: torch.Tensor, generator: torch.Generator | None) -> None:
    """Fill ``tensor`` with Xavier-uniform values, optionally from ``generator``.

    Parameters
    ----------
    tensor : torch.Tensor
        Parameter to overwrite in place.
    generator : torch.Generator or None
        CPU generator. ``None`` uses the default RNG.

    Returns
    -------
    None
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        if generator is None:
            tensor.uniform_(-bound, bound)
            return
        noise = torch.rand(tensor.shape, dtype=tensor.dtype, generator=generator)
        tensor.copy_(noise * (2.0 * bound) - bound)


def _normal_(tensor: torch.Tensor, generator: torch.Generator | None) -> None:
    """Fill ``tensor`` from ``N(0, 1)``, optionally from ``generator``.

    Parameters
    ----------
    tensor : torch.Tensor
        Parameter to overwrite in place.
    generator : torch.Generator or None
        CPU generator. ``None`` uses the default RNG.

    Returns
    -------
    None
    """
    with torch.no_grad():
        if generator is None:
            tensor.normal_(0.0, 1.0)
            return
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator)
        tensor.copy_(noise)


class MultVAE(nn.Module):
    """Server-resident Mult-VAE (encoder, decoder, no per-user state).

    The default widths ``hidden_dims=(600,)``, ``latent_dim=200`` are the
    paper's architecture ``|I| -> 600 -> 200`` (encoder) with a symmetric
    decoder. ``hidden_dims`` lists the tanh hidden widths of the encoder;
    the decoder reverses them.

    Parameters
    ----------
    num_items : int
        Catalogue size after ``InteractionDataset`` remapping.
    hidden_dims : sequence of int, optional
        Encoder hidden widths. Default is ``(600,)``.
    latent_dim : int, optional
        Size of ``mu`` and ``logvar``. Default is 200.
    dropout : float, optional
        Input dropout rate in training. Default is 0.5.
    generator : torch.Generator or None, optional
        CPU generator used for Xavier/normal initialisation. ``None``
        uses the default RNG.

    Raises
    ------
    ValueError
        If ``num_items``, ``latent_dim``, or a hidden width is not a
        positive int, or if ``dropout`` is outside ``[0, 1]``.

    Examples
    --------
    >>> m = MultVAE(num_items=5, hidden_dims=(4,), latent_dim=2, dropout=0.5)
    >>> logits, mu, logvar = m(torch.zeros(1, 5))
    >>> logits.shape, mu.shape, logvar.shape
    (torch.Size([1, 5]), torch.Size([1, 2]), torch.Size([1, 2]))
    """

    def __init__(
        self,
        num_items: int,
        hidden_dims: Sequence[int] = (600,),
        latent_dim: int = 200,
        dropout: float = 0.5,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        hidden = tuple(int(h) for h in hidden_dims)
        if not isinstance(num_items, int) or isinstance(num_items, bool) or num_items < 1:
            raise ValueError(f"num_items must be a positive int, got {num_items!r}")
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim < 1:
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if any(h < 1 for h in hidden):
            raise ValueError(f"hidden_dims must be positive ints, got {hidden_dims!r}")
        dropout = float(dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.num_items = int(num_items)
        self.hidden_dims = hidden
        self.latent_dim = int(latent_dim)
        self.dropout_p = dropout

        enc_in = [self.num_items, *hidden]
        enc_out = [*hidden, 2 * self.latent_dim]
        self.enc_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(enc_in, enc_out)]
        )
        dec_in = [self.latent_dim, *reversed(hidden)]
        dec_out = [*reversed(hidden), self.num_items]
        self.dec_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(dec_in, dec_out)]
        )
        self.init_weights(generator)

    def init_weights(self, generator: torch.Generator | None = None) -> None:
        """Xavier-uniform weights and ``N(0, 1)`` biases.

        Parameters
        ----------
        generator : torch.Generator or None, optional
            CPU generator. ``None`` uses the default RNG.

        Returns
        -------
        None
        """
        for layer in list(self.enc_layers) + list(self.dec_layers):
            _xavier_uniform_(layer.weight, generator)
            _normal_(layer.bias, generator)

    def encode(self, x: torch.Tensor, generator: torch.Generator | None = None):
        """Return ``(mu, logvar)`` from a bag-of-items batch.

        The batch is L2-normalised. In training, inverted dropout is then
        applied with rate ``dropout_p``. Hidden layers use tanh; the last
        encoder layer is linear and split in half.

        Parameters
        ----------
        x : torch.Tensor
            Binary (or weighted) user vectors, shape ``(batch, num_items)``.
        generator : torch.Generator or None, optional
            CPU generator for dropout masks. ``None`` uses the default RNG.

        Returns
        -------
        mu : torch.Tensor
            Mean of ``q(z|x)``, shape ``(batch, latent_dim)``.
        logvar : torch.Tensor
            Log-variance of ``q(z|x)``, same shape.

        Examples
        --------
        >>> mu, logvar = MultVAE(3, (2,), 2)(torch.ones(1, 3))[:2]  # doctest: +SKIP
        """
        h = F.normalize(x, p=2, dim=1)
        if self.training and self.dropout_p > 0.0:
            h = self._inverted_dropout(h, generator)
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
        mu = h[:, : self.latent_dim]
        logvar = h[:, self.latent_dim :]
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent sample to item logits.

        Hidden decoder layers use tanh; the last layer is linear (no
        softmax). Callers apply ``log_softmax`` in the multinomial loss.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors, shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(batch, num_items)``.
        """
        h = z
        for layer in self.dec_layers[:-1]:
            h = torch.tanh(layer(h))
        return self.dec_layers[-1](h)

    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, sample (in training) or take ``mu`` (in eval), and decode.

        Parameters
        ----------
        x : torch.Tensor
            User vectors, shape ``(batch, num_items)``.
        generator : torch.Generator or None, optional
            CPU generator for dropout and ``eps``. ``None`` uses the
            default RNG.

        Returns
        -------
        logits : torch.Tensor
            Decoder output, shape ``(batch, num_items)``.
        mu : torch.Tensor
            Variational mean, shape ``(batch, latent_dim)``.
        logvar : torch.Tensor
            Variational log-variance, same shape as ``mu``.

        Examples
        --------
        >>> m = MultVAE(4, (3,), 2)
        >>> m.eval()
        >>> logits, mu, logvar = m(torch.zeros(1, 4))
        >>> torch.equal(logits, m.decode(mu))
        True
        """
        mu, logvar = self.encode(x, generator)
        z = self._reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar

    def _reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Return ``mu + eps * std`` in training, ``mu`` in eval.

        Parameters
        ----------
        mu, logvar : torch.Tensor
            Variational parameters, shape ``(batch, latent_dim)``.
        generator : torch.Generator or None
            CPU generator for ``eps``.

        Returns
        -------
        torch.Tensor
            Latent sample or mean, same shape as ``mu``.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        if generator is None:
            eps = torch.randn_like(std)
        else:
            eps = torch.randn(std.shape, dtype=std.dtype, generator=generator).to(
                device=std.device
            )
        return mu + eps * std

    def _inverted_dropout(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Keep each entry with probability ``1 - dropout_p`` and rescale.

        Matches ``nn.Dropout`` (inverted dropout). The mask is drawn on
        CPU from ``generator`` when one is given, then moved.

        Parameters
        ----------
        x : torch.Tensor
            Normalised input, shape ``(batch, num_items)``.
        generator : torch.Generator or None
            CPU generator for the Bernoulli mask.

        Returns
        -------
        torch.Tensor
            Dropped-out input, same shape as ``x``.
        """
        keep = 1.0 - self.dropout_p
        if keep <= 0.0:
            return torch.zeros_like(x)
        probs = torch.full(x.shape, keep, dtype=x.dtype)
        if generator is None:
            mask = torch.bernoulli(probs.to(device=x.device))
        else:
            mask = torch.bernoulli(probs, generator=generator).to(device=x.device)
        return x * mask / keep

    def get_server_state_groups(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return every parameter as the dense ``vae`` ParamIO group.

        Inner keys are fully qualified ``named_parameters()`` names.
        Values are CPU clones.

        Returns
        -------
        dict of str to dict of str to torch.Tensor
            One group, ``vae``.

        Examples
        --------
        >>> m = MultVAE(num_items=3, hidden_dims=(2,), latent_dim=2)
        >>> sorted(m.get_server_state_groups()["vae"])[:2]
        ['dec_layers.0.bias', 'dec_layers.0.weight']
        """
        return {
            "vae": {
                name: param.detach().cpu().clone()
                for name, param in self.named_parameters()
            }
        }

    def load_server_weights(self, server_payload: dict[str, Any]) -> None:
        """Copy server groups onto this skeleton.

        Accepts torch tensors or NumPy arrays (post-aggregation values
        may be arrays).

        Parameters
        ----------
        server_payload : dict
            ParamIO groups as produced by ``get_server_state_groups``.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the payload names do not match this skeleton exactly.
        """
        flat = flatten_grouped_state(server_payload)
        named = {n: p for n, p in self.named_parameters()}
        named.update({n: b for n, b in self.named_buffers()})
        missing = set(named) - set(flat)
        extra = set(flat) - set(named)
        if missing or extra:
            raise KeyError(
                "Server payload / skeleton mismatch: "
                f"missing_from_payload={sorted(missing)} extra_in_payload={sorted(extra)}"
            )
        for name, tensor in named.items():
            tensor.data.copy_(torch.as_tensor(flat[name], device=tensor.device))
