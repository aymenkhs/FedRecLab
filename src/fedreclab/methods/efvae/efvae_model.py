"""Embedding Mult-VAE skeleton for EFVAE (FCIS).

Source: Zhang et al., CIKM 2024, and the authors' ``EFMultiVAE`` reference
model. The first encoder layer and last decoder layer are item embedding
tables so Federated Collaborative Importance Sampling can exchange only
selected rows. Middle layers are dense linears (encoder head to
``mu``/``logvar``, symmetric decoder).

Scoring uses ``z = mu`` (no sampling). Dropout and reparameterisation noise
accept an optional ``torch.Generator``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from fedreclab.federated.utils import flatten_grouped_state


def _xavier_normal_(
    tensor: torch.Tensor,
    generator: torch.Generator | None,
) -> None:
    """Fill ``tensor`` with Xavier-normal values (reference init).

    Parameters
    ----------
    tensor : torch.Tensor
        Parameter overwritten in place.
    generator : torch.Generator or None
        CPU generator. ``None`` uses the default RNG.

    Returns
    -------
    None
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = (2.0 / float(fan_in + fan_out)) ** 0.5
    with torch.no_grad():
        if generator is None:
            tensor.normal_(0.0, std)
            return
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator)
        tensor.copy_(noise * std)


def _normal_(
    tensor: torch.Tensor,
    generator: torch.Generator | None,
    std: float = 0.001,
) -> None:
    """Fill ``tensor`` from ``N(0, std)``.

    Parameters
    ----------
    tensor : torch.Tensor
        Parameter overwritten in place.
    generator : torch.Generator or None
        CPU generator. ``None`` uses the default RNG.
    std : float, optional
        Standard deviation. Default is ``0.001`` (reference bias init).

    Returns
    -------
    None
    """
    with torch.no_grad():
        if generator is None:
            tensor.normal_(0.0, std)
            return
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator)
        tensor.copy_(noise * float(std))


class EFMultVAE(nn.Module):
    """Server-resident embedding Mult-VAE for EFVAE.

    Architecture ``|I| -> hidden_dims -> latent_dim`` (encoder) with a
    mirrored dense decoder into an item embedding table of width
    ``hidden_dims[0]``. Defaults ``hidden_dims=(120,)``, ``latent_dim=40``
    match the paper's HN3S-aligned setup.

    Parameters
    ----------
    num_items : int
        Catalogue size after ``InteractionDataset`` remapping.
    hidden_dims : sequence of int, optional
        Encoder hidden widths before the latent split. Default is
        ``(120,)``.
    latent_dim : int, optional
        Size of ``mu`` and ``logvar``. Default is 40.
    dropout : float, optional
        Input / embedding dropout rate. Default is 0.5.
    generator : torch.Generator or None, optional
        CPU generator for initialisation. ``None`` uses the default RNG.

    Raises
    ------
    ValueError
        If sizes are invalid or ``dropout`` is outside ``[0, 1]``.

    Examples
    --------
    >>> m = EFMultVAE(num_items=5, hidden_dims=(4,), latent_dim=2, dropout=0.5)
    >>> logits, mu, logvar = m(torch.zeros(1, 5))
    >>> logits.shape, mu.shape, logvar.shape
    (torch.Size([1, 5]), torch.Size([1, 2]), torch.Size([1, 2]))
    """

    def __init__(
        self,
        num_items: int,
        hidden_dims: Sequence[int] = (120,),
        latent_dim: int = 40,
        dropout: float = 0.5,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        hidden = tuple(int(h) for h in hidden_dims)
        if not isinstance(num_items, int) or isinstance(num_items, bool) or num_items < 1:
            raise ValueError(f"num_items must be a positive int, got {num_items!r}")
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim < 1:
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if not hidden or any(h < 1 for h in hidden):
            raise ValueError(f"hidden_dims must be non-empty positive ints, got {hidden_dims!r}")
        dropout = float(dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.num_items = int(num_items)
        self.hidden_dims = hidden
        self.latent_dim = int(latent_dim)
        self.dropout_p = dropout
        self.emb_dim = int(hidden[0])

        # Encoder: emb -> (optional tanh hiddens) -> linear 2*latent
        enc_widths = [self.emb_dim, *hidden[1:], 2 * self.latent_dim]
        self.q_layers = nn.ModuleList(
            [
                nn.Linear(enc_widths[i], enc_widths[i + 1])
                for i in range(len(enc_widths) - 1)
            ]
        )
        # Decoder: latent -> reverse hiddens -> emb_dim
        dec_widths = [self.latent_dim, *reversed(hidden)]
        self.p_layers = nn.ModuleList(
            [
                nn.Linear(dec_widths[i], dec_widths[i + 1])
                for i in range(len(dec_widths) - 1)
            ]
        )
        self.enc_embeddings = nn.Embedding(self.num_items, self.emb_dim)
        self.enc_bias = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.dec_embeddings = nn.Embedding(self.num_items, self.emb_dim)
        self.dec_bias = nn.Embedding(self.num_items, 1)
        self.init_weights(generator)

    def init_weights(self, generator: torch.Generator | None = None) -> None:
        """Xavier-normal on weights/embeddings; ``N(0, 0.001)`` on biases.

        Parameters
        ----------
        generator : torch.Generator or None, optional
            CPU generator. ``None`` uses the default RNG.

        Returns
        -------
        None
        """
        for layer in list(self.q_layers) + list(self.p_layers):
            _xavier_normal_(layer.weight, generator)
            _normal_(layer.bias, generator, std=0.001)
        _xavier_normal_(self.enc_embeddings.weight, generator)
        _xavier_normal_(self.dec_embeddings.weight, generator)
        _normal_(self.enc_bias, generator, std=0.001)
        _normal_(self.dec_bias.weight, generator, std=0.001)

    def encode_from_hidden(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a post-embedding hidden vector to ``(mu, logvar)``.

        Parameters
        ----------
        h : torch.Tensor
            Shape ``(batch, emb_dim)``, typically after tanh on the
            embedding sum.

        Returns
        -------
        mu, logvar : torch.Tensor
            Each shape ``(batch, latent_dim)``.
        """
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
        mu = h[:, : self.latent_dim]
        logvar = h[:, self.latent_dim :]
        return mu, logvar

    def decode_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent vector through the dense decoder stack.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, emb_dim)``.
        """
        h = z
        for layer in self.p_layers:
            h = torch.tanh(layer(h))
        return h

    def encode_binary(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a bag-of-items batch (full dense path used at scoring).

        Parameters
        ----------
        x : torch.Tensor
            Binary user vectors, shape ``(batch, num_items)``.
        generator : torch.Generator or None, optional
            CPU generator for dropout.

        Returns
        -------
        mu, logvar : torch.Tensor
            Variational parameters.
        """
        h = F.normalize(x, p=2, dim=1)
        if self.training and self.dropout_p > 0.0:
            h = self._inverted_dropout(h, generator)
        h = h @ self.enc_embeddings.weight + self.enc_bias
        h = torch.tanh(h)
        return self.encode_from_hidden(h)

    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-catalogue encode–decode (evaluation / dense ablation).

        Parameters
        ----------
        x : torch.Tensor
            User vectors, shape ``(batch, num_items)``.
        generator : torch.Generator or None, optional
            CPU generator for dropout and ``eps``.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(batch, num_items)``.
        mu, logvar : torch.Tensor
            Shape ``(batch, latent_dim)``.
        """
        mu, logvar = self.encode_binary(x, generator)
        z = self._reparameterize(mu, logvar, generator)
        h = self.decode_hidden(z)
        logits = h @ self.dec_embeddings.weight.T + self.dec_bias.weight.T
        return logits, mu, logvar

    def encode_items(
        self,
        item_ids: torch.Tensor,
        lengths: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FCIS encoder path from padded positive item ids.

        Parameters
        ----------
        item_ids : torch.Tensor
            Shape ``(batch, max_len)``, int64. Padded slots must be valid
            item ids; they are zeroed by ``lengths``.
        lengths : torch.Tensor
            True history lengths, shape ``(batch,)``, int64. Must be
            positive.
        generator : torch.Generator or None, optional
            CPU generator for embedding dropout masks.

        Returns
        -------
        h0 : torch.Tensor
            Post-tanh embedding aggregate, shape ``(batch, emb_dim)``.
        mu, logvar : torch.Tensor
            Variational parameters.

        Raises
        ------
        ValueError
            If any length is non-positive.
        """
        if bool((lengths <= 0).any()):
            raise ValueError("encode_items requires positive lengths")
        emb = self.enc_embeddings(item_ids)
        if self.training and self.dropout_p > 0.0:
            emb = self._embedding_dropout(emb, generator)
        max_len = int(item_ids.shape[1])
        mask = (
            torch.arange(max_len, device=item_ids.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        ).to(dtype=emb.dtype)
        emb = emb * mask.unsqueeze(-1)
        denom = lengths.to(dtype=emb.dtype).sqrt().clamp(min=1e-8).unsqueeze(-1)
        h = emb.sum(dim=1) / denom + self.enc_bias
        h = torch.tanh(h)
        mu, logvar = self.encode_from_hidden(h)
        return h, mu, logvar

    def decoder_logits_for_items(
        self,
        hidden: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score selected items with the decoder embedding table.

        Parameters
        ----------
        hidden : torch.Tensor
            Decoder hidden states, shape ``(batch, emb_dim)``.
        item_ids : torch.Tensor
            Item ids, shape ``(batch, L)``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(batch, L)``.
        """
        emb = self.dec_embeddings(item_ids)
        bias = self.dec_bias(item_ids).squeeze(-1)
        return torch.einsum("bd,bld->bl", hidden, emb) + bias

    def _reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
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
        keep = 1.0 - self.dropout_p
        if keep <= 0.0:
            return torch.zeros_like(x)
        probs = torch.full(x.shape, keep, dtype=x.dtype)
        if generator is None:
            mask = torch.bernoulli(probs.to(device=x.device))
        else:
            mask = torch.bernoulli(probs, generator=generator).to(device=x.device)
        return x * mask / keep

    def _embedding_dropout(
        self,
        emb: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Drop whole embedding channels (reference ``Dropout2d`` on ``B,L,D``).

        Parameters
        ----------
        emb : torch.Tensor
            Shape ``(batch, length, emb_dim)``.
        generator : torch.Generator or None
            CPU generator for the channel mask.

        Returns
        -------
        torch.Tensor
            Same shape as ``emb``.
        """
        keep = 1.0 - self.dropout_p
        if keep <= 0.0:
            return torch.zeros_like(emb)
        # Channel axis = last dim; one Bernoulli per (batch, channel).
        probs = torch.full(
            (emb.shape[0], 1, emb.shape[2]), keep, dtype=emb.dtype
        )
        if generator is None:
            mask = torch.bernoulli(probs.to(device=emb.device))
        else:
            mask = torch.bernoulli(probs, generator=generator).to(device=emb.device)
        return emb * mask / keep

    def get_server_state_groups(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return ParamIO groups as CPU clones.

        Returns
        -------
        dict of str to dict of str to torch.Tensor
            ``enc_embeddings``, ``dec_embeddings``, ``dec_bias`` (row tables)
            and ``middle`` (dense encoder/decoder linears plus ``enc_bias``).

        Examples
        --------
        >>> m = EFMultVAE(num_items=3, hidden_dims=(4,), latent_dim=2)
        >>> sorted(m.get_server_state_groups())
        ['dec_bias', 'dec_embeddings', 'enc_embeddings', 'middle']
        """
        middle: dict[str, torch.Tensor] = {
            "enc_bias": self.enc_bias.detach().cpu().clone(),
        }
        for name, param in self.q_layers.named_parameters(prefix="q_layers"):
            middle[name] = param.detach().cpu().clone()
        for name, param in self.p_layers.named_parameters(prefix="p_layers"):
            middle[name] = param.detach().cpu().clone()
        return {
            "enc_embeddings": {
                "enc_embeddings.weight": self.enc_embeddings.weight.detach().cpu().clone()
            },
            "dec_embeddings": {
                "dec_embeddings.weight": self.dec_embeddings.weight.detach().cpu().clone()
            },
            "dec_bias": {
                "dec_bias.weight": self.dec_bias.weight.detach().cpu().clone()
            },
            "middle": middle,
        }

    def load_server_weights(self, server_payload: dict[str, Any]) -> None:
        """Copy ParamIO groups onto this skeleton.

        Parameters
        ----------
        server_payload : dict
            Groups as produced by ``get_server_state_groups``.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If payload names do not match this skeleton exactly.
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
