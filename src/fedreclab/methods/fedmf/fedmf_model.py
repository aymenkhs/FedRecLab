"""Item-table skeleton for federated matrix factorization.

Source: Chai, Wang, Chen and Yang, "Secure Federated Matrix Factorization",
IEEE Intelligent Systems 2021.

The server stores item factors ``V``. User factors are client-resident and
are passed into ``forward``; they are not parameters of this module.
"""

import torch
from torch import nn

from fedreclab.federated.utils import flatten_grouped_state

_INIT_MODES = ("constant", "normal")


class MatrixFactorization(nn.Module):
    """Server-resident item factors for inner-product matrix factorization.

    Parameters
    ----------
    num_items : int
        Number of items after ``InteractionDataset`` remapping.
    latent_dim : int, optional
        Width of each factor vector. Default is 100 (reference
        ``shared_parameter.py``).
    init_mode : {"constant", "normal"}, optional
        ``"constant"`` fills every entry with ``init_value``.
        ``"normal"`` draws ``N(0, init_std)`` from ``generator``.
        Default is ``"constant"``.
    init_value : float, optional
        Fill value for ``init_mode="constant"``. Default is 0.01.
    init_std : float, optional
        Standard deviation for ``init_mode="normal"``. Default is 0.01.
    generator : torch.Generator or None, optional
        Seeded generator used when ``init_mode="normal"``. Ignored for
        constant initialisation.

    Raises
    ------
    ValueError
        If ``init_mode`` is unknown.

    Examples
    --------
    >>> m = MatrixFactorization(num_items=4, latent_dim=3)
    >>> tuple(m.get_server_state_groups()["item_embeddings"])
    ('item_embeddings.weight',)
    """

    def __init__(
        self,
        num_items: int,
        latent_dim: int = 100,
        init_mode: str = "constant",
        init_value: float = 0.01,
        init_std: float = 0.01,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if init_mode not in _INIT_MODES:
            raise ValueError(
                f"init_mode must be one of {_INIT_MODES}, got {init_mode!r}"
            )
        self.latent_dim = int(latent_dim)
        self.init_mode = init_mode
        self.init_value = float(init_value)
        self.init_std = float(init_std)
        self.item_embeddings = nn.Embedding(num_items, self.latent_dim)
        self._initialize_weights(generator)

    def _initialize_weights(self, generator: torch.Generator | None) -> None:
        """Overwrite the embedding table; ``nn.Embedding`` default is unused.

        Parameters
        ----------
        generator : torch.Generator or None
            Seeded generator for ``init_mode="normal"``.

        Returns
        -------
        None
        """
        with torch.no_grad():
            weight = self.item_embeddings.weight
            if self.init_mode == "constant":
                weight.fill_(self.init_value)
                return
            weight.normal_(0.0, self.init_std, generator=generator)

    def forward(self, user_vectors: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return ``⟨u, v_j⟩`` for each pair.

        Parameters
        ----------
        user_vectors : torch.Tensor
            User factors, shape ``(batch, latent_dim)``.
        item_ids : torch.Tensor
            Item ids, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Inner products, shape ``(batch,)``.
        """
        item_vectors = self.item_embeddings(item_ids)
        return (user_vectors * item_vectors).sum(dim=-1)

    def get_server_state_groups(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return the item table as a ParamIO group of CPU clones.

        Returns
        -------
        dict of str to dict of str to torch.Tensor
            One group ``item_embeddings`` with key
            ``item_embeddings.weight``.

        Examples
        --------
        >>> m = MatrixFactorization(num_items=2, latent_dim=3)
        >>> m.get_server_state_groups()["item_embeddings"]["item_embeddings.weight"].shape
        torch.Size([2, 3])
        """
        return {
            "item_embeddings": {
                "item_embeddings.weight": (
                    self.item_embeddings.weight.data.detach().cpu().clone()
                ),
            }
        }

    def load_server_weights(self, server_payload: dict) -> None:
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
