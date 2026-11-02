"""Meta-network skeleton for Meta Matrix Factorization.

Source: Lin et al., "Meta Matrix Factorization for Federated Rating
Predictions", SIGIR 2020, and the authors' ``MetaMF.py``.

The module stores only the server-resident meta network Theta:
collaborative memory (user embeddings and shared memory) and the meta
recommender that generates per-user item factors and a two-layer rating
MLP. Generated parameters are outputs, not members of this module.

Item generation follows the reference code: one hidden layer
(``hidden_layer_3``) feeds both ``emb_layer_1`` (low-dimensional item
factors) and ``emb_layer_2`` (the rise-dimensional matrix). The paper's
eq. (3) gives those two maps separate hidden layers.

Initialisation follows the reference ``weights_init``: Xavier-normal on
``Linear`` weights, zero Linear biases, Xavier-normal on the memory,
and ``N(0, 1)`` on the user embedding (the ``nn.Embedding`` default).
The paper text says Xavier for all trainable parameters.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.fedreclab.federated.utils import flatten_grouped_state

ITEM_PAYLOADS = ("factors", "full")
FACTOR_PHI_KEYS = ("item_left", "item_right", "w1", "b1", "w2", "b2")
FULL_PHI_KEYS = ("item_embeddings", "w1", "b1", "w2", "b2")
RP_KEYS = ("w1", "b1", "w2", "b2")


def resolve_rp_hidden_dim(item_emb_dim: int, rp_hidden_dim: int | None) -> int:
    """Return the rating-MLP hidden width.

    ``None`` is the reference ``int(item_emb_size / 4)``, so changing
    only ``item_emb_dim`` reproduces the paper's Table 4 scales.

    Parameters
    ----------
    item_emb_dim : int
        Width of the generated item embeddings.
    rp_hidden_dim : int or None
        Explicit hidden width, or ``None`` to use ``item_emb_dim // 4``.

    Returns
    -------
    int
        Hidden width, at least 1.

    Raises
    ------
    ValueError
        If ``rp_hidden_dim`` is not a positive int or ``None``, or the
        resolved width is below 1 (for example ``item_emb_dim < 4``
        with ``rp_hidden_dim=None``).

    Examples
    --------
    >>> resolve_rp_hidden_dim(32, None)
    8
    >>> resolve_rp_hidden_dim(8, None)
    2
    """
    if rp_hidden_dim is None:
        width = int(item_emb_dim) // 4
    else:
        if (
            not isinstance(rp_hidden_dim, int)
            or isinstance(rp_hidden_dim, bool)
            or rp_hidden_dim < 1
        ):
            raise ValueError(
                f"rp_hidden_dim must be a positive int or None, got {rp_hidden_dim!r}"
            )
        width = int(rp_hidden_dim)
    if width < 1:
        raise ValueError(
            f"rp_hidden_dim resolved to {width}; must be >= 1 "
            f"(item_emb_dim={item_emb_dim}, rp_hidden_dim={rp_hidden_dim!r})"
        )
    return width


def _xavier_normal_(tensor: torch.Tensor, generator: torch.Generator | None) -> None:
    """Fill ``tensor`` with Xavier-normal values, optionally from ``generator``.

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
    with torch.no_grad():
        if generator is None:
            tensor.normal_(0.0, std)
            return
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator)
        tensor.copy_(noise * std)


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


def predict_ratings(phi_u: dict[str, torch.Tensor], item_ids: torch.Tensor) -> torch.Tensor:
    """Score items with a generated per-user rating MLP.

    Accepts either payload form. For ``factors``, item embeddings are
    ``I^l_u[item_ids] @ I^r_u`` and the full ``I_u`` is never built.

    Parameters
    ----------
    phi_u : dict of str to torch.Tensor
        One user's generated parameters, unbatched. Factor form uses
        ``item_left`` of shape ``(n_items, s)`` and ``item_right`` of
        shape ``(s, d)``. Full form uses ``item_embeddings`` of shape
        ``(n_items, d)``. Rating weights are ``w1 (d, h)``, ``b1 (1, h)``,
        ``w2 (h, 1)``, ``b2 (1, 1)``.
    item_ids : torch.Tensor
        Item ids, shape ``(n,)``, on the same device as ``phi_u``.

    Returns
    -------
    torch.Tensor
        Predicted ratings, shape ``(n,)``.

    Raises
    ------
    KeyError
        If neither ``item_embeddings`` nor both factor keys are present.

    Examples
    --------
    >>> # doctest: +SKIP
    >>> predict_ratings(phi_u, torch.tensor([0, 2])).shape
    torch.Size([2])
    """
    item_emb = _item_embeddings_at(phi_u, item_ids)
    return apply_rating_mlp(
        item_emb, phi_u["w1"], phi_u["b1"], phi_u["w2"], phi_u["b2"]
    )


def apply_rating_mlp(
    item_emb: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Apply the two-layer rating MLP of the prediction module.

    ``ReLU(e W1 + b1) W2 + b2``, matching the reference ``MetaMF.forward``.

    Parameters
    ----------
    item_emb : torch.Tensor
        Item vectors, shape ``(n, d)``.
    w1 : torch.Tensor
        First-layer weights, shape ``(d, h)``.
    b1 : torch.Tensor
        First-layer bias, shape ``(1, h)`` or ``(h,)``.
    w2 : torch.Tensor
        Output weights, shape ``(h, 1)``.
    b2 : torch.Tensor
        Output bias, shape ``(1, 1)`` or ``(1,)`` or scalar.

    Returns
    -------
    torch.Tensor
        Predicted ratings, shape ``(n,)``.

    Examples
    --------
    >>> e = torch.zeros(3, 4)
    >>> w1 = torch.zeros(4, 2)
    >>> b1 = torch.zeros(1, 2)
    >>> w2 = torch.zeros(2, 1)
    >>> b2 = torch.zeros(1, 1)
    >>> apply_rating_mlp(e, w1, b1, w2, b2).shape
    torch.Size([3])
    """
    hidden = F.relu(item_emb @ w1 + b1)
    return (hidden @ w2 + b2).reshape(-1)


def _item_embeddings_at(
    phi_u: dict[str, torch.Tensor], item_ids: torch.Tensor
) -> torch.Tensor:
    """Index generated item embeddings at ``item_ids``."""
    if "item_embeddings" in phi_u:
        return phi_u["item_embeddings"].index_select(0, item_ids)
    if "item_left" in phi_u and "item_right" in phi_u:
        left_rows = phi_u["item_left"].index_select(0, item_ids)
        return left_rows @ phi_u["item_right"]
    raise KeyError(
        "phi_u must contain 'item_embeddings' or both 'item_left' and "
        f"'item_right'; got {sorted(phi_u)}"
    )


def unbatch_phi(batched: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    """Take one user's generated parameters from a batched ``generate`` output.

    Parameters
    ----------
    batched : dict of str to torch.Tensor
        Tensors whose leading dimension is the user batch.
    index : int
        User position in that batch.

    Returns
    -------
    dict of str to torch.Tensor
        Views of row ``index``, still attached to the batched graph.

    Examples
    --------
    >>> batched = {"w2": torch.zeros(2, 3, 1)}
    >>> unbatch_phi(batched, 1)["w2"].shape
    torch.Size([3, 1])
    """
    return {key: tensor[index] for key, tensor in batched.items()}


class MetaRecommender(nn.Module):
    """Server-resident meta network that generates per-user MF parameters.

    Parameter names match the reference ``MetaRecommender`` so the
    state dict can be traced back to that code.

    Parameters
    ----------
    num_users : int
        Number of users after ``InteractionDataset`` remapping.
    num_items : int
        Catalogue size after remapping.
    user_emb_dim : int, optional
        Width of ``e_u``. Default is 32.
    item_emb_dim : int, optional
        Width of generated item embeddings. Default is 32.
    item_mem_num : int, optional
        Low-dimensional item width ``s``. Default is 8.
    mem_size : int, optional
        Collaborative-vector width ``k``. Default is 128.
    hidden_size : int, optional
        Meta-recommender hidden width ``o``. Default is 512.
    rp_hidden_dim : int or None, optional
        Rating-MLP hidden width. ``None`` (default) is
        ``item_emb_dim // 4``.
    generator : torch.Generator or None, optional
        CPU generator for initialisation. ``None`` uses the default RNG.

    Raises
    ------
    ValueError
        If a dimension is not a positive int, or the resolved rating
        hidden width is below 1.

    Examples
    --------
    >>> m = MetaRecommender(num_users=3, num_items=5, mem_size=4, hidden_size=6)
    >>> phi = m.generate(torch.tensor([0, 2]))
    >>> phi["item_left"].shape
    torch.Size([2, 5, 8])
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb_dim: int = 32,
        item_emb_dim: int = 32,
        item_mem_num: int = 8,
        mem_size: int = 128,
        hidden_size: int = 512,
        rp_hidden_dim: int | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        for name, value in (
            ("num_users", num_users),
            ("num_items", num_items),
            ("user_emb_dim", user_emb_dim),
            ("item_emb_dim", item_emb_dim),
            ("item_mem_num", item_mem_num),
            ("mem_size", mem_size),
            ("hidden_size", hidden_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.user_emb_dim = int(user_emb_dim)
        self.item_emb_dim = int(item_emb_dim)
        self.item_mem_num = int(item_mem_num)
        self.mem_size = int(mem_size)
        self.hidden_size = int(hidden_size)
        self.rp_hidden_dim = resolve_rp_hidden_dim(self.item_emb_dim, rp_hidden_dim)

        self.user_embedding = nn.Embedding(self.num_users, self.user_emb_dim)
        self.memory = nn.Parameter(torch.empty(self.user_emb_dim, self.mem_size))

        self.hidden_layer_1, self.weight_layer_1, self.bias_layer_1 = (
            self._define_one_layer(
                self.mem_size,
                self.hidden_size,
                self.item_emb_dim,
                self.rp_hidden_dim,
            )
        )
        self.hidden_layer_2, self.weight_layer_2, self.bias_layer_2 = (
            self._define_one_layer(
                self.mem_size,
                self.hidden_size,
                self.rp_hidden_dim,
                1,
            )
        )
        self.hidden_layer_3, self.emb_layer_1, self.emb_layer_2 = (
            self._define_item_embedding()
        )
        self._initialize_weights(generator)

    def _define_one_layer(
        self, mem_size: int, hidden_size: int, in_size: int, out_size: int
    ) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Build the three Linear maps of one generated RP layer."""
        hidden_layer = nn.Linear(mem_size, hidden_size)
        weight_layer = nn.Linear(hidden_size, in_size * out_size)
        bias_layer = nn.Linear(hidden_size, out_size)
        return hidden_layer, weight_layer, bias_layer

    def _define_item_embedding(self) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Build the shared item-generation hidden layer and two heads."""
        hidden_layer = nn.Linear(self.mem_size, self.hidden_size)
        emb_layer_1 = nn.Linear(
            self.hidden_size, self.num_items * self.item_mem_num
        )
        emb_layer_2 = nn.Linear(
            self.hidden_size, self.item_mem_num * self.item_emb_dim
        )
        return hidden_layer, emb_layer_1, emb_layer_2

    def _initialize_weights(self, generator: torch.Generator | None) -> None:
        """Overwrite parameters with the reference initialisation.

        Parameters
        ----------
        generator : torch.Generator or None
            Seeded CPU generator, or ``None`` for the default RNG.

        Returns
        -------
        None
        """
        _normal_(self.user_embedding.weight, generator)
        _xavier_normal_(self.memory, generator)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.Linear):
                _xavier_normal_(module.weight, generator)
                with torch.no_grad():
                    module.bias.zero_()

    def collaborative_vector(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return ``c_u = e_u M`` for each user.

        Parameters
        ----------
        user_ids : torch.Tensor
            User ids, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Collaborative vectors, shape ``(batch, mem_size)``.
        """
        user_emb = self.user_embedding(user_ids)
        return user_emb @ self.memory

    def _rating_weights(self, cf_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate the two-layer rating MLP from the collaborative vector."""
        hid1 = F.relu(self.hidden_layer_1(cf_vec))
        w1 = self.weight_layer_1(hid1).view(
            -1, self.item_emb_dim, self.rp_hidden_dim
        )
        b1 = self.bias_layer_1(hid1).view(-1, 1, self.rp_hidden_dim)
        hid2 = F.relu(self.hidden_layer_2(cf_vec))
        w2 = self.weight_layer_2(hid2).view(-1, self.rp_hidden_dim, 1)
        b2 = self.bias_layer_2(hid2).view(-1, 1, 1)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _item_factors(self, cf_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate ``I^l_u`` and ``I^r_u`` for the full catalogue."""
        hid = F.relu(self.hidden_layer_3(cf_vec))
        item_left = self.emb_layer_1(hid).view(
            -1, self.num_items, self.item_mem_num
        )
        item_right = self.emb_layer_2(hid).view(
            -1, self.item_mem_num, self.item_emb_dim
        )
        return item_left, item_right

    def generate(
        self, user_ids: torch.Tensor, item_payload: str = "factors"
    ) -> dict[str, torch.Tensor]:
        """Generate per-user item factors and rating-MLP weights.

        Parameters
        ----------
        user_ids : torch.Tensor
            User ids, shape ``(batch,)``.
        item_payload : {"factors", "full"}, optional
            ``"factors"`` (default) returns ``item_left`` and
            ``item_right``. ``"full"`` returns ``item_embeddings = I^l I^r``.

        Returns
        -------
        dict of str to torch.Tensor
            Batched generated parameters. Leading dimension is ``batch``.

        Raises
        ------
        ValueError
            If ``item_payload`` is not ``"factors"`` or ``"full"``.

        Examples
        --------
        >>> m = MetaRecommender(num_users=2, num_items=3, mem_size=4, hidden_size=6)
        >>> phi = m.generate(torch.tensor([0]))
        >>> sorted(phi)
        ['b1', 'b2', 'item_left', 'item_right', 'w1', 'w2']
        """
        if item_payload not in ITEM_PAYLOADS:
            raise ValueError(
                f"item_payload must be one of {ITEM_PAYLOADS}, got {item_payload!r}"
            )
        cf_vec = self.collaborative_vector(user_ids)
        out = self._rating_weights(cf_vec)
        item_left, item_right = self._item_factors(cf_vec)
        if item_payload == "factors":
            out["item_left"] = item_left
            out["item_right"] = item_right
        else:
            out["item_embeddings"] = item_left @ item_right
        return out

    def item_rows(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        item_payload: str = "factors",
    ) -> dict[str, torch.Tensor]:
        """Generate Phi only for ``item_ids``, by slicing ``emb_layer_1``.

        The reference reshapes ``emb_layer_1`` row-major, so item ``j``
        owns output units ``j s, ..., j s + s - 1``. This method gathers
        those rows of the Linear map and never materialises the full
        ``n x s`` output.

        Parameters
        ----------
        user_ids : torch.Tensor
            User ids, shape ``(batch,)``.
        item_ids : torch.Tensor
            Item ids, shape ``(n_sel,)``. Duplicates are allowed and
            produce duplicate rows. An empty tensor yields an empty
            item-factor leading dimension.
        item_payload : {"factors", "full"}, optional
            Same meaning as in :meth:`generate`.

        Returns
        -------
        dict of str to torch.Tensor
            Batched generated parameters. ``item_left`` / ``item_embeddings``
            have shape ``(batch, n_sel, ...)``. Rating weights match
            :meth:`generate`.

        Raises
        ------
        ValueError
            If ``item_payload`` is unknown.
        IndexError
            If an item id is outside ``[0, num_items)``.

        Examples
        --------
        >>> m = MetaRecommender(num_users=2, num_items=5, mem_size=4, hidden_size=6)
        >>> phi = m.item_rows(torch.tensor([0]), torch.tensor([0, 4]))
        >>> phi["item_left"].shape
        torch.Size([1, 2, 8])
        """
        if item_payload not in ITEM_PAYLOADS:
            raise ValueError(
                f"item_payload must be one of {ITEM_PAYLOADS}, got {item_payload!r}"
            )
        item_ids = item_ids.reshape(-1).long()
        if item_ids.numel() and (
            torch.any(item_ids < 0) or torch.any(item_ids >= self.num_items)
        ):
            raise IndexError(
                f"item_ids must be in [0, {self.num_items}), "
                f"got min={int(item_ids.min())} max={int(item_ids.max())}"
            )

        cf_vec = self.collaborative_vector(user_ids)
        out = self._rating_weights(cf_vec)
        hid = F.relu(self.hidden_layer_3(cf_vec))
        n_sel = int(item_ids.numel())
        batch = int(cf_vec.shape[0])
        s = self.item_mem_num
        if n_sel == 0:
            item_left = hid.new_zeros(batch, 0, s)
        else:
            idx = (
                item_ids.unsqueeze(1) * s
                + torch.arange(s, device=item_ids.device, dtype=item_ids.dtype)
            ).reshape(-1)
            weight = self.emb_layer_1.weight.index_select(0, idx)
            bias = self.emb_layer_1.bias.index_select(0, idx)
            item_left = (hid @ weight.transpose(0, 1) + bias).view(batch, n_sel, s)
        item_right = self.emb_layer_2(hid).view(
            batch, s, self.item_emb_dim
        )
        if item_payload == "factors":
            out["item_left"] = item_left
            out["item_right"] = item_right
        else:
            out["item_embeddings"] = item_left @ item_right
        return out

    def get_server_state_groups(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return every parameter as the dense ``meta_network`` ParamIO group.

        Inner keys are fully qualified ``named_parameters()`` names.
        Values are CPU clones. The returned dict is a new object, so
        callers that store it on ``global_parameters`` get a distinct
        identity after each write-back.

        Returns
        -------
        dict of str to dict of str to torch.Tensor
            One group, ``meta_network``.

        Examples
        --------
        >>> m = MetaRecommender(num_users=2, num_items=3, mem_size=4, hidden_size=6)
        >>> "user_embedding.weight" in m.get_server_state_groups()["meta_network"]
        True
        """
        return {
            "meta_network": {
                name: param.detach().cpu().clone()
                for name, param in self.named_parameters()
            }
        }

    def load_server_weights(self, server_payload: dict[str, Any]) -> None:
        """Copy server groups onto this skeleton in place.

        Copies with ``param.copy_`` so a ``torch.optim.Adam`` keyed by
        Parameter identity keeps its moments. Never reassigns a
        Parameter, never uses ``load_state_dict(..., assign=True)``, and
        never rebuilds the module.

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
        missing = set(named) - set(flat)
        extra = set(flat) - set(named)
        if missing or extra:
            raise KeyError(
                "Server payload / skeleton mismatch: "
                f"missing_from_payload={sorted(missing)} extra_in_payload={sorted(extra)}"
            )
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(torch.as_tensor(flat[name], device=param.device, dtype=param.dtype))
