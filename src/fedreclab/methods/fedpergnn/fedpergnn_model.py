"""GAT skeleton and inner-product predictor for FedPerGNN.

Wu et al., Nature Communications 2022. The supplementary information
uses a graph attention network and a dot-product rating predictor
:math:`\\hat y_{ij} = h_i^u \\cdot h_{i,j}^t`. The public Keras copy
instead applies ``Dense(1, sigmoid)`` to ``concat([user, item])``. That
head is additive: within a user, ranking reduces to a function of the
item vector alone. This module follows the paper, not that GitHub head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedPerGNNModel(nn.Module):
    """Server-resident GAT and item table; the owner user row is passed in.

    History and neighbour attention build a user hidden state that does
    not depend on the target item. ``user_representation`` returns that
    state once per user. ``score`` is its inner product with the target
    item embeddings (SI predictor). There is no concat-linear output
    layer and no output sigmoid: ranking order of an inner product is
    unchanged by a monotone map, and rating regression needs an
    unbounded score.

    The sigmoid on ``history_item_proj`` is a hidden activation from the
    Keras GAT, not the predictor. Target items that are not in the local
    subgraph use the item embedding; they have no per-user GNN item node.

    Parameters
    ----------
    num_items : int
        Catalogue size after ``InteractionDataset`` remapping. The
        embedding has ``num_items + 1`` rows: index 0 is padding and
        catalogue item ``i`` is row ``i + 1``.
    hidden_dim : int
        Width of embeddings and GAT hidden states.
    history_len : int
        Padded history length ``L``.
    neighbor_len : int
        Padded neighbour count ``N`` per history item.

    Examples
    --------
    >>> m = FedPerGNNModel(num_items=3, hidden_dim=4, history_len=2, neighbor_len=1)
    >>> "output_layer.weight" in m.get_server_state()
    False
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int,
        history_len: int,
        neighbor_len: int,
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len

        # Row 0 is padding. Catalogue item i is row i + 1, so item 0
        # is not the padding token. Callers mark an empty history slot
        # with -1; the lookup adds 1 and lands on this row.
        self.item_embeddings = nn.Embedding(
            num_items + 1, hidden_dim, padding_idx=0
        )

        # Dense used to project history item embeddings (uiemb in Keras code).
        self.history_item_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dense used to project neighbor user embeddings.
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)

        # GAT-style attention over neighbors for each history item.
        # Implemented as two-layer MLP over [item_emb, neighbor_emb].
        self.neighbor_att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Combine aggregated neighbor embedding with item embedding for history.
        self.history_with_neighbors = nn.Linear(2 * hidden_dim, hidden_dim)

        # Attention over history items conditioned on the user embedding.
        self.history_att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Combine history-based user representation with base user embedding.
        self.user_final_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # normal_ fills the padding row as well. padding_idx only blocks
        # its gradient; a nonzero value would still mix into a history
        # that is entirely padding.
        with torch.no_grad():
            self.item_embeddings.weight[0].zero_()

    ITEM_EMBEDDING_NAME = "item_embeddings.weight"

    def get_model_state(self) -> dict:
        """Server-resident tensors excluding the item embedding table."""
        state: dict[str, torch.Tensor] = {}
        for k, v in self.named_parameters():
            if k == self.ITEM_EMBEDDING_NAME:
                continue
            state[k] = v.detach().cpu().clone()
        for k, v in self.named_buffers():
            if k == self.ITEM_EMBEDDING_NAME:
                continue
            state[k] = v.detach().cpu().clone()
        return state

    def get_item_embeddings(self) -> torch.Tensor:
        """Catalogue item table, without the padding row.

        Returns
        -------
        torch.Tensor
            CPU float tensor of shape ``(num_items, hidden_dim)``.
            Row ``i`` is catalogue item ``i`` (module row ``i + 1``).

        Examples
        --------
        >>> m = FedPerGNNModel(num_items=2, hidden_dim=1, history_len=1, neighbor_len=1)
        >>> m.load_item_embeddings(torch.tensor([[3.0], [4.0]]))
        >>> m.get_item_embeddings()[:, 0].tolist()
        [3.0, 4.0]
        """
        return self.item_embeddings.weight.detach()[1:].cpu().clone()

    def get_server_state(self) -> dict:
        """Every skeleton tensor name, with the catalogue item table.

        ``item_embeddings.weight`` here is the table from
        :meth:`get_item_embeddings`, shape ``(num_items, hidden_dim)``.
        The module parameter has one extra padding row, which is not
        included.
        """
        state = self.get_model_state()
        state[self.ITEM_EMBEDDING_NAME] = self.get_item_embeddings()
        return state

    def load_server_state(self, server_state: dict) -> None:
        """Load non-item server tensors from ``get_model_state()``."""
        self._load_named(server_state, exclude={self.ITEM_EMBEDDING_NAME})

    def load_item_embeddings(self, table) -> None:
        """Load the catalogue item table into embedding rows ``1:``.

        Parameters
        ----------
        table : torch.Tensor or array_like
            Shape ``(num_items, hidden_dim)``. Row ``i`` is catalogue
            item ``i``. The padding row at index 0 is set to zero and
            is not read from ``table``.

        Raises
        ------
        ValueError
            If ``table`` does not have shape ``(num_items, hidden_dim)``.

        Examples
        --------
        >>> m = FedPerGNNModel(num_items=2, hidden_dim=1, history_len=1, neighbor_len=1)
        >>> m.load_item_embeddings(torch.tensor([[3.0], [4.0]]))
        >>> m.item_embeddings.weight[:, 0].tolist()
        [0.0, 3.0, 4.0]
        """
        weight = self.item_embeddings.weight
        tensor = table if isinstance(table, torch.Tensor) else torch.as_tensor(table)
        tensor = tensor.to(device=weight.device, dtype=weight.dtype)
        expected = (self.num_items, self.hidden_dim)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                "item embedding table must have shape "
                f"{expected}; got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            weight[1:].copy_(tensor)
            weight[0].zero_()

    def _load_named(self, server_state: dict, exclude: set[str]) -> None:
        def as_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

        named = {n: p for n, p in self.named_parameters() if n not in exclude}
        named.update({n: b for n, b in self.named_buffers() if n not in exclude})
        missing = set(named) - set(server_state)
        extra = set(server_state) - set(named)
        if missing or extra:
            raise KeyError(
                "Server payload / skeleton mismatch: "
                f"missing_from_payload={sorted(missing)} extra_in_payload={sorted(extra)}"
            )
        for name, tensor in named.items():
            tensor.data.copy_(as_tensor(server_state[name]).to(device=tensor.device))

    def user_representation(
        self,
        history_item_ids: torch.Tensor,
        user_embedding: torch.Tensor,
        neighbor_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the item-independent GAT hidden state for one user.

        History and neighbour tensors are encoded once. Call ``score``
        with the result for every target item of this user.

        Parameters
        ----------
        history_item_ids : torch.Tensor
            History ids, shape ``(L,)``, dtype int64. A negative entry
            is an empty slot. A non-negative entry is a catalogue id,
            including 0. The lookup adds 1, so an empty slot reads
            padding row 0 and catalogue item ``i`` reads row ``i + 1``.
        user_embedding : torch.Tensor
            Client-resident owner row, shape ``(H,)`` or ``(1, H)``.
        neighbor_embeddings : torch.Tensor or None, optional
            Neighbour rows, shape ``(L, N, H)``. ``None`` skips the
            neighbour GAT and uses projected history items only.
            All-zero neighbour rows are padding.

        Returns
        -------
        torch.Tensor
            User hidden state, shape ``(H,)``.

        Raises
        ------
        ValueError
            If ``history_item_ids`` is not a vector, ``user_embedding``
            does not contain ``hidden_dim`` elements, or
            ``neighbor_embeddings`` is not ``(L, N, H)`` aligned with
            the history.

        Examples
        --------
        >>> m = FedPerGNNModel(num_items=5, hidden_dim=4, history_len=2, neighbor_len=1)
        >>> h = m.user_representation(torch.tensor([0, -1]), torch.zeros(4))
        >>> tuple(h.shape)
        (4,)
        """
        if history_item_ids.dim() != 1:
            raise ValueError(
                "history_item_ids must have shape (L,); "
                f"got {tuple(history_item_ids.shape)}"
            )
        u_base = user_embedding.reshape(-1)
        if u_base.numel() != self.hidden_dim:
            raise ValueError(
                f"user_embedding must have {self.hidden_dim} elements; "
                f"got shape {tuple(user_embedding.shape)}"
            )

        # -1 is an empty slot and lands on the padding row. Catalogue
        # id 0 lands on row 1.
        hist_item_emb = self.item_embeddings(history_item_ids + 1)  # (L, H)
        hist_item_emb = torch.sigmoid(self.history_item_proj(hist_item_emb))

        if neighbor_embeddings is not None:
            if neighbor_embeddings.dim() != 3:
                raise ValueError(
                    "neighbor_embeddings must have shape (L, N, H); "
                    f"got {tuple(neighbor_embeddings.shape)}"
                )
            if neighbor_embeddings.shape[0] != history_item_ids.shape[0]:
                raise ValueError(
                    "neighbor_embeddings history length "
                    f"{neighbor_embeddings.shape[0]} does not match "
                    f"history_item_ids length {history_item_ids.shape[0]}"
                )
            if neighbor_embeddings.shape[-1] != self.hidden_dim:
                raise ValueError(
                    "neighbor_embeddings width "
                    f"{neighbor_embeddings.shape[-1]} does not match "
                    f"hidden_dim {self.hidden_dim}"
                )
            neigh_proj = self.neighbor_proj(neighbor_embeddings)  # (L, N, H)
            n_neighbors = neighbor_embeddings.size(1)
            hist_expanded = hist_item_emb.unsqueeze(1).expand(-1, n_neighbors, -1)
            att_input = torch.cat([hist_expanded, neigh_proj], dim=-1)  # (L, N, 2H)
            neigh_scores = self.neighbor_att_mlp(att_input).squeeze(-1)  # (L, N)

            # All-zero neighbour rows are padding. Mirrors ComputeMasking.
            with torch.no_grad():
                mask = (neighbor_embeddings.abs().sum(dim=-1) > 0).to(neigh_scores.dtype)
            neigh_scores = neigh_scores + (1.0 - mask) * (-1e4)
            neigh_att = F.softmax(neigh_scores, dim=-1)  # (L, N)
            agg_neigh = (neigh_att.unsqueeze(-1) * neigh_proj).sum(dim=1)  # (L, H)
            hist_with_neigh = torch.cat([agg_neigh, hist_item_emb], dim=-1)
            hist_repr = self.history_with_neighbors(hist_with_neigh)  # (L, H)
        else:
            hist_repr = hist_item_emb  # (L, H)

        u_expanded = u_base.unsqueeze(0).expand(history_item_ids.size(0), -1)
        hist_att_input = torch.cat([u_expanded, hist_repr], dim=-1)  # (L, 2H)
        hist_scores = self.history_att_mlp(hist_att_input).squeeze(-1)  # (L,)

        # Empty slots are negative. Catalogue id 0 stays in the softmax.
        with torch.no_grad():
            hist_mask = (history_item_ids >= 0).to(hist_scores.dtype)
        hist_scores = hist_scores + (1.0 - hist_mask) * (-1e4)
        hist_att = F.softmax(hist_scores, dim=-1)  # (L,)
        u_hist = (hist_att.unsqueeze(-1) * hist_repr).sum(dim=0)  # (H,)
        return self.user_final_proj(torch.cat([u_hist, u_base], dim=-1))

    def score(self, user_repr: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Inner product of one user hidden state with target item embeddings.

        Parameters
        ----------
        user_repr : torch.Tensor
            Output of ``user_representation``, shape ``(H,)`` or ``(1, H)``.
        item_ids : torch.Tensor
            Catalogue item ids, shape ``(B,)``, dtype int64. Lookup
            adds 1, so id 0 reads embedding row 1, not the padding row.

        Returns
        -------
        torch.Tensor
            Unbounded scores, shape ``(B,)``. Higher is better.

        Raises
        ------
        ValueError
            If ``item_ids`` is not a vector, or ``user_repr`` does not
            contain ``hidden_dim`` elements.

        Examples
        --------
        >>> m = FedPerGNNModel(num_items=5, hidden_dim=4, history_len=2, neighbor_len=1)
        >>> h = m.user_representation(torch.tensor([1, 2]), torch.zeros(4))
        >>> m.score(h, torch.tensor([1, 3])).shape
        torch.Size([2])
        """
        if item_ids.dim() != 1:
            raise ValueError(
                "item_ids must have shape (B,); "
                f"got {tuple(item_ids.shape)}"
            )
        u = user_repr.reshape(-1)
        if u.numel() != self.hidden_dim:
            raise ValueError(
                f"user_repr must have {self.hidden_dim} elements; "
                f"got shape {tuple(user_repr.shape)}"
            )
        item_emb = self.item_embeddings(item_ids + 1)  # (B, H)
        return (item_emb * u).sum(dim=-1)

    def forward(
        self,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        user_embedding: torch.Tensor,
        neighbor_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score target items for one user.

        The user tower runs once. Each target id contributes only its
        item embedding to the inner product.

        Parameters
        ----------
        item_ids : torch.Tensor
            Catalogue item ids, shape ``(B,)``, dtype int64. Lookup
            adds 1, so id 0 reads embedding row 1, not the padding row.
        history_item_ids : torch.Tensor
            History ids, shape ``(L,)``, dtype int64. Negative entries
            are empty slots. Non-negative entries are catalogue ids.
        user_embedding : torch.Tensor
            Client-resident owner row, shape ``(H,)`` or ``(1, H)``.
        neighbor_embeddings : torch.Tensor or None, optional
            Neighbour rows, shape ``(L, N, H)``. ``None`` skips the
            neighbour GAT and uses projected history items only.

        Returns
        -------
        torch.Tensor
            Unbounded scores, shape ``(B,)``. Higher is better.

        Raises
        ------
        ValueError
            If a tensor rank or width does not match
            ``user_representation`` or ``score``.

        Examples
        --------
        >>> m = FedPerGNNModel(num_items=5, hidden_dim=4, history_len=2, neighbor_len=1)
        >>> m(torch.tensor([1, 2]), torch.tensor([1, 2]), torch.zeros(4)).shape
        torch.Size([2])
        """
        user_repr = self.user_representation(
            history_item_ids, user_embedding, neighbor_embeddings
        )
        return self.score(user_repr, item_ids)

