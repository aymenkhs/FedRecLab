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
        Catalogue size after ``InteractionDataset`` remapping.
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

        # Item embedding table (server-side parameter)
        self.item_embeddings = nn.Embedding(num_items, hidden_dim)

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
        return self.item_embeddings.weight.data.detach().cpu().clone()

    def get_server_state(self) -> dict:
        """Every skeleton parameter and buffer, including the item table."""
        state = self.get_model_state()
        state[self.ITEM_EMBEDDING_NAME] = self.get_item_embeddings()
        return state

    def load_server_state(self, server_state: dict) -> None:
        """Load non-item server tensors from ``get_model_state()``."""
        self._load_named(server_state, exclude={self.ITEM_EMBEDDING_NAME})

    def load_item_embeddings(self, table) -> None:
        """Load the item table from a full ``(|I|, d)`` tensor or ndarray."""
        weight = self.item_embeddings.weight
        tensor = table if isinstance(table, torch.Tensor) else torch.as_tensor(table)
        weight.data.copy_(tensor.to(device=weight.device, dtype=weight.dtype))

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
            Padded history ids, shape ``(L,)``, dtype int64. Item id 0
            is treated as a padded history position.
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
        >>> h = m.user_representation(torch.tensor([1, 2]), torch.zeros(4))
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

        hist_item_emb = self.item_embeddings(history_item_ids)  # (L, H)
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

        # Padded history positions use item id 0.
        with torch.no_grad():
            hist_mask = (history_item_ids != 0).to(hist_scores.dtype)
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
            Target item ids, shape ``(B,)``, dtype int64.

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
        item_emb = self.item_embeddings(item_ids)  # (B, H)
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
            Target item ids, shape ``(B,)``, dtype int64.
        history_item_ids : torch.Tensor
            Padded history ids, shape ``(L,)``, dtype int64.
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

