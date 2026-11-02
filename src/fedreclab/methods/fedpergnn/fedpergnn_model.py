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
    not depend on the target item. The score is the inner product of that
    state and the target item embedding (SI predictor). There is no
    concat-linear output layer and no output sigmoid: ranking order of
    an inner product is unchanged by a monotone map, and rating
    regression needs an unbounded score.

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

    def forward(
        self,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        user_embedding: torch.Tensor,
        neighbor_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score each target item against this user's GAT hidden state.

        History and neighbour tensors may be repeated across the batch;
        they must not depend on the target id. The returned scores are
        unbounded inner products (higher is better).

        Parameters
        ----------
        item_ids : torch.Tensor
            Target item ids, shape ``(B,)``, dtype int64.
        history_item_ids : torch.Tensor
            Padded history ids, shape ``(B, L)``, dtype int64.
        user_embedding : torch.Tensor
            Client-resident owner row, shape ``(H,)`` or ``(1, H)``.
        neighbor_embeddings : torch.Tensor or None, optional
            Neighbour rows, shape ``(B, L, N, H)``. ``None`` skips the
            neighbour GAT and uses projected history items only.

        Returns
        -------
        torch.Tensor
            Scores of shape ``(B,)``.

        Examples
        --------
        >>> m = FedPerGNNModel(num_items=5, hidden_dim=4, history_len=2, neighbor_len=1)
        >>> items = torch.tensor([1, 2])
        >>> hist = torch.tensor([[1, 2], [1, 2]])
        >>> m(items, hist, torch.zeros(4)).shape
        torch.Size([2])
        """
        batch_size = item_ids.size(0)

        # Target items
        item_emb = self.item_embeddings(item_ids)  # (B, H)

        # History items
        hist_item_emb = self.item_embeddings(history_item_ids)  # (B, L, H)
        hist_item_emb = torch.sigmoid(self.history_item_proj(hist_item_emb))  # (B, L, H)

        u_base = user_embedding.reshape(-1).unsqueeze(0).expand(batch_size, -1)

        if neighbor_embeddings is not None:
            # neighbor_embeddings: (B, L, N, H)
            # Project neighbors
            neigh_proj = self.neighbor_proj(neighbor_embeddings)  # (B, L, N, H)

            # Expand hist_item_emb to match neighbor dimension
            hist_expanded = hist_item_emb.unsqueeze(2).expand(-1, -1, self.neighbor_len, -1)  # (B, L, N, H)

            # Concatenate along last dim -> (B, L, N, 2H)
            att_input = torch.cat([hist_expanded, neigh_proj], dim=-1)

            # Compute unnormalized attention scores over neighbors
            neigh_scores = self.neighbor_att_mlp(att_input).squeeze(-1)  # (B, L, N)

            # Mask out neighbors whose embeddings are all zeros (padding).
            # This mirrors the original ComputeMasking(+(-99)) behaviour.
            with torch.no_grad():
                mask = (neighbor_embeddings.abs().sum(dim=-1) > 0).to(neigh_scores.dtype)  # (B, L, N)
            neigh_scores = neigh_scores + (1.0 - mask) * (-1e4)

            neigh_att = F.softmax(neigh_scores, dim=-1)  # (B, L, N)

            # Weighted sum over neighbors
            neigh_att_exp = neigh_att.unsqueeze(-1)  # (B, L, N, 1)
            agg_neigh = (neigh_att_exp * neigh_proj).sum(dim=2)  # (B, L, H)

            # Combine aggregated neighbor embedding with history item embedding
            hist_with_neigh = torch.cat([agg_neigh, hist_item_emb], dim=-1)  # (B, L, 2H)
            hist_repr = self.history_with_neighbors(hist_with_neigh)  # (B, L, H)
        else:
            # No neighbors: just use projected history item embeddings
            hist_repr = hist_item_emb  # (B, L, H)

        # Repeat user embedding over history length
        u_expanded = u_base.unsqueeze(1).expand(-1, self.history_len, -1)  # (B, L, H)

        hist_att_input = torch.cat([u_expanded, hist_repr], dim=-1)  # (B, L, 2H)
        hist_scores = self.history_att_mlp(hist_att_input).squeeze(-1)  # (B, L)

        # Mask padded history positions where item id == 0
        with torch.no_grad():
            hist_mask = (history_item_ids != 0).to(hist_scores.dtype)  # (B, L)
        hist_scores = hist_scores + (1.0 - hist_mask) * (-1e4)

        hist_att = F.softmax(hist_scores, dim=-1)  # (B, L)

        # Aggregate history representations
        hist_att_exp = hist_att.unsqueeze(-1)  # (B, L, 1)
        u_hist = (hist_att_exp * hist_repr).sum(dim=1)  # (B, H)

        # Combine with base user embedding
        u_final = self.user_final_proj(torch.cat([u_hist, u_base], dim=-1))  # (B, H)

        # SI predictor ŷ = h_u · e_i. A linear layer on the concat is
        # additive: ranking inside a user ignores h_u.
        return (u_final * item_emb).sum(dim=-1)

