import torch
import torch.nn as nn
import torch.nn.functional as F


class FedPerGNNModel(nn.Module):

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

        # Client-personalized user embedding (one vector per client/user).
        # TODO: Allow for cross-silo scenarios or for multiple users per client.
        self.user_embedding = nn.Embedding(1, hidden_dim)

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

        # Final predictor over [user_repr, target_item_emb]
        self.output_layer = nn.Linear(2 * hidden_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_server_state(self) -> dict:
        """
        Return a dict of tensors that represent server-side parameters.

        By default this excludes the client-personalized user embedding.
        """
        state: dict[str, torch.Tensor] = {
            "item_embeddings.weight": self.item_embeddings.weight.data.clone(),
            "history_item_proj.weight": self.history_item_proj.weight.data.clone(),
            "history_item_proj.bias": self.history_item_proj.bias.data.clone(),
            "neighbor_proj.weight": self.neighbor_proj.weight.data.clone(),
            "neighbor_proj.bias": self.neighbor_proj.bias.data.clone(),
            "user_final_proj.weight": self.user_final_proj.weight.data.clone(),
            "user_final_proj.bias": self.user_final_proj.bias.data.clone(),
            "output_layer.weight": self.output_layer.weight.data.clone(),
            "output_layer.bias": self.output_layer.bias.data.clone(),
        }

        # Flatten MLP state dicts so that aggregation sees only tensor leaves.
        for k, v in self.neighbor_att_mlp.state_dict().items():
            state[f"neighbor_att_mlp.{k}"] = v.data.clone()
        for k, v in self.history_att_mlp.state_dict().items():
            state[f"history_att_mlp.{k}"] = v.data.clone()

        return state

    def load_server_state(self, server_state: dict) -> None:
        """
        Load server-side parameters from a dict produced by get_server_state().

        Accepts both torch.Tensor and numpy.ndarray values.
        """
        def as_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

        self.item_embeddings.weight.data.copy_(as_tensor(server_state["item_embeddings.weight"]))
        self.history_item_proj.weight.data.copy_(as_tensor(server_state["history_item_proj.weight"]))
        self.history_item_proj.bias.data.copy_(as_tensor(server_state["history_item_proj.bias"]))
        self.neighbor_proj.weight.data.copy_(as_tensor(server_state["neighbor_proj.weight"]))
        self.neighbor_proj.bias.data.copy_(as_tensor(server_state["neighbor_proj.bias"]))

        # Neighbor attention MLP
        neighbor_prefix = "neighbor_att_mlp."
        neighbor_att_state = {
            k[len(neighbor_prefix) :]: as_tensor(v)
            for k, v in server_state.items()
            if k.startswith(neighbor_prefix)
        }
        if neighbor_att_state:
            self.neighbor_att_mlp.load_state_dict(neighbor_att_state)

        # History attention MLP
        history_prefix = "history_att_mlp."
        history_att_state = {
            k[len(history_prefix) :]: as_tensor(v)
            for k, v in server_state.items()
            if k.startswith(history_prefix)
        }
        if history_att_state:
            self.history_att_mlp.load_state_dict(history_att_state)

        self.user_final_proj.weight.data.copy_(as_tensor(server_state["user_final_proj.weight"]))
        self.user_final_proj.bias.data.copy_(as_tensor(server_state["user_final_proj.bias"]))
        self.output_layer.weight.data.copy_(as_tensor(server_state["output_layer.weight"]))
        self.output_layer.bias.data.copy_(as_tensor(server_state["output_layer.bias"]))

    def get_user_state(self) -> dict:
        """Return client-local user embedding state."""
        return {
            "user_embedding.weight": self.user_embedding.weight.data.clone(),
        }

    def load_user_state(self, user_state: dict) -> None:
        """Load client-local user embedding state."""
        w = user_state.get("user_embedding.weight", None)
        if w is not None:
            self.user_embedding.weight.data.copy_(
                w if isinstance(w, torch.Tensor) else torch.as_tensor(w)
            )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        neighbor_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            user_ids: (B,) int64 – kept for API parity, but the model uses a
                single local user embedding. All examples in the batch are
                assumed to belong to the same client/user.
            item_ids: (B,) int64
            history_item_ids: (B, L) int64
            neighbor_embeddings: (B, L, N, H) float32, or None. When None,
                neighbors are ignored and only item history drives the user
                representation.
        """
        device = item_ids.device
        batch_size = item_ids.size(0)

        # Target items
        item_emb = self.item_embeddings(item_ids)  # (B, H)

        # History items
        hist_item_emb = self.item_embeddings(history_item_ids)  # (B, L, H)
        hist_item_emb = torch.sigmoid(self.history_item_proj(hist_item_emb))  # (B, L, H)

        # Base user embedding: same vector repeated for all examples in batch.
        # Shape (1, H) -> (B, H)
        u_base = self.user_embedding.weight[0].unsqueeze(0).expand(batch_size, -1)

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

        out = self.output_layer(torch.cat([u_final, item_emb], dim=-1))  # (B, 1)
        out = torch.sigmoid(out).squeeze(-1)  # (B,)
        return out

