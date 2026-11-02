"""General Matrix Factorisation (GMF) skeleton for FedFast.

Sources:
- Muhammad et al., "FedFast: Going Beyond Average for Faster Training of
  Federated Recommender Systems", KDD 2020.
- He et al., "Neural Collaborative Filtering", WWW 2017 (GMF).

The server holds the item embedding table and the linear prediction head.
User embeddings are client-resident and are passed into ``forward``.
"""

from __future__ import annotations

import torch
from torch import nn

from src.fedreclab.federated.utils import flatten_grouped_state


class GeneralMatrixFactorization(nn.Module):
    """Server-resident GMF item table and linear output head.

    Score is ``Linear(p_u ⊙ q_i)`` with a bias on the linear layer
    (paper Fig. 2: element-wise product then a fully-connected layer with
    linear activation).

    Parameters
    ----------
    num_items : int
        Number of items after ``InteractionDataset`` remapping.
    embedding_dim : int, optional
        Width of user and item embeddings. Default is 10 (FedFast
        experimental setting).

    Examples
    --------
    >>> m = GeneralMatrixFactorization(num_items=5, embedding_dim=4)
    >>> sorted(m.get_server_state_groups())
    ['item_embeddings', 'output_weights']
    """

    SERVER_PARAM_GROUPS = ("item_embeddings", "output_weights")

    def __init__(self, num_items: int, embedding_dim: int = 10):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, self.embedding_dim)
        self.output_logits = nn.Linear(self.embedding_dim, 1)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialise embeddings from N(0, 0.01) and the head with Xavier.

        Returns
        -------
        None
        """
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.xavier_uniform_(self.output_logits.weight)
        nn.init.zeros_(self.output_logits.bias)

    def forward(
        self,
        user_item_ids: torch.Tensor,
        user_table: torch.Tensor,
    ) -> torch.Tensor:
        """Return a score for each (user, item) pair.

        Under BCE these are logits. Ranking order matches sigmoid
        probabilities.

        Parameters
        ----------
        user_item_ids : torch.Tensor
            Long tensor of shape ``(batch, 2)`` with local user index in
            column 0 and item id in column 1.
        user_table : torch.Tensor
            Client-resident user embeddings, shape
            ``(num_local_users, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Scores of shape ``(batch,)``.
        """
        local_uids = user_item_ids[:, 0]
        item_ids = user_item_ids[:, 1]
        user_vec = user_table[local_uids]
        item_vec = self.item_embeddings(item_ids)
        return self.output_logits(user_vec * item_vec).squeeze(-1)

    def get_server_state_groups(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return item and output groups as CPU clones.

        Returns
        -------
        dict of str to dict of str to torch.Tensor
            ``item_embeddings`` and ``output_weights``.

        Examples
        --------
        >>> m = GeneralMatrixFactorization(3, 2)
        >>> m.get_server_state_groups()["output_weights"]["output_logits.bias"].shape
        torch.Size([1])
        """
        return {
            "item_embeddings": {
                "item_embeddings.weight": (
                    self.item_embeddings.weight.data.detach().cpu().clone()
                ),
            },
            "output_weights": {
                "output_logits.weight": (
                    self.output_logits.weight.data.detach().cpu().clone()
                ),
                "output_logits.bias": (
                    self.output_logits.bias.data.detach().cpu().clone()
                ),
            },
        }

    def load_server_weights(self, server_payload: dict) -> None:
        """Copy server groups onto this skeleton.

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
                f"missing_from_payload={sorted(missing)} "
                f"extra_in_payload={sorted(extra)}"
            )
        for name, tensor in named.items():
            tensor.data.copy_(torch.as_tensor(flat[name], device=tensor.device))
