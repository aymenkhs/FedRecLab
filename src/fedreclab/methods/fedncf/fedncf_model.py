"""Neural Collaborative Filtering (NCF) model for federated learning.

Sources:
- Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.
- He et al., "Neural Collaborative Filtering", WWW 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fedreclab.federated.utils import flatten_grouped_state


class NeuralCollaborativeFiltering(nn.Module):
    """Server-resident NCF skeleton (item tables, MLP, output).

    User embedding tables are client-resident and are passed into ``forward``
    as tensors; they are not stored on this module.
    """

    SERVER_PARAM_GROUPS = ("item_embeddings", "shared_layers", "output_weights")

    def __init__(self, num_items, predictive_factor=32):
        super().__init__()

        self.gmf_item_embeddings = nn.Embedding(num_items, 2 * predictive_factor)
        self.mlp_item_embeddings = nn.Embedding(num_items, 2 * predictive_factor)

        # TODO: Allow for custom MLP and GMF layers with a custom activation function, and parameters.
        self.mlp = nn.Sequential(
            nn.Linear(4 * predictive_factor, 2 * predictive_factor),
            nn.ReLU(),
            nn.Linear(2 * predictive_factor, predictive_factor),
            nn.ReLU(),
            nn.Linear(predictive_factor, predictive_factor // 2),
            nn.ReLU(),
        )

        # Pre-output layers for each pathway (used for weight initialization only;
        # these do not participate in forward() (see join_output_weights).
        self.gmf_out = nn.Linear(2 * predictive_factor, 1)
        self.gmf_out.weight = nn.Parameter(torch.ones(1, 2 * predictive_factor))
        self.mlp_out = nn.Linear(predictive_factor // 2, 1)

        # Combined output: input dim = gmf (2*pf) + mlp (pf//2), derived
        # from gmf_out and mlp_out via join_output_weights()
        self.output_logits = nn.Linear(2 * predictive_factor + predictive_factor // 2, 1)

        self.model_blending = 0.5

        self.initialize_weights()
        self.join_output_weights()

    def initialize_weights(self):
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def join_output_weights(self):
        """Derive output_logits.weight from gmf_out and mlp_out (Eq. 13)."""
        W = torch.cat(
            (self.model_blending * self.gmf_out.weight.data,
             (1 - self.model_blending) * self.mlp_out.weight.data),
            dim=1,
        )
        self.output_logits.weight.data.copy_(W)

    def forward(self, x, gmf_user_table, mlp_user_table):
        """Return preference logits for each (user, item) pair.

        The paper's sigmoid sits in BCE-with-logits on the client, not
        here, so ranking scores stay unbounded. Order is identical to
        sigmoid probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 2)`` with columns ``[user_local_id, item_id]``.
        gmf_user_table, mlp_user_table : torch.Tensor
            Client-resident user tables, shape ``(n_local_users, dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch,)``.
        """
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = F.embedding(user_id, gmf_user_table) * self.gmf_item_embeddings(item_id)
        mlp_output = self.mlp(torch.cat([
            F.embedding(user_id, mlp_user_table),
            self.mlp_item_embeddings(item_id),
        ], dim=1))
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def get_server_state_groups(self):
        """Extract server-side parameters as three logical groups of cloned tensors.

        Inner keys are fully qualified ``named_parameters()`` names.
        """
        return {
            "item_embeddings": {
                "gmf_item_embeddings.weight": self.gmf_item_embeddings.weight.data.detach().cpu().clone(),
                "mlp_item_embeddings.weight": self.mlp_item_embeddings.weight.data.detach().cpu().clone(),
            },
            "shared_layers": {
                f"mlp.{k}": v.detach().cpu().clone() for k, v in self.mlp.state_dict().items()
            },
            "output_weights": {
                "gmf_out.weight": self.gmf_out.weight.data.detach().cpu().clone(),
                "gmf_out.bias": self.gmf_out.bias.data.detach().cpu().clone(),
                "mlp_out.weight": self.mlp_out.weight.data.detach().cpu().clone(),
                "mlp_out.bias": self.mlp_out.bias.data.detach().cpu().clone(),
                "output_logits.weight": self.output_logits.weight.data.detach().cpu().clone(),
                "output_logits.bias": self.output_logits.bias.data.detach().cpu().clone(),
            },
        }

    def load_server_weights(self, server_payload):
        """Load server parameters from the global payload (group_name -> dict of tensors).

        Accepts both torch.Tensor and numpy.ndarray values (post-aggregation values
        are numpy arrays; torch.as_tensor handles the conversion).
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
