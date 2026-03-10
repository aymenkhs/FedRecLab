"""Neural Collaborative Filtering (NCF) model for federated learning.

Sources:
- Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.
- He et al., "Neural Collaborative Filtering", WWW 2017
"""

import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """Full NCF model used on the client side during local training.

    Combines two pathways:
      - GMF: element-wise product of user/item embeddings
      - MLP: concatenated user/item embeddings through a deep network

    The final prediction uses a learned combination of both pathways,
    re-initialized via join_output_weights() before each training round.
    """

    SERVER_PARAM_GROUPS = ("item_embeddings", "shared_layers", "output_weights")

    def __init__(self, num_users, num_items, predictive_factor=32):
        super().__init__()

        # GMF pathway embeddings
        self.gmf_user_embeddings = nn.Embedding(num_users, 2 * predictive_factor)
        self.gmf_item_embeddings = nn.Embedding(num_items, 2 * predictive_factor)

        # MLP pathway embeddings
        self.mlp_user_embeddings = nn.Embedding(num_users, 2 * predictive_factor)
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
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1)

    def join_output_weights(self):
        """Derive output_logits.weight from gmf_out and mlp_out (Eq. 13).

        Uses in-place copy so the Parameter *object* stays the same across
        calls.  This keeps a persistent optimizer's reference valid while
        giving the same gradient behaviour: only output_logits receives
        gradients; gmf_out/mlp_out are not in the computation graph.
        """
        W = torch.cat(
            (self.model_blending * self.gmf_out.weight.data,
             (1 - self.model_blending) * self.mlp_out.weight.data),
            dim=1,
        )
        self.output_logits.weight.data.copy_(W)

    def forward(self, x):
        """Forward pass. x has shape (batch, 2) with columns [user_local_id, item_id]."""
        user_id, item_id = x[:, 0], x[:, 1]
        gmf_product = self._gmf_forward(user_id, item_id)
        mlp_output = self._mlp_forward(user_id, item_id)
        return self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1)

    def _gmf_forward(self, user_id, item_id):
        return torch.mul(
            self.gmf_user_embeddings(user_id),
            self.gmf_item_embeddings(item_id),
        )

    def _mlp_forward(self, user_id, item_id):
        return self.mlp(torch.cat([
            self.mlp_user_embeddings(user_id),
            self.mlp_item_embeddings(item_id),
        ], dim=1))

    def get_server_state_groups(self):
        """Extract server-side parameters as three logical groups of cloned tensors."""
        return {
            "item_embeddings": {
                "gmf_item_embeddings.weight": self.gmf_item_embeddings.weight.data.clone(),
                "mlp_item_embeddings.weight": self.mlp_item_embeddings.weight.data.clone(),
            },
            "shared_layers": {
                k: v.clone() for k, v in self.mlp.state_dict().items()
            },
            "output_weights": {
                "gmf_out.weight": self.gmf_out.weight.data.clone(),
                "gmf_out.bias": self.gmf_out.bias.data.clone(),
                "mlp_out.weight": self.mlp_out.weight.data.clone(),
                "mlp_out.bias": self.mlp_out.bias.data.clone(),
                "output_logits.weight": self.output_logits.weight.data.clone(),
                "output_logits.bias": self.output_logits.bias.data.clone(),
            },
        }

    def get_user_state(self):
        """Extract client-side (user) embedding weights as cloned tensors."""
        return {
            "gmf_user_embeddings.weight": self.gmf_user_embeddings.weight.data.clone(),
            "mlp_user_embeddings.weight": self.mlp_user_embeddings.weight.data.clone(),
        }

    def load_server_weights(self, server_payload):
        """Load server parameters from the global payload (group_name -> dict of tensors).

        Accepts both torch.Tensor and numpy.ndarray values (post-aggregation values
        are numpy arrays; torch.as_tensor handles the conversion).
        """
        item_emb = server_payload["item_embeddings"]
        self.gmf_item_embeddings.weight.data.copy_(torch.as_tensor(item_emb["gmf_item_embeddings.weight"]))
        self.mlp_item_embeddings.weight.data.copy_(torch.as_tensor(item_emb["mlp_item_embeddings.weight"]))

        shared = server_payload["shared_layers"]
        self.mlp.load_state_dict({k: torch.as_tensor(v) for k, v in shared.items()})

        out = server_payload["output_weights"]
        self.gmf_out.weight.data.copy_(torch.as_tensor(out["gmf_out.weight"]))
        self.gmf_out.bias.data.copy_(torch.as_tensor(out["gmf_out.bias"]))
        self.mlp_out.weight.data.copy_(torch.as_tensor(out["mlp_out.weight"]))
        self.mlp_out.bias.data.copy_(torch.as_tensor(out["mlp_out.bias"]))
        self.output_logits.weight.data.copy_(torch.as_tensor(out["output_logits.weight"]))
        self.output_logits.bias.data.copy_(torch.as_tensor(out["output_logits.bias"]))

    def load_user_weights(self, user_state):
        """Load client-side user embedding weights from a dict of tensors."""
        self.gmf_user_embeddings.weight.data.copy_(torch.as_tensor(user_state["gmf_user_embeddings.weight"]))
        self.mlp_user_embeddings.weight.data.copy_(torch.as_tensor(user_state["mlp_user_embeddings.weight"]))
