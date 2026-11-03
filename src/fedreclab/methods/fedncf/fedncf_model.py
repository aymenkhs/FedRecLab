"""Neural Collaborative Filtering (NCF) model for federated learning.

Sources:
- Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.
- He et al., "Neural Collaborative Filtering", WWW 2017

The KBS paper is the fidelity target. Public GitHub copies that re-blend a
GMF/MLP head every round (He et al. Eq. 13) are independent
reimplementations, not the experiment code. Eq. 13 is one-shot pretrained
NeuMF initialisation; FedNCF trains the prediction layer as part of N.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedreclab.federated.utils import flatten_grouped_state


class NeuralCollaborativeFiltering(nn.Module):
    """Server-resident NCF skeleton (item tables, MLP, prediction layer).

    User embedding tables are client-resident and are passed into ``forward``
    as tensors; they are not stored on this module. Embedding width is
    ``2 * predictive_factor``.

    The prediction layer is a single ``Linear`` over the concatenated GMF
    product and MLP tower. It is Xavier-initialised once (paper §4.2) and
    exchanged in ``output_weights``. See the module docstring for why dummy
    GMF/MLP output heads are omitted.

    Parameters
    ----------
    num_items : int
        Number of items after ``InteractionDataset`` remapping.
    predictive_factor : int, optional
        Controls embedding width (``2 * predictive_factor``) and the MLP
        tower. Default is 32. The paper's D=12 is
        ``predictive_factor=6``, but the MLP shape still cannot match
        ``24 → 48 → 24 → 12 → 6``.

    Examples
    --------
    >>> m = NeuralCollaborativeFiltering(num_items=10, predictive_factor=8)
    >>> sorted(m.get_server_state_groups()["output_weights"])
    ['output_logits.bias', 'output_logits.weight']
    """

    SERVER_PARAM_GROUPS = ("item_embeddings", "shared_layers", "output_weights")

    def __init__(self, num_items, predictive_factor=32):
        super().__init__()

        self.gmf_item_embeddings = nn.Embedding(num_items, 2 * predictive_factor)
        self.mlp_item_embeddings = nn.Embedding(num_items, 2 * predictive_factor)

        # TODO(NCF-03): paper NeuMF is D=12 with MLP 24→48→24→12→6; this
        # tower cannot express that until MLP/GMF widths are configurable.
        self.mlp = nn.Sequential(
            nn.Linear(4 * predictive_factor, 2 * predictive_factor),
            nn.ReLU(),
            nn.Linear(2 * predictive_factor, predictive_factor),
            nn.ReLU(),
            nn.Linear(predictive_factor, predictive_factor // 2),
            nn.ReLU(),
        )

        # Concatenated GMF product (2*pf) and MLP tower (pf//2).
        self.output_logits = nn.Linear(2 * predictive_factor + predictive_factor // 2, 1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialise embeddings from N(0, 0.01) and neural weights with Xavier.

        Item tables are the profile I, not N; they follow the usual NCF
        Gaussian. MLP and ``output_logits`` weights use Xavier uniform
        (paper §4.2). ``output_logits.bias`` is zeroed; MLP biases keep
        the ``nn.Linear`` default.

        Returns
        -------
        None
        """
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_logits.weight)
        nn.init.zeros_(self.output_logits.bias)

    def forward(self, x, gmf_user_table, mlp_user_table):
        """Return a score for each (user, item) pair.

        Under BCE these are logits: the paper's sigmoid sits in
        BCE-with-logits on the client, and ranking order matches sigmoid
        probabilities. Under MSE they are rating regressions.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 2)`` with columns ``[user_local_id, item_id]``.
        gmf_user_table, mlp_user_table : torch.Tensor
            Client-resident user tables, shape ``(n_local_users, dim)``.

        Returns
        -------
        torch.Tensor
            Scores of shape ``(batch,)`` (logits under BCE).
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
        ``output_weights`` holds only the prediction layer (part of N).

        Returns
        -------
        dict of str to dict of str to torch.Tensor
            ``item_embeddings``, ``shared_layers``, and ``output_weights``.

        Examples
        --------
        >>> m = NeuralCollaborativeFiltering(num_items=4, predictive_factor=8)
        >>> sorted(m.get_server_state_groups()["output_weights"])
        ['output_logits.bias', 'output_logits.weight']
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
                "output_logits.weight": self.output_logits.weight.data.detach().cpu().clone(),
                "output_logits.bias": self.output_logits.bias.data.detach().cpu().clone(),
            },
        }

    def load_server_weights(self, server_payload):
        """Load server parameters from the global payload (group_name -> dict of tensors).

        Accepts both torch.Tensor and numpy.ndarray values (post-aggregation values
        are numpy arrays; torch.as_tensor handles the conversion).

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
