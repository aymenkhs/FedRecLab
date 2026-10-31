"""FedNCF local_train with persist-all Adam should match a single nn.Module.

If every parameter (item tables and user tables) keeps its Adam moments,
training through the federated client must equal training a module that
owns both sets of tables and runs Adam on ``module.parameters()``.
"""

import numpy as np
import pytest
import torch
from torch import nn

from src.fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering


class _ModuleWithUserTables(nn.Module):
    """Reference: user tables live on the module, one Adam over all parameters."""

    def __init__(self, backbone, gmf_user, mlp_user):
        super().__init__()
        self.backbone = backbone
        self.gmf_user_embeddings = nn.Parameter(gmf_user.clone())
        self.mlp_user_embeddings = nn.Parameter(mlp_user.clone())

    def forward(self, x):
        return self.backbone(x, self.gmf_user_embeddings, self.mlp_user_embeddings)


def _clone_groups(groups):
    return {g: {k: v.clone() for k, v in tensors.items()} for g, tensors in groups.items()}


def test_persist_all_moments_matches_a_module_that_owns_user_tables(
    fedncf_client_and_request,
):
    client, request = fedncf_client_and_request(optimizer_state_scope="all")
    user0 = {
        k: v.clone()
        for k, v in request.client_state.local_parameters["user_embeddings"].items()
    }
    payload0 = _clone_groups(request.global_payload)

    np.random.seed(0)
    torch.manual_seed(0)
    new_resp = client.local_train(request)

    backbone = NeuralCollaborativeFiltering(num_items=6, predictive_factor=8)
    backbone.load_server_weights(payload0)
    backbone.join_output_weights()
    owned = _ModuleWithUserTables(
        backbone,
        user0["gmf_user_embeddings.weight"],
        user0["mlp_user_embeddings.weight"],
    )
    optimizer = torch.optim.Adam(owned.parameters(), lr=client.lr)

    pos_local = np.array([0, 0])
    pos_items = np.array([1, 2])
    pos_ratings = np.array([1.0, 0.5], dtype=np.float32)
    user_pos = {0: {1, 2}}
    np.random.seed(0)
    torch.manual_seed(0)
    samples_x, samples_y = client._build_epoch_samples(
        pos_local, pos_items, pos_ratings, user_pos
    )
    perm = np.random.permutation(len(samples_x))
    samples_x, samples_y = samples_x[perm], samples_y[perm]

    owned.train()
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, len(samples_x), client.batch_size):
        bx = torch.tensor(samples_x[start : start + client.batch_size], dtype=torch.long)
        by = torch.tensor(
            samples_y[start : start + client.batch_size], dtype=torch.float32
        )
        preds = owned(bx)
        loss = torch.nn.functional.mse_loss(preds, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    ref_losses = [epoch_loss / n_batches]

    assert new_resp.training_losses == pytest.approx(ref_losses, rel=1e-6, abs=1e-6)
    new_user = new_resp.updated_client_state.local_parameters["user_embeddings"]
    torch.testing.assert_close(
        new_user["gmf_user_embeddings.weight"],
        owned.gmf_user_embeddings.detach().cpu(),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        new_user["mlp_user_embeddings.weight"],
        owned.mlp_user_embeddings.detach().cpu(),
        atol=1e-6,
        rtol=1e-6,
    )
    new_items = new_resp.updates_packages["item_embeddings"].param_value
    ref_items = owned.backbone.get_server_state_groups()["item_embeddings"]
    for key in new_items:
        torch.testing.assert_close(new_items[key], ref_items[key], atol=1e-6, rtol=1e-6)
