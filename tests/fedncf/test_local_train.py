"""FedNCF local_train with persist-all Adam should match a single nn.Module.

If every parameter (item tables and user tables) keeps its Adam moments,
training through the federated client must equal training a module that
owns both sets of tables and runs Adam on ``module.parameters()``.
"""

import numpy as np
import pytest
import torch
from torch import nn

from src.fedreclab.federated.base_client import ClientTrainingRequest
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

    request.seed = 0
    new_resp = client.local_train(request)

    backbone = NeuralCollaborativeFiltering(num_items=6, predictive_factor=8)
    backbone.load_server_weights(payload0)
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
    rng = np.random.default_rng([request.seed, request.client_id, request.current_round])
    samples_x, samples_y = client._build_epoch_samples(
        pos_local, pos_items, pos_ratings, user_pos, rng
    )
    perm = rng.permutation(len(samples_x))
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
        loss = client._compute_loss(preds, by)
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
    rows = torch.as_tensor(new_items["rows"], dtype=torch.long)
    np.testing.assert_array_equal(np.sort(rows.numpy()), np.array([1, 2]))
    for key, uploaded in new_items["values"].items():
        torch.testing.assert_close(
            torch.as_tensor(uploaded),
            ref_items[key][rows],
            atol=1e-6,
            rtol=1e-6,
        )


def test_prediction_layer_is_trained_and_not_reset_between_rounds(
    fedncf_client_and_request,
):
    """output_logits is part of N: trained locally, then kept as the next payload."""
    client, request = fedncf_client_and_request()
    w0 = request.global_payload["output_weights"]["output_logits.weight"].clone()
    resp1 = client.local_train(request)
    w1 = torch.as_tensor(
        resp1.updates_packages["output_weights"].param_value["output_logits.weight"]
    ).clone()
    assert not torch.equal(w0, w1)

    payload2 = _clone_groups(request.global_payload)
    payload2["output_weights"] = {
        k: torch.as_tensor(v).clone()
        for k, v in resp1.updates_packages["output_weights"].param_value.items()
    }
    client.lr = 0.0
    request2 = ClientTrainingRequest(
        client_id=0,
        current_round=1,
        users_in_client=[0],
        client_state=resp1.updated_client_state,
        global_payload=payload2,
        training_data=request.training_data,
        param_ios=request.param_ios,
        seed=request.seed,
    )
    resp2 = client.local_train(request2)
    w2 = torch.as_tensor(
        resp2.updates_packages["output_weights"].param_value["output_logits.weight"]
    )
    torch.testing.assert_close(w2, w1, atol=1e-7, rtol=0.0)


def test_scoring_does_not_reset_the_prediction_layer(fedncf_client_and_request):
    client, request = fedncf_client_and_request()
    resp = client.local_train(request)
    payload = _clone_groups(request.global_payload)
    payload["output_weights"] = {
        k: torch.as_tensor(v).clone()
        for k, v in resp.updates_packages["output_weights"].param_value.items()
    }
    w_trained = payload["output_weights"]["output_logits.weight"].clone()
    client.compute_scores(
        0,
        {0: np.array([1, 2, 3])},
        payload,
        resp.updated_client_state,
        None,
    )
    torch.testing.assert_close(
        client._skeleton.output_logits.weight.detach().cpu(),
        w_trained,
        atol=0.0,
        rtol=0.0,
    )
