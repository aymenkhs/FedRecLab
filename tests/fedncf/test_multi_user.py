"""FedNCF clients that own more than one user index embeddings by local row.

Global user ids are not row indices. These tests use ids 5 and 2 so a path
that embeds the global id (or always uses row 0) cannot pass.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering

# users_in_client order is the row order: 5 -> 0, 2 -> 1.
_USERS = [5, 2]
_TWO_USERS = pd.DataFrame(
    {
        "user_id": [5, 2, 5],
        "item_id": [1, 3, 4],
        "rating": [1.0, 0.5, 1.0],
    }
)
_ONLY_USER_5 = _TWO_USERS[_TWO_USERS["user_id"] == 5].reset_index(drop=True)


class _ModuleWithUserTables(nn.Module):
    """Reference: both user rows live on one module, one Adam over all parameters."""

    def __init__(self, backbone, gmf_user, mlp_user):
        super().__init__()
        self.backbone = backbone
        self.gmf_user_embeddings = nn.Parameter(gmf_user.clone())
        self.mlp_user_embeddings = nn.Parameter(mlp_user.clone())

    def forward(self, x):
        return self.backbone(x, self.gmf_user_embeddings, self.mlp_user_embeddings)


def _user_row(local_parameters, user_id, table):
    row = local_parameters["user_id_to_local"][user_id]
    return local_parameters["user_embeddings"][table][row]


def test_local_rows_follow_users_in_client_order(fedncf_client_and_request):
    """The embedding table has one row per local user, not one per global id."""
    _, request = fedncf_client_and_request(
        training_data=_TWO_USERS, users_in_client=_USERS
    )
    local = request.client_state.local_parameters
    assert local["user_id_to_local"] == {5: 0, 2: 1}
    for name in ("gmf_user_embeddings.weight", "mlp_user_embeddings.weight"):
        assert local["user_embeddings"][name].shape == (2, 16)


def test_multi_user_train_matches_local_row_indices(fedncf_client_and_request):
    """One step on both users equals a module indexed by user_id_to_local."""
    client, request = fedncf_client_and_request(
        training_data=_TWO_USERS, users_in_client=_USERS, seed=0
    )
    user0 = {
        k: v.clone()
        for k, v in request.client_state.local_parameters["user_embeddings"].items()
    }
    payload0 = {
        g: {k: v.clone() for k, v in tensors.items()}
        for g, tensors in request.global_payload.items()
    }
    mapping = request.client_state.local_parameters["user_id_to_local"]

    resp = client.local_train(request)

    backbone = NeuralCollaborativeFiltering(num_items=6, predictive_factor=8)
    backbone.load_server_weights(payload0)
    owned = _ModuleWithUserTables(
        backbone,
        user0["gmf_user_embeddings.weight"],
        user0["mlp_user_embeddings.weight"],
    )
    optimizer = torch.optim.Adam(owned.parameters(), lr=client.lr)
    data = request.training_data
    pos_local = np.array([mapping[int(u)] for u in data["user_id"].to_numpy()])
    pos_items = data["item_id"].to_numpy()
    pos_ratings = data["rating"].to_numpy(dtype=np.float32)
    user_pos = {}
    for local_uid, item_id in zip(pos_local, pos_items):
        user_pos.setdefault(int(local_uid), set()).add(int(item_id))
    rng = np.random.default_rng(
        [request.seed, request.client_id, request.current_round]
    )
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
        loss = client._compute_loss(owned(bx), by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    assert resp.training_losses == pytest.approx(
        [epoch_loss / n_batches], rel=1e-6, abs=1e-6
    )
    assert resp.num_training_examples == len(_TWO_USERS)
    updated = resp.updated_client_state.local_parameters
    assert updated["user_id_to_local"] == {5: 0, 2: 1}
    torch.testing.assert_close(
        updated["user_embeddings"]["gmf_user_embeddings.weight"],
        owned.gmf_user_embeddings.detach().cpu(),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        updated["user_embeddings"]["mlp_user_embeddings.weight"],
        owned.mlp_user_embeddings.detach().cpu(),
        atol=1e-6,
        rtol=1e-6,
    )
    for uid in _USERS:
        assert not torch.equal(
            _user_row(updated, uid, "gmf_user_embeddings.weight"),
            _user_row(
                {"user_id_to_local": mapping, "user_embeddings": user0},
                uid,
                "gmf_user_embeddings.weight",
            ),
        )

    uploaded = resp.updates_packages["item_embeddings"].param_value
    rows = np.asarray(uploaded["rows"])
    np.testing.assert_array_equal(np.sort(rows), np.array([1, 3, 4]))
    ref_items = owned.backbone.get_server_state_groups()["item_embeddings"]
    row_index = torch.as_tensor(rows, dtype=torch.long)
    for key, values in uploaded["values"].items():
        torch.testing.assert_close(
            torch.as_tensor(values), ref_items[key][row_index], atol=1e-6, rtol=1e-6
        )


def test_idle_user_row_stays_at_its_initial_value(fedncf_client_and_request):
    """A user with no rows in the batch does not lend its embedding to the other."""
    client, request = fedncf_client_and_request(
        training_data=_ONLY_USER_5, users_in_client=_USERS, seed=0
    )
    mapping = request.client_state.local_parameters["user_id_to_local"]
    before = {
        k: v.clone()
        for k, v in request.client_state.local_parameters["user_embeddings"].items()
    }
    resp = client.local_train(request)
    updated = resp.updated_client_state.local_parameters["user_embeddings"]
    idle = mapping[2]
    active = mapping[5]
    for name in ("gmf_user_embeddings.weight", "mlp_user_embeddings.weight"):
        torch.testing.assert_close(
            updated[name][idle], before[name][idle], atol=0.0, rtol=0.0
        )
    assert not torch.equal(
        updated["gmf_user_embeddings.weight"][active],
        before["gmf_user_embeddings.weight"][active],
    )
    rows = np.asarray(resp.updates_packages["item_embeddings"].param_value["rows"])
    np.testing.assert_array_equal(np.sort(rows), np.array([1, 4]))
    assert resp.num_training_examples == len(_ONLY_USER_5)


def test_scores_use_the_local_row_for_each_global_id(fedncf_client_and_request):
    """Scoring user 5 reads row 0 and user 2 reads row 1."""
    client, request = fedncf_client_and_request(
        training_data=_TWO_USERS, users_in_client=_USERS
    )
    local = request.client_state.local_parameters
    gmf = local["user_embeddings"]["gmf_user_embeddings.weight"]
    mlp = local["user_embeddings"]["mlp_user_embeddings.weight"]
    gmf.zero_()
    mlp.zero_()
    gmf[0].fill_(1.0)
    gmf[1].fill_(-1.0)
    items = request.global_payload["item_embeddings"]["gmf_item_embeddings.weight"]
    items[1].fill_(1.0)

    candidates = np.array([1, 3, 4])
    scores = client.compute_scores(
        0,
        {5: candidates, 2: candidates},
        request.global_payload,
        request.client_state,
        {},
    )

    model = client._skeleton
    model.load_server_weights(request.global_payload)
    model.eval()

    def _row_scores(local_uid):
        user_ids = torch.full((len(candidates),), local_uid, dtype=torch.long)
        item_ids = torch.tensor(candidates, dtype=torch.long)
        x = torch.stack([user_ids, item_ids], dim=1)
        with torch.no_grad():
            return model(x, gmf, mlp).numpy()

    row0 = _row_scores(0)
    row1 = _row_scores(1)
    np.testing.assert_allclose(scores[5], row0, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(scores[2], row1, atol=1e-6, rtol=0.0)
    assert not np.allclose(scores[5], row1, atol=1e-5)
    assert scores[5].shape == (3,)
    assert scores[2].shape == (3,)
