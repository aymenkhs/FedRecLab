"""What FedNCF stores on the client between rounds.

ClientState should hold the two user embedding tables (and optional Adam
moments for them), not a copy of the full NCF module.
"""

import numpy as np
import pytest
import torch

from src.fedreclab.federated.base_client import ClientTrainingRequest
from src.fedreclab.federated.utils import (
    assert_skeleton_server_coverage,
    restore_named_optimizer_state,
)
from src.fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering


def test_local_train_does_not_store_the_module(fedncf_client_and_request):
    client, request = fedncf_client_and_request()
    resp = client.local_train(request)
    local = resp.updated_client_state.local_parameters
    assert "model" not in local
    assert "gmf_user_embeddings.weight" in local["user_embeddings"]
    assert "mlp_user_embeddings.weight" in local["user_embeddings"]


def test_default_scope_stores_adam_moments_only_for_user_tables(
    fedncf_client_and_request,
):
    client, request = fedncf_client_and_request(optimizer_state_scope="client_resident")
    resp = client.local_train(request)
    keys = set(resp.updated_client_state.optimizer_state)
    assert keys <= {"gmf_user_embeddings.weight", "mlp_user_embeddings.weight"}
    assert "gmf_item_embeddings.weight" not in keys


def test_none_scope_stores_no_optimizer_state(fedncf_client_and_request):
    client, request = fedncf_client_and_request(optimizer_state_scope="none")
    resp = client.local_train(request)
    assert resp.updated_client_state.optimizer_state is None


def test_unknown_optimizer_name_raises(fedncf_client_and_request):
    client, _ = fedncf_client_and_request()
    named = dict(client._skeleton.named_parameters())
    optimizer = torch.optim.Adam(client._skeleton.parameters(), lr=0.01)
    with pytest.raises(KeyError, match="no parameter"):
        restore_named_optimizer_state(
            optimizer,
            named,
            {"not_a_real_parameter": {"exp_avg": torch.zeros(1)}},
        )


def test_item_table_moments_are_not_restored_on_the_next_round(
    fedncf_client_and_request,
):
    """Default scope never writes item-table moments, so round 2 cannot restore them."""
    client, request = fedncf_client_and_request()
    resp1 = client.local_train(request)
    request2 = ClientTrainingRequest(
        client_id=0,
        current_round=1,
        users_in_client=[0],
        client_state=resp1.updated_client_state,
        global_payload=request.global_payload,
        training_data=request.training_data,
        param_ios=request.param_ios,
    )
    resp2 = client.local_train(request2)
    assert "gmf_item_embeddings.weight" not in (
        resp2.updated_client_state.optimizer_state or {}
    )


def test_scoring_does_not_need_a_stored_module(fedncf_client_and_request):
    client, request = fedncf_client_and_request()
    scores, _ = client.compute_scores(
        0,
        {0: np.array([1, 2, 3])},
        request.global_payload,
        request.client_state,
        None,
    )
    assert scores[0].shape == (3,)


def test_every_skeleton_tensor_is_in_the_server_groups():
    model = NeuralCollaborativeFiltering(num_items=4, predictive_factor=8)
    names = {k for g in model.get_server_state_groups().values() for k in g}
    assert_skeleton_server_coverage(model, names)
    broken = set(names)
    broken.remove("gmf_item_embeddings.weight")
    with pytest.raises(RuntimeError, match="Skeleton coverage mismatch"):
        assert_skeleton_server_coverage(model, broken)
