"""Local training must never modify server-held q_global or r_i.

These tests run on CPU, where torch.as_tensor and .to(device) share
memory with the payload. On a CUDA device the copy would hide the bug.
"""

import numpy as np
import torch

from src.fedreclab.methods.gpfedrec.gpfedrec_model import initial_item_table

NUM_ITEMS = 8
LATENT = 4


def _clone_payload(payload):
    return {k: torch.as_tensor(v).clone() for k, v in payload.items()}


def _assert_payload_unchanged(payload, snapshot):
    for key, before in snapshot.items():
        assert torch.equal(torch.as_tensor(payload[key]), before), key


def test_round_zero_payload_sharing_one_tensor_is_not_modified(make_client_and_request):
    client, request = make_client_and_request()
    payload = request.global_payload
    assert payload["item_embeddings"] is payload["personalized_item_embeddings"]
    snapshot = _clone_payload(payload)
    client.local_train(request)
    _assert_payload_unchanged(payload, snapshot)


def test_distinct_payload_tensors_are_not_modified(make_client_and_request):
    q = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=0)
    r = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=1)
    client, request = make_client_and_request(
        global_payload={"item_embeddings": q, "personalized_item_embeddings": r}
    )
    snapshot = _clone_payload(request.global_payload)
    client.local_train(request)
    _assert_payload_unchanged(request.global_payload, snapshot)


def test_numpy_payload_is_not_modified(make_client_and_request):
    q = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=0).numpy()
    r = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=1).numpy()
    client, request = make_client_and_request(
        global_payload={"item_embeddings": q, "personalized_item_embeddings": r}
    )
    q_before, r_before = q.copy(), r.copy()
    client.local_train(request)
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(r, r_before)


def test_upload_and_stored_table_do_not_share_storage(make_client_and_request):
    for scale in (0.0, 0.5):
        client, request = make_client_and_request(ldp_laplace_scale=scale)
        response = client.local_train(request)
        upload = response.updates_packages["item_embeddings"].param_value
        stored = response.updated_client_state.local_parameters["local_item_embeddings"]
        assert upload.untyped_storage().data_ptr() != stored.untyped_storage().data_ptr()


def test_aggregated_payload_is_not_modified_by_the_next_round(make_server):
    server = make_server()
    server.run_round(0)
    snapshot = _clone_payload(server._build_server_payload(client_id=0))
    stored = {cid: r.clone() for cid, r in server.personalized_item_embeddings.items()}

    server._train_client(0, 1)
    _assert_payload_unchanged(server._build_server_payload(client_id=0), snapshot)
    for cid, before in stored.items():
        assert torch.equal(server.personalized_item_embeddings[cid], before)


def test_numpy_framework_keeps_a_torch_q_global_that_training_does_not_modify(make_server):
    server = make_server(training_framework="numpy")
    server.run_round(0)
    q_global = server.global_parameters["item_embeddings"]
    assert isinstance(q_global, torch.Tensor)
    before = q_global.clone()
    server._train_client(0, 1)
    assert torch.equal(server.global_parameters["item_embeddings"], before)


def test_clients_in_one_round_see_the_same_q_global(make_server, monkeypatch):
    server = make_server()
    server.run_round(0)
    earlier_r = {cid: r.clone() for cid, r in server.personalized_item_embeddings.items()}
    q_before = server.global_parameters["item_embeddings"].clone()

    seen = []
    original = server.client_algorithm.local_train

    def recording_train(request):
        seen.append(torch.as_tensor(request.global_payload["item_embeddings"]).clone())
        return original(request)

    monkeypatch.setattr(server.client_algorithm, "local_train", recording_train)
    server._train_client(0, 1)
    server._train_client(1, 1)

    assert torch.equal(seen[0], seen[1])
    assert torch.equal(server.global_parameters["item_embeddings"], q_before)
    for cid, before in earlier_r.items():
        assert torch.equal(server.personalized_item_embeddings[cid], before)
