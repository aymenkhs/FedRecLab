"""FedFast local train uploads and client-state user update."""

import numpy as np
import torch


def test_local_train_updates_user_and_uploads_rows(fedfast_client_and_request):
    client, request = fedfast_client_and_request()
    before = request.client_state.local_parameters["user_embeddings"].clone()
    response = client.local_train(request)

    after = response.updated_client_state.local_parameters["user_embeddings"]
    assert not torch.allclose(before, after)

    item_up = response.updates_packages["item_embeddings"].param_value
    assert "rows" in item_up and "values" in item_up
    assert set(item_up["rows"].tolist()) == {1, 2}

    out_up = response.updates_packages["output_weights"].param_value
    assert "output_logits.weight" in out_up


def test_compute_scores_shape(fedfast_client_and_request):
    client, request = fedfast_client_and_request()
    response = client.local_train(request)
    scores = client.compute_scores(
        client_id=0,
        candidate_items_per_user={0: np.array([0, 1, 2])},
        global_parameters=request.global_payload,
        client_state=response.updated_client_state,
        extra_payload={},
    )
    assert scores[0].shape == (3,)
