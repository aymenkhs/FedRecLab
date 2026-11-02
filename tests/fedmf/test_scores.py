"""FedMF compute_scores returns a user-id dict of inner products."""

import numpy as np
import torch


def test_scores_are_inner_products_keyed_by_user(fedmf_client_and_request):
    client, request = fedmf_client_and_request()
    resp = client.local_train(request)
    u = resp.updated_client_state.local_parameters["user_embeddings"][0]
    v = request.global_payload["item_embeddings"]["item_embeddings.weight"]
    candidates = np.array([1, 2, 3])
    scores = client.compute_scores(
        0,
        {0: candidates},
        request.global_payload,
        resp.updated_client_state,
        {},
    )
    assert list(scores) == [0]
    expected = (u.unsqueeze(0) * v[candidates]).sum(dim=-1).numpy()
    np.testing.assert_allclose(scores[0], expected, rtol=1e-5, atol=1e-5)


def test_scores_do_not_return_a_loss_tuple(fedmf_client_and_request):
    client, request = fedmf_client_and_request()
    resp = client.local_train(request)
    out = client.compute_scores(
        0,
        {0: np.array([1, 2])},
        request.global_payload,
        resp.updated_client_state,
        {},
    )
    assert isinstance(out, dict)
    assert not isinstance(out, tuple)
