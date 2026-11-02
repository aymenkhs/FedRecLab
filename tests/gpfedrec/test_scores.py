"""GPFedRec scoring: alignment, local table, fallback to q_global."""

import numpy as np
import torch

from src.fedreclab.federated.base_client import ClientStateBase
from src.fedreclab.methods.gpfedrec.gpfedrec_model import predict_logits


def _with_table(state, table):
    local = dict(state.local_parameters)
    local["local_item_embeddings"] = table
    return ClientStateBase(client_id=state.client_id, local_parameters=local)


def test_scores_are_aligned_with_candidates(make_client_and_request):
    client, request = make_client_and_request()
    state = client.local_train(request).updated_client_state
    candidates = np.array([5, 0, 3])
    scores = client.compute_scores(
        0, {0: candidates}, {"item_embeddings": request.global_payload["item_embeddings"]}, state, {}
    )
    local = state.local_parameters
    model = client._skeleton
    model.load_state(local["score_function"])
    model.eval()
    with torch.no_grad():
        expected = predict_logits(
            model,
            torch.zeros(3, dtype=torch.long),
            torch.as_tensor(candidates),
            local["user_embeddings"],
            local["local_item_embeddings"],
        ).numpy()
    assert scores[0].shape == (3,)
    np.testing.assert_allclose(scores[0], expected, rtol=1e-6)


def test_trained_client_scores_with_its_local_table(make_client_and_request):
    client, request = make_client_and_request()
    state = client.local_train(request).updated_client_state
    local_table = state.local_parameters["local_item_embeddings"]
    candidates = {0: np.arange(8)}
    with_local = client.compute_scores(0, candidates, {"item_embeddings": local_table * 0.0}, state, {})
    as_global = client.compute_scores(
        0, candidates, {"item_embeddings": local_table}, _with_table(state, None), {}
    )
    np.testing.assert_allclose(with_local[0], as_global[0], rtol=1e-6)


def test_untrained_client_scores_with_q_global(make_client_and_request):
    client, request = make_client_and_request()
    state = request.client_state
    assert state.local_parameters["local_item_embeddings"] is None
    q = request.global_payload["item_embeddings"]
    candidates = {0: np.arange(8)}
    fallback = client.compute_scores(0, candidates, {"item_embeddings": q}, state, {})
    explicit = client.compute_scores(0, candidates, {}, _with_table(state, q.clone()), {})
    np.testing.assert_allclose(fallback[0], explicit[0], rtol=1e-6)
