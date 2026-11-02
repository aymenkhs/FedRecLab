"""EFVAE compute_scores: finite scores for empty and non-empty history."""

import numpy as np


def test_scores_nonempty_history(efvae_client_and_request):
    client, request = efvae_client_and_request()
    scores = client.compute_scores(
        client_id=0,
        candidate_items_per_user={0: np.array([0, 3, 5], dtype=np.int64)},
        global_parameters=request.global_payload,
        client_state=request.client_state,
        extra_payload={},
    )
    assert scores[0].shape == (3,)
    assert np.isfinite(scores[0]).all()


def test_scores_empty_history(efvae_client_and_request):
    client, request = efvae_client_and_request(
        training_data=__import__("pandas").DataFrame(
            columns=["user_id", "item_id", "rating"]
        ),
        users_in_client=[0],
    )
    # Re-init empty history on state
    request.client_state.local_parameters["train_items_per_user"][0] = np.array(
        [], dtype=np.int64
    )
    scores = client.compute_scores(
        client_id=0,
        candidate_items_per_user={0: np.array([1, 2], dtype=np.int64)},
        global_parameters=request.global_payload,
        client_state=request.client_state,
        extra_payload={},
    )
    assert scores[0].shape == (2,)
    assert np.isfinite(scores[0]).all()
