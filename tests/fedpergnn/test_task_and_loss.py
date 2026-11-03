"""FedPerGNN task modes: rating MSE on observed rows, ranking BCE with negatives."""

import numpy as np
import pytest
import torch

from src.fedreclab.methods.fedpergnn.fedpergnn_client import FedPerGNNClientAlgorithm


def test_defaults_are_rating_mse_and_no_negatives():
    client = FedPerGNNClientAlgorithm()
    assert client.task == "rating"
    assert client.neg_samples == 0


def test_ranking_defaults_to_four_negatives():
    client = FedPerGNNClientAlgorithm(task="ranking")
    assert client.task == "ranking"
    assert client.neg_samples == 4


def test_rating_rejects_nonzero_negatives():
    with pytest.raises(ValueError, match="task='rating' uses observed items only"):
        FedPerGNNClientAlgorithm(task="rating", neg_samples=4)


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="task must be one of"):
        FedPerGNNClientAlgorithm(task="implicit")


def test_negative_neg_samples_raises():
    with pytest.raises(ValueError, match="neg_samples"):
        FedPerGNNClientAlgorithm(task="ranking", neg_samples=-1)


def test_rating_keeps_raw_ratings(fedpergnn_client_and_request):
    client, _ = fedpergnn_client_and_request()
    items, y = client._build_epoch_samples(
        np.array([1]), np.array([4.0], dtype=np.float32), None
    )
    np.testing.assert_array_equal(items, np.array([1]))
    np.testing.assert_allclose(y, np.array([4.0], dtype=np.float32))


def test_ranking_binarises_a_rating_of_four(fedpergnn_client_and_request):
    client, _ = fedpergnn_client_and_request(task="ranking", neg_samples=0)
    _, y = client._build_epoch_samples(
        np.array([1]), np.array([4.0], dtype=np.float32), None
    )
    np.testing.assert_array_equal(y, np.array([1.0], dtype=np.float32))


def test_ranking_sampled_negatives_are_zero(fedpergnn_client_and_request):
    client, _ = fedpergnn_client_and_request(task="ranking", neg_samples=1)
    rng = np.random.default_rng(0)
    items, y = client._build_epoch_samples(
        np.array([1]), np.array([4.0], dtype=np.float32), rng
    )
    assert items[0] == 1
    assert y[0] == 1.0
    assert items[1] != 1
    assert y[1] == 0.0
    assert items[1] not in {1}


def test_ranking_cannot_sample_when_every_item_is_observed():
    client = FedPerGNNClientAlgorithm(task="ranking", neg_samples=1)
    client._num_items = 2
    with pytest.raises(RuntimeError, match="every catalogue item"):
        client._build_epoch_samples(
            np.array([0, 1]), np.array([1.0, 1.0], dtype=np.float32), None
        )


def test_rating_loss_is_mse_ranking_loss_is_bce():
    preds = torch.tensor([0.0, 1.0])
    rating_targets = torch.tensor([4.0, 1.0])
    binary_targets = torch.tensor([1.0, 0.0])
    rating = FedPerGNNClientAlgorithm()
    ranking = FedPerGNNClientAlgorithm(task="ranking", neg_samples=0)
    expected_mse = torch.nn.functional.mse_loss(preds, rating_targets)
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        preds, binary_targets
    )
    assert torch.allclose(rating._compute_loss(preds, rating_targets), expected_mse)
    assert torch.allclose(ranking._compute_loss(preds, binary_targets), expected_bce)


def test_local_train_records_task_and_ranking_evaluator_scores(
    fedpergnn_client_and_request,
):
    client, request = fedpergnn_client_and_request(task="ranking", neg_samples=1)
    resp = client.local_train(request)
    assert resp.extra_response["task"] == "ranking"
    assert resp.num_training_examples == 4

    scores = client.compute_scores(
        client_id=0,
        candidate_items_per_user={0: np.array([1, 3, 4])},
        global_parameters=request.global_payload,
        client_state=resp.updated_client_state,
        extra_payload=request.extra_payload,
    )
    assert scores[0].shape == (3,)
    assert np.isfinite(scores[0]).all()


def test_user_tower_runs_once_per_batch_and_once_per_score_call(
    fedpergnn_client_and_request, monkeypatch
):
    """Training and scoring must not rebuild the user tower per item.

    Two local rows share one batch, and five candidates share one score
    call. Each of those steps encodes the user once.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.
    monkeypatch : pytest.MonkeyPatch
        Replaces ``user_representation`` with a counter.

    Raises
    ------
    AssertionError
        If the tower runs more than once in training or in scoring.
    """
    client, request = fedpergnn_client_and_request()
    calls = {"n": 0}
    original = client._skeleton.user_representation

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(client._skeleton, "user_representation", wrapped)
    client.local_train(request)
    assert calls["n"] == 1

    calls["n"] = 0
    scores = client.compute_scores(
        client_id=0,
        candidate_items_per_user={0: np.array([1, 2, 3, 4, 5])},
        global_parameters=request.global_payload,
        client_state=request.client_state,
        extra_payload=request.extra_payload,
    )
    assert calls["n"] == 1
    assert scores[0].shape == (5,)


def test_trainable_neighbour_row_is_uploaded_from_the_shared_tower(
    fedpergnn_client_and_request,
):
    """A neighbour gradient from one shared user state is uploaded.

    The expansion has a single real neighbour row. After one local step
    that row must appear in the user-embedding upload and must have moved.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If the neighbour id is missing or its uploaded vector is unchanged.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    hidden = client.hidden_dim
    expansion = request.extra_payload["expanded_user_data"][0]
    history_len, neighbor_len, _ = expansion["neighbor_embeddings"].shape
    embeddings = np.zeros((history_len, neighbor_len, hidden), dtype=np.float32)
    embeddings[0, 0] = np.linspace(0.1, 0.8, hidden, dtype=np.float32)
    neighbor_ids = np.full((history_len, neighbor_len), -1, dtype=np.int64)
    neighbor_ids[0, 0] = 4
    expansion["neighbor_embeddings"] = embeddings
    expansion["neighbor_user_ids"] = neighbor_ids

    response = client.local_train(request)
    upload = response.updates_packages["user_embeddings"].param_value
    rows = [int(row) for row in upload["rows"]]
    assert 4 in rows
    updated = upload["values"][rows.index(4)]
    assert updated.shape == (hidden,)
    assert np.isfinite(updated).all()
    assert not np.allclose(updated, embeddings[0, 0])
