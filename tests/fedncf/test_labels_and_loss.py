"""FedNCF paper protocol: binary labels and BCE, not rating-valued MSE."""

import numpy as np
import pytest

from src.fedreclab.methods.fedncf.fedncf_client import FedNCFClientAlgorithm


def test_paper_defaults_are_binary_bce_and_four_negatives():
    client = FedNCFClientAlgorithm()
    assert client.label_mode == "binary"
    assert client.loss == "bce"
    assert client.neg_samples == 4


def test_binary_labels_ignore_rating_magnitudes(fedncf_client_and_request):
    client, _ = fedncf_client_and_request()
    _, y = client._build_epoch_samples(
        np.array([0, 0]),
        np.array([1, 2]),
        np.array([5.0, 0.5], dtype=np.float32),
        {0: {1, 2}},
        None,
    )
    np.testing.assert_array_equal(y, np.array([1.0, 1.0], dtype=np.float32))


def test_rating_labels_keep_raw_values(fedncf_client_and_request):
    client, _ = fedncf_client_and_request(label_mode="rating", loss="mse")
    _, y = client._build_epoch_samples(
        np.array([0, 0]),
        np.array([1, 2]),
        np.array([5.0, 0.5], dtype=np.float32),
        {0: {1, 2}},
        None,
    )
    np.testing.assert_allclose(y, np.array([5.0, 0.5], dtype=np.float32))


def test_sampled_negatives_are_zero(fedncf_client_and_request):
    client, _ = fedncf_client_and_request()
    client.neg_samples = 1
    client.num_items = 6
    rng = np.random.default_rng(0)
    _, y = client._build_epoch_samples(
        np.array([0]),
        np.array([1]),
        np.array([4.0], dtype=np.float32),
        {0: {1}},
        rng,
    )
    assert y[0] == 1.0
    assert y[1] == 0.0


def test_bce_rejects_rating_labels():
    with pytest.raises(ValueError, match="loss='bce' requires label_mode='binary'"):
        FedNCFClientAlgorithm(label_mode="rating", loss="bce")


def test_unknown_label_mode_raises():
    with pytest.raises(ValueError, match="label_mode"):
        FedNCFClientAlgorithm(label_mode="implicit")


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="loss must be one of"):
        FedNCFClientAlgorithm(loss="hinge")
