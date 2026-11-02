"""EFVAE local_train: finite loss, sparse support, dense middle."""

import numpy as np
import torch

from tests.efvae.conftest import NUM_ITEMS


def test_local_train_finite_and_packages(efvae_client_and_request):
    client, request = efvae_client_and_request()
    response = client.local_train(request)
    assert np.isfinite(response.training_losses[0])
    assert set(response.updates_packages) == {
        "enc_embeddings",
        "dec_embeddings",
        "dec_bias",
        "middle",
    }
    middle = response.updates_packages["middle"].param_value
    assert "enc_bias" in middle
    assert any(k.startswith("q_layers.") for k in middle)
    assert any(k.startswith("p_layers.") for k in middle)
    for name in middle:
        assert torch.isfinite(torch.as_tensor(middle[name])).all()


def test_sparse_rows_subset_of_catalogue(efvae_client_and_request):
    client, request = efvae_client_and_request()
    response = client.local_train(request)
    for group in ("enc_embeddings", "dec_embeddings", "dec_bias"):
        packed = response.updates_packages[group].param_value
        rows = np.asarray(packed["rows"], dtype=np.int64)
        assert rows.ndim == 1
        assert len(rows) == len(np.unique(rows))
        if rows.size:
            assert rows.min() >= 0
            assert rows.max() < NUM_ITEMS
        values = packed["values"]
        assert isinstance(values, dict)
        key = next(iter(values))
        assert values[key].shape[0] == rows.size


def test_missing_sampler_raises(efvae_client_and_request):
    client, request = efvae_client_and_request(extra_payload={"anneal": 0.2})
    try:
        client.local_train(request)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "fcis_sampler_state" in str(exc)
