"""FedPerGNN uploads pseudo-item rows and can clip them with Laplace LDP."""

import numpy as np
import pytest

from src.fedreclab.federated.base_client import BaseClientAlgorithm
from src.fedreclab.federated.mechanisms import ClipLaplaceLDP
from src.fedreclab.methods.fedpergnn.fedpergnn_client import FedPerGNNClientAlgorithm


def _l1(parts):
    return float(sum(np.abs(np.asarray(part)).sum() for part in parts))


def test_privacy_defaults_are_off():
    client = FedPerGNNClientAlgorithm()
    assert client.num_pseudo_items == 0
    assert client.ldp_clip is None
    assert client.ldp_scale == 0.0
    assert client.apply_privacy(None, "response") == "response"
    assert BaseClientAlgorithm().apply_privacy(None, "response") == "response"


def test_privacy_off_leaves_the_item_upload_unchanged(fedpergnn_client_and_request):
    client, request = fedpergnn_client_and_request()
    response = client.local_train(request)
    upload = response.updates_packages["item_embeddings"].param_value
    rows = upload["rows"].copy()
    values = upload["values"].copy()
    returned = client.apply_privacy(request, response)
    assert returned is response
    after = response.updates_packages["item_embeddings"].param_value
    np.testing.assert_array_equal(after["rows"], rows)
    np.testing.assert_array_equal(after["values"], values)


def test_pseudo_items_are_appended_and_do_not_enter_the_loss(fedpergnn_client_and_request):
    client, request = fedpergnn_client_and_request(num_pseudo_items=2)
    response = client.local_train(request)
    assert response.num_training_examples == 2
    raw = response.updates_packages["item_embeddings"].param_value
    raw_rows = np.asarray(raw["rows"]).copy()
    raw_values = np.asarray(raw["values"]).copy()

    response = client.apply_privacy(request, response)
    upload = response.updates_packages["item_embeddings"].param_value
    rows = np.asarray(upload["rows"])
    values = np.asarray(upload["values"])
    np.testing.assert_array_equal(rows[: raw_rows.size], raw_rows)
    np.testing.assert_allclose(values[: raw_rows.size], raw_values)

    extra = rows[raw_rows.size :]
    assert extra.shape == (2,)
    assert set(int(i) for i in extra).isdisjoint(set(int(i) for i in raw_rows))
    assert set(int(i) for i in extra).isdisjoint({1, 2})

    downloaded = np.asarray(request.global_payload["item_embeddings"], dtype=np.float32)
    rng = client._privacy_rng(request)
    observed = {1, 2}
    expected_ids = client.pseudo_items.sample_ids(
        downloaded.shape[0], observed.union(int(i) for i in raw_rows), rng
    )
    real_idx = np.array([int(row) in observed for row in raw_rows])
    real_deltas = downloaded[raw_rows[real_idx]] - raw_values[real_idx]
    expected_deltas = client.pseudo_items.matched_gradients(real_deltas, rng)
    np.testing.assert_array_equal(extra, expected_ids)
    np.testing.assert_allclose(
        values[raw_rows.size :], downloaded[expected_ids] - expected_deltas
    )


def test_pseudo_sampling_requires_a_seed_and_enough_items(fedpergnn_client_and_request):
    client, request = fedpergnn_client_and_request(num_pseudo_items=1)
    response = client.local_train(request)
    request.seed = None
    with pytest.raises(RuntimeError, match="seed"):
        client.apply_privacy(request, response)

    client, request = fedpergnn_client_and_request(num_pseudo_items=4)
    response = client.local_train(request)
    with pytest.raises(ValueError, match="pseudo items"):
        client.apply_privacy(request, response)


def test_clip_rewrites_the_upload_from_one_shared_l1_budget(fedpergnn_client_and_request):
    clip = 1e-3
    client, request = fedpergnn_client_and_request(ldp_clip=clip, ldp_scale=0.0)
    response = client.local_train(request)
    items = response.updates_packages["item_embeddings"].param_value
    downloaded_items = np.asarray(
        request.global_payload["item_embeddings"], dtype=np.float32
    )
    rows = np.asarray(items["rows"])
    item_values = np.asarray(items["values"], dtype=np.float32)
    item_deltas = downloaded_items[rows] - item_values
    user_starts, user_deltas = client._user_deltas(request, response)
    keys, model_starts, model_deltas = client._model_deltas(request, response)
    raw_l1 = _l1([item_deltas, user_deltas, *model_deltas])
    assert raw_l1 > clip

    pseudo = np.zeros((0, item_values.shape[1]), dtype=np.float32)
    protected = ClipLaplaceLDP(clip=clip, scale=0.0).apply(
        [item_deltas, pseudo, user_deltas, *model_deltas],
        np.random.default_rng(0),
    )
    assert _l1(protected) == pytest.approx(clip, rel=1e-4)

    client.apply_privacy(request, response)
    after_items = response.updates_packages["item_embeddings"].param_value
    np.testing.assert_array_equal(after_items["rows"], rows)
    np.testing.assert_allclose(after_items["values"], downloaded_items[rows] - protected[0])
    np.testing.assert_allclose(
        response.updates_packages["user_embeddings"].param_value["values"],
        user_starts - protected[2],
    )
    after_model = response.updates_packages["model_state"].param_value
    for key, start, delta in zip(keys, model_starts, protected[3:]):
        np.testing.assert_allclose(after_model[key], start - delta)


def test_pseudo_items_survive_clipping(fedpergnn_client_and_request):
    client, request = fedpergnn_client_and_request(
        num_pseudo_items=2, ldp_clip=0.5, ldp_scale=0.01
    )
    response = client.local_train(request)
    before = len(response.updates_packages["item_embeddings"].param_value["rows"])
    client.apply_privacy(request, response)
    upload = response.updates_packages["item_embeddings"].param_value
    assert len(upload["rows"]) == before + 2
    assert np.isfinite(upload["values"]).all()
    model = response.updates_packages["model_state"].param_value
    assert all(np.isfinite(np.asarray(value)).all() for value in model.values())


def test_constructor_rejects_bad_privacy_settings():
    with pytest.raises(ValueError, match="count"):
        FedPerGNNClientAlgorithm(num_pseudo_items=-1)
    with pytest.raises(ValueError, match="clip"):
        FedPerGNNClientAlgorithm(ldp_clip=0.0)
    with pytest.raises(ValueError, match="scale"):
        FedPerGNNClientAlgorithm(ldp_scale=-1.0)
