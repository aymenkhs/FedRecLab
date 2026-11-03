"""Local MSE gradient on generated Phi, rated-row uploads, batch sampling."""

import numpy as np
import pandas as pd
import pytest
import torch

from fedreclab.methods.metamf.metamf_model import apply_rating_mlp


def _autograd_dphi(payload_u, items, ratings, item_payload):
    """Gradient of mean squared error on the given rating multiset."""
    rows, inverse = np.unique(items, return_inverse=True)
    item_key = "item_left" if item_payload == "factors" else "item_embeddings"
    leaves = {}
    for key, value in payload_u.items():
        tensor = torch.as_tensor(value).clone()
        if key == item_key:
            tensor = tensor.index_select(0, torch.as_tensor(rows, dtype=torch.long))
        tensor.requires_grad_(True)
        leaves[key] = tensor
    if item_payload == "factors":
        local = leaves["item_left"].index_select(
            0, torch.as_tensor(inverse, dtype=torch.long)
        )
        item_emb = local @ leaves["item_right"]
    else:
        item_emb = leaves["item_embeddings"].index_select(
            0, torch.as_tensor(inverse, dtype=torch.long)
        )
    preds = apply_rating_mlp(
        item_emb, leaves["w1"], leaves["b1"], leaves["w2"], leaves["b2"]
    )
    rates = torch.as_tensor(ratings, dtype=preds.dtype)
    loss = torch.mean((preds - rates) ** 2)
    loss.backward()
    out = {
        "item_rows": rows.astype(np.int64),
        item_key: leaves[item_key].grad.detach().clone(),
        "w1": leaves["w1"].grad.detach().clone(),
        "b1": leaves["b1"].grad.detach().clone(),
        "w2": leaves["w2"].grad.detach().clone(),
        "b2": leaves["b2"].grad.detach().clone(),
    }
    if item_payload == "factors":
        out["item_right"] = leaves["item_right"].grad.detach().clone()
    return out


@pytest.mark.parametrize("item_payload", ["factors", "full"])
def test_upload_matches_autograd_of_mse(metamf_client_and_request, item_payload):
    client, request = metamf_client_and_request(item_payload=item_payload)
    response = client.local_train(request)
    upload = response.updates_packages["meta_network"].param_value[0]
    expected = _autograd_dphi(
        request.global_payload["generated_parameters"][0],
        request.training_data["item_id"].to_numpy(),
        request.training_data["rating"].to_numpy(),
        item_payload,
    )
    np.testing.assert_array_equal(upload["item_rows"], expected["item_rows"])
    item_key = "item_left" if item_payload == "factors" else "item_embeddings"
    for key in expected:
        if key == "item_rows":
            continue
        np.testing.assert_allclose(
            upload[key].detach().cpu().numpy(),
            expected[key].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    assert item_key in upload
    assert set(np.unique(request.training_data["item_id"])) == set(upload["item_rows"])
    assert response.num_training_examples == 2


def test_only_rated_rows_are_uploaded(metamf_client_and_request):
    client, request = metamf_client_and_request()
    response = client.local_train(request)
    upload = response.updates_packages["meta_network"].param_value[0]
    assert sorted(upload["item_rows"].tolist()) == [1, 2]
    assert upload["item_left"].shape[0] == 2


def test_missing_rating_column_raises(metamf_client_and_request):
    client, request = metamf_client_and_request()
    request.training_data = request.training_data.drop(columns=["rating"])
    with pytest.raises(ValueError, match="rating"):
        client.local_train(request)


def test_non_finite_loss_raises(metamf_client_and_request):
    data = pd.DataFrame(
        {"user_id": [0, 0], "item_id": [1, 2], "rating": [4.0, np.inf]}
    )
    client, request = metamf_client_and_request(training_data=data)
    with pytest.raises(ValueError, match="Non-finite"):
        client.local_train(request)


def test_seeded_batch_sampling_is_reproducible(metamf_client_and_request):
    data = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0],
            "item_id": [1, 2, 3, 4],
            "rating": [1.0, 2.0, 3.0, 4.0],
        }
    )
    client_a, request_a = metamf_client_and_request(
        training_data=data, local_batch_size=2, seed=11
    )
    client_b, request_b = metamf_client_and_request(
        training_data=data, local_batch_size=2, seed=11
    )
    samples_a = client_a._samples_per_user(
        request_a.users_in_client,
        request_a.training_data,
        client_a._round_rng(11, 0, 0),
    )
    samples_b = client_b._samples_per_user(
        request_b.users_in_client,
        request_b.training_data,
        client_b._round_rng(11, 0, 0),
    )
    np.testing.assert_array_equal(samples_a[0][1], samples_b[0][1])
    np.testing.assert_array_equal(samples_a[0][2], samples_b[0][2])


def test_with_replacement_upload_has_unique_rows(metamf_client_and_request):
    data = pd.DataFrame({"user_id": [0], "item_id": [3], "rating": [4.0]})
    client, request = metamf_client_and_request(
        training_data=data,
        local_batch_size=2,
        local_batch_replacement=True,
        seed=0,
    )
    items, ratings = client._sample_user_batch(
        data["item_id"].to_numpy(),
        data["rating"].to_numpy(),
        client._round_rng(0, 0, 0),
    )
    assert items.tolist() == [3, 3]
    response = client.local_train(request)
    upload = response.updates_packages["meta_network"].param_value[0]
    np.testing.assert_array_equal(upload["item_rows"], np.array([3], dtype=np.int64))
    expected = _autograd_dphi(
        request.global_payload["generated_parameters"][0],
        items,
        ratings,
        "factors",
    )
    np.testing.assert_allclose(
        upload["item_left"].detach().cpu().numpy(),
        expected["item_left"].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    assert response.num_training_examples == 2
