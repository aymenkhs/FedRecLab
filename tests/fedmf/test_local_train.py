"""FedMF local_train matches the reference sequential pass and ParamIO layout."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedmf.fedmf import default_param_ios
from src.fedreclab.methods.fedmf.fedmf_client import FedMFClientAlgorithm


def _github_user_update(u, ratings, item_vector, lr, reg_u, reg_v):
    """Authors' FedMF_Full.py / FedMF_Part.py user_update, without encryption."""
    u = np.array(u, dtype=np.float64, copy=True)
    gradient = np.zeros_like(item_vector, dtype=np.float64)
    for item_id, rate in ratings:
        error = rate - np.dot(u, item_vector[item_id])
        u = u - lr * (-2.0 * error * item_vector[item_id] + 2.0 * reg_u * u)
        gradient[item_id] = lr * (
            -2.0 * error * u + 2.0 * reg_v * item_vector[item_id]
        )
    return u, gradient


def test_defaults_follow_the_reference_code():
    client = FedMFClientAlgorithm()
    assert client.local_update == "sgd_sequential"
    assert client.u_regularisation == "per_rating"
    assert client.v_regularisation == "per_rating"
    assert client.lr_placement == "client"
    assert client.loss_scale == "none"
    assert client.latent_dim == 100
    assert client.lr == pytest.approx(1e-3)
    assert client.reg_u == pytest.approx(1e-4)
    assert client.reg_v == pytest.approx(1e-4)
    assert client.init_mode == "constant"
    assert client.init_value == pytest.approx(0.01)
    assert client.local_epochs == 1


def test_sequential_pass_matches_github_user_update(fedmf_client_and_request):
    client, request = fedmf_client_and_request(lr=0.01, reg_u=0.1, reg_v=0.2)
    v = request.global_payload["item_embeddings"]["item_embeddings.weight"].numpy()
    u0 = request.client_state.local_parameters["user_embeddings"][0].numpy()
    ratings = [(1, 4.0), (2, 5.0)]
    u_ref, g_ref = _github_user_update(u0, ratings, v, client.lr, client.reg_u, client.reg_v)

    resp = client.local_train(request)
    u_new = resp.updated_client_state.local_parameters["user_embeddings"][0].numpy()
    np.testing.assert_allclose(u_new, u_ref, rtol=1e-5, atol=1e-5)

    uploaded = resp.updates_packages["item_embeddings"].param_value
    rows = np.asarray(uploaded["rows"])
    values = np.asarray(uploaded["values"]["item_embeddings.weight"])
    np.testing.assert_array_equal(np.sort(rows), np.array([1, 2]))
    order = np.argsort(rows)
    np.testing.assert_allclose(values[order], g_ref[[1, 2]], rtol=1e-5, atol=1e-5)


def test_parttext_omits_unrated_items(fedmf_client_and_request):
    client, request = fedmf_client_and_request()
    resp = client.local_train(request)
    uploaded = resp.updates_packages["item_embeddings"].param_value
    assert set(np.asarray(uploaded["rows"]).tolist()) == {1, 2}
    assert uploaded["values"]["item_embeddings.weight"].shape[0] == 2


def test_fulltext_dense_layout_sends_all_rows(fedmf_client_and_request):
    pio = {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="gradient",
            apply_to="server",
            layout="dense",
        )
    }
    client, request = fedmf_client_and_request(param_ios=pio)
    resp = client.local_train(request)
    uploaded = resp.updates_packages["item_embeddings"].param_value
    table = np.asarray(uploaded["item_embeddings.weight"])
    assert table.shape == (6, 4)
    np.testing.assert_allclose(table[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(table[3], 0.0, atol=1e-7)
    assert np.linalg.norm(table[1]) > 0
    assert np.linalg.norm(table[2]) > 0


def test_fulltext_zeros_match_parttext_on_rated_rows(fedmf_client_and_request):
    part, req_part = fedmf_client_and_request()
    dense_io = {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="gradient",
            apply_to="server",
            layout="dense",
        )
    }
    full, req_full = fedmf_client_and_request(param_ios=dense_io)
    part_resp = part.local_train(req_part)
    full_resp = full.local_train(req_full)
    part_up = part_resp.updates_packages["item_embeddings"].param_value
    full_table = np.asarray(
        full_resp.updates_packages["item_embeddings"].param_value[
            "item_embeddings.weight"
        ]
    )
    rows = np.asarray(part_up["rows"])
    part_vals = np.asarray(part_up["values"]["item_embeddings.weight"])
    np.testing.assert_allclose(full_table[rows], part_vals, rtol=1e-6, atol=1e-6)
    unrated = np.setdiff1d(np.arange(6), rows)
    np.testing.assert_allclose(full_table[unrated], 0.0, atol=1e-7)


def test_full_gradient_item_grad_uses_old_u(fedmf_client_and_request):
    """Sequential SGD evaluates the item gradient at the updated u; the paper does not."""
    seq, req_seq = fedmf_client_and_request(lr=0.05, reg_u=0.0, reg_v=0.0)
    v = req_seq.global_payload["item_embeddings"]["item_embeddings.weight"]
    v[1] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    v[2] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    full, req_full = fedmf_client_and_request(
        lr=0.05, reg_u=0.0, reg_v=0.0, local_update="full_gradient"
    )
    req_full.global_payload["item_embeddings"]["item_embeddings.weight"] = v.clone()
    req_seq.global_payload["item_embeddings"]["item_embeddings.weight"] = v.clone()
    g_seq = seq.local_train(req_seq).updates_packages["item_embeddings"].param_value
    g_full = full.local_train(req_full).updates_packages["item_embeddings"].param_value
    seq_rows = {int(r): i for i, r in enumerate(np.asarray(g_seq["rows"]))}
    full_rows = {int(r): i for i, r in enumerate(np.asarray(g_full["rows"]))}
    seq_v1 = np.asarray(g_seq["values"]["item_embeddings.weight"][seq_rows[1]])
    full_v1 = np.asarray(g_full["values"]["item_embeddings.weight"][full_rows[1]])
    assert not np.allclose(seq_v1, full_v1, atol=1e-6)


def test_global_m_scales_only_the_data_term(fedmf_client_and_request):
    client, request = fedmf_client_and_request(
        lr=0.01,
        reg_u=0.0,
        reg_v=0.0,
        loss_scale="global_M",
        extra_payload={"n_train_ratings": 4},
        local_update="full_gradient",
    )
    v = request.global_payload["item_embeddings"]["item_embeddings.weight"]
    u0 = request.client_state.local_parameters["user_embeddings"][0].clone()
    resp = client.local_train(request)
    # One rating of item 1: data term is (-2 error v)/M, then u -= lr * that.
    u1 = u0.clone()
    item = v[1]
    error = 4.0 - float(torch.dot(u0, item))
    item2 = v[2]
    error2 = 5.0 - float(torch.dot(u0, item2))
    data_u = ((-2.0 * error) * item + (-2.0 * error2) * item2) / 4.0
    expected = u0 - 0.01 * data_u
    got = resp.updated_client_state.local_parameters["user_embeddings"][0]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_global_m_without_payload_fails(fedmf_client_and_request):
    client, request = fedmf_client_and_request(
        loss_scale="global_M", extra_payload={}
    )
    with pytest.raises(ValueError, match="n_train_ratings"):
        client.local_train(request)


def test_missing_rating_column_fails(fedmf_client_and_request):
    data = pd.DataFrame({"user_id": [0], "item_id": [1]})
    client, request = fedmf_client_and_request(training_data=data)
    with pytest.raises(ValueError, match="rating"):
        client.local_train(request)


def test_unknown_local_update_raises():
    with pytest.raises(ValueError, match="local_update"):
        FedMFClientAlgorithm(local_update="adam")


def test_client_reads_param_ios_not_a_private_flag(fedmf_client_and_request):
    client, request = fedmf_client_and_request()
    assert not hasattr(client, "part_text")
    assert request.param_ios["item_embeddings"].layout == "rows"
    pio = default_param_ios()
    assert pio["item_embeddings"].layout == "rows"
    assert pio["item_embeddings"].exchange == "gradient"
