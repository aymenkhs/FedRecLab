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


_ITEM_WEIGHT = "item_embeddings.weight"

# Two users sharing item 2, with ratings interleaved in stored order.
_TWO_USERS = pd.DataFrame(
    {
        "user_id": [0, 1, 0, 1],
        "item_id": [1, 4, 2, 2],
        "rating": [4.0, 3.0, 5.0, 1.0],
    }
)


def _dense_param_ios():
    return {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="gradient",
            apply_to="server",
            layout="dense",
        )
    }


def _set_item_table(request, table):
    request.global_payload["item_embeddings"][_ITEM_WEIGHT] = table.clone()


def _random_item_table(num_items=6, latent_dim=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(num_items, latent_dim, generator=generator)


def _uploaded_dense(response, num_items):
    """Item gradient upload as a dense float64 table, whatever its layout."""
    up = response.updates_packages["item_embeddings"].param_value
    if "rows" not in up:
        return np.asarray(up[_ITEM_WEIGHT], dtype=np.float64)
    values = np.asarray(up["values"][_ITEM_WEIGHT], dtype=np.float64)
    table = np.zeros((num_items, values.shape[1]))
    table[np.asarray(up["rows"])] = values
    return table


def _user_vector(response, user_id):
    local = response.updated_client_state.local_parameters
    return local["user_embeddings"][local["user_id_to_local"][user_id]]


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


@pytest.mark.parametrize("local_update", ["sgd_sequential", "full_gradient"])
def test_multi_user_client_sums_item_gradients_across_users(
    fedmf_client_and_request, local_update
):
    hp = dict(lr=0.05, reg_u=0.1, reg_v=0.2, local_update=local_update)
    table = _random_item_table(seed=0)
    client, request = fedmf_client_and_request(training_data=_TWO_USERS, **hp)
    _set_item_table(request, table)
    resp = client.local_train(request)

    expected = np.zeros((6, 4))
    for uid in (0, 1):
        own_rows = _TWO_USERS[_TWO_USERS["user_id"] == uid].reset_index(drop=True)
        single, req = fedmf_client_and_request(training_data=own_rows, **hp)
        _set_item_table(req, table)
        single_resp = single.local_train(req)
        expected += _uploaded_dense(single_resp, 6)
        torch.testing.assert_close(
            _user_vector(resp, uid), _user_vector(single_resp, uid)
        )

    uploaded = resp.updates_packages["item_embeddings"].param_value
    assert np.asarray(uploaded["rows"]).tolist() == [1, 2, 4]
    np.testing.assert_allclose(
        _uploaded_dense(resp, 6), expected, rtol=1e-5, atol=1e-7
    )
    assert resp.num_training_examples == 4


def test_local_epochs_carry_user_factors_and_sum_item_gradients(
    fedmf_client_and_request,
):
    epochs = 3
    client, request = fedmf_client_and_request(
        lr=0.05, reg_u=0.1, reg_v=0.2, local_epochs=epochs
    )
    table = _random_item_table(seed=1)
    _set_item_table(request, table)
    v = table.numpy().astype(np.float64)
    u_ref = request.client_state.local_parameters["user_embeddings"][0].numpy()
    g_ref = np.zeros_like(v)
    for _ in range(epochs):
        u_ref, g = _github_user_update(
            u_ref, [(1, 4.0), (2, 5.0)], v, client.lr, client.reg_u, client.reg_v
        )
        g_ref += g

    resp = client.local_train(request)
    np.testing.assert_allclose(
        _user_vector(resp, 0).numpy(), u_ref, rtol=1e-5, atol=1e-6
    )
    uploaded = resp.updates_packages["item_embeddings"].param_value
    assert np.asarray(uploaded["rows"]).tolist() == [1, 2]
    np.testing.assert_allclose(
        _uploaded_dense(resp, 6), g_ref, rtol=1e-5, atol=1e-7
    )
    assert resp.num_training_examples == epochs * 2
    assert len(resp.training_losses) == epochs


@pytest.mark.parametrize("local_update", ["sgd_sequential", "full_gradient"])
def test_local_epochs_equal_chained_single_epoch_calls(
    fedmf_client_and_request, local_update
):
    hp = dict(
        lr=0.05,
        reg_u=0.1,
        reg_v=0.2,
        local_update=local_update,
        training_data=_TWO_USERS,
    )
    table = _random_item_table(seed=2)
    multi, req_multi = fedmf_client_and_request(local_epochs=2, **hp)
    _set_item_table(req_multi, table)
    resp = multi.local_train(req_multi)

    single, req = fedmf_client_and_request(**hp)
    _set_item_table(req, table)
    expected = np.zeros((6, 4))
    for _ in range(2):
        step = single.local_train(req)
        expected += _uploaded_dense(step, 6)
        req.client_state = step.updated_client_state

    np.testing.assert_allclose(
        _uploaded_dense(resp, 6), expected, rtol=1e-5, atol=1e-7
    )
    for uid in (0, 1):
        torch.testing.assert_close(_user_vector(resp, uid), _user_vector(step, uid))
    assert resp.num_training_examples == 2 * len(_TWO_USERS)


@pytest.mark.parametrize("local_update", ["sgd_sequential", "full_gradient"])
def test_fulltext_matches_parttext_for_multi_user_multi_epoch(
    fedmf_client_and_request, local_update
):
    hp = dict(
        lr=0.05,
        reg_u=0.1,
        reg_v=0.2,
        local_epochs=2,
        local_update=local_update,
        training_data=_TWO_USERS,
    )
    table = _random_item_table(seed=3)
    part, req_part = fedmf_client_and_request(**hp)
    full, req_full = fedmf_client_and_request(param_ios=_dense_param_ios(), **hp)
    _set_item_table(req_part, table)
    _set_item_table(req_full, table)
    part_resp = part.local_train(req_part)
    full_resp = full.local_train(req_full)

    part_up = part_resp.updates_packages["item_embeddings"].param_value
    full_table = full_resp.updates_packages["item_embeddings"].param_value[
        _ITEM_WEIGHT
    ]
    rows = np.asarray(part_up["rows"])
    assert full_table.shape == (6, 4)
    assert full_table.dtype == part_up["values"][_ITEM_WEIGHT].dtype
    torch.testing.assert_close(
        full_table[torch.from_numpy(rows)],
        part_up["values"][_ITEM_WEIGHT],
        rtol=0.0,
        atol=0.0,
    )
    unrated = torch.from_numpy(np.setdiff1d(np.arange(6), rows))
    assert torch.count_nonzero(full_table[unrated]) == 0
    for uid in (0, 1):
        torch.testing.assert_close(
            _user_vector(full_resp, uid), _user_vector(part_resp, uid),
            rtol=0.0, atol=0.0,
        )


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
