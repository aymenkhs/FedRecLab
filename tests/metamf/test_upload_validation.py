"""_collect_updates rejects foreign users, bad rows, and coalesces duplicates."""

import numpy as np
import pytest
import torch

from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingResponse
from src.fedreclab.federated.utils import UpdatePackage
from src.fedreclab.methods.metamf.metamf import default_param_ios
from tests.metamf.conftest import make_server, tiny_dataset


def _empty_state(client_id=0):
    return ClientStateBase(client_id=client_id, local_parameters={}, optimizer_state=None)


def _response(client_id, upload):
    pio = default_param_ios()["meta_network"]
    return ClientTrainingResponse(
        client_id=client_id,
        updated_client_state=_empty_state(client_id),
        updates_packages={
            "meta_network": UpdatePackage(
                param_name="meta_network",
                param_io=pio,
                param_value=upload,
                param_metadata={},
            )
        },
        training_losses=[0.0],
        training_duration=0.0,
        num_training_examples=1,
    )


def _valid_dphi(server, user_id, rows, item_left=None):
    rows = np.asarray(rows, dtype=np.int64)
    s = server.item_mem_num
    d = server.item_emb_dim
    h = server.rp_hidden_dim
    if item_left is None:
        item_left = torch.ones((len(rows), s), dtype=torch.float32)
    return {
        "item_rows": rows,
        "item_left": item_left,
        "item_right": torch.zeros(s, d),
        "w1": torch.zeros(d, h),
        "b1": torch.zeros(1, h),
        "w2": torch.zeros(h, 1),
        "b2": torch.zeros(1, 1),
    }


def _translate(server, client_id, upload):
    pio = server.param_ios["meta_network"]
    return server._collect_updates(
        "meta_network", pio, [_response(client_id, upload)]
    )


def test_foreign_user_id_raises():
    server = make_server(tiny_dataset(n_users=3))
    dphi = _valid_dphi(server, 1, [1])
    with pytest.raises(ValueError, match="user_id=1"):
        _translate(server, 0, {1: dphi})


def test_out_of_range_rows_raise():
    server = make_server(tiny_dataset(n_users=2))
    with pytest.raises(IndexError, match="item_rows"):
        _translate(server, 0, {0: _valid_dphi(server, 0, [-1])})
    with pytest.raises(IndexError, match="item_rows"):
        _translate(
            server, 0, {0: _valid_dphi(server, 0, [server.interaction_dataset.num_items])}
        )


def test_wrong_keys_and_shapes_raise():
    server = make_server(tiny_dataset(n_users=2))
    dphi = _valid_dphi(server, 0, [1])
    bad_keys = dict(dphi)
    bad_keys.pop("w1")
    with pytest.raises(ValueError, match="keys"):
        _translate(server, 0, {0: bad_keys})
    bad_shape = dict(dphi)
    bad_shape["w2"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="w2"):
        _translate(server, 0, {0: bad_shape})


def test_non_finite_values_raise():
    server = make_server(tiny_dataset(n_users=2))
    dphi = _valid_dphi(server, 0, [1])
    dphi["b2"] = torch.tensor([[np.nan]])
    with pytest.raises(ValueError, match="Non-finite"):
        _translate(server, 0, {0: dphi})


def test_duplicate_rows_are_summed_before_vjp():
    server = make_server(tiny_dataset(n_users=2), seed=0)
    s = server.item_mem_num
    a = torch.full((s,), 1.0)
    b = torch.full((s,), 2.0)
    c = torch.full((s,), 3.0)
    dup = _valid_dphi(
        server,
        0,
        [3, 3, 5],
        item_left=torch.stack([a, b, c]),
    )
    coalesced = _valid_dphi(
        server,
        0,
        [3, 5],
        item_left=torch.stack([a + b, c]),
    )
    grad_dup, _ = _translate(server, 0, {0: dup})
    grad_sum, _ = _translate(server, 0, {0: coalesced})
    for name in grad_dup[0]:
        np.testing.assert_allclose(
            grad_dup[0][name].numpy(),
            grad_sum[0][name].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
