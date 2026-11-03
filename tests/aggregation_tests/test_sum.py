"""Sum aggregator: dense and row paths, type preservation, zeros vs omit."""

import numpy as np
import pytest
import torch

from fedreclab.federated.aggregation import Sum
from fedreclab.federated.base_client import ClientStateBase, ClientTrainingResponse
from fedreclab.federated.base_server import BaseFederatedServer
from fedreclab.federated.utils import ParamIO, RoundStatistics, UpdatePackage


def _info(selected=(0, 1)):
    selected = list(selected)
    return RoundStatistics(
        current_round=0,
        selected=selected,
        responded=selected,
        dropped=[],
        num_training_examples={cid: 1 for cid in selected},
        training_losses={cid: [] for cid in selected},
        training_durations={cid: 0.0 for cid in selected},
    )


def _pio(exchange="gradient", layout="dense"):
    return ParamIO(
        residency="server",
        exchange=exchange,
        apply_to="server",
        layout=layout,
    )


def test_dense_numpy_sums_the_client_axis():
    out = Sum(2).aggregate(
        _info(), None, [np.array([1.0, 2.0]), np.array([3.0, 4.0])], _pio()
    )
    np.testing.assert_allclose(out, [4.0, 6.0])


def test_dict_of_tensors_sums_each_key():
    updates = [
        {"w": np.array([1.0]), "b": np.array([10.0])},
        {"w": np.array([2.0]), "b": np.array([20.0])},
    ]
    out = Sum(2).aggregate(_info(), None, updates, _pio())
    np.testing.assert_allclose(out["w"], [3.0])
    np.testing.assert_allclose(out["b"], [30.0])


def test_torch_preserves_dtype_and_device():
    updates = [
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.5, 2.0], dtype=torch.float32),
    ]
    out = Sum(2).aggregate(_info(), None, updates, _pio())
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    np.testing.assert_allclose(out.detach().cpu().numpy(), [1.5, 2.0])


def test_empty_updates_fail_loudly():
    with pytest.raises(ValueError, match="No client updates"):
        Sum(2).aggregate(_info(), None, [], _pio())


def test_non_finite_values_fail_loudly():
    with pytest.raises(ValueError, match="Non-finite"):
        Sum(2).aggregate(
            _info(),
            None,
            [np.array([1.0]), np.array([np.nan])],
            _pio(),
        )


def test_row_path_sums_only_contributors():
    values = np.array([[[1.0]], [[100.0]], [[3.0]]], dtype=np.float64)
    mask = np.array([[True], [False], [True]])
    out = Sum(3).aggregate_rows(
        _info([0, 1, 2]), np.zeros((1, 1)), values, mask, _pio()
    )
    np.testing.assert_allclose(out[0], [4.0])


def test_row_zeros_match_omitting_the_row():
    """FullText zeros and PartText omission are the same under a sum."""
    pio = _pio()
    dense = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 3.0]],
        ],
        dtype=np.float64,
    )
    dense_mask = np.array([[True, True], [True, True]])
    sparse = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 3.0]],
        ],
        dtype=np.float64,
    )
    sparse_mask = np.array([[True, False], [False, True]])
    prev = np.zeros((2, 2))
    full = Sum(2).aggregate_rows(_info(), prev, dense, dense_mask, pio)
    part = Sum(2).aggregate_rows(_info(), prev, sparse, sparse_mask, pio)
    np.testing.assert_allclose(full, part)
    np.testing.assert_allclose(full, [[1.0, 0.0], [2.0, 3.0]])


def test_row_without_a_contributor_fails():
    values = np.array([[[1.0], [0.0]], [[2.0], [0.0]]], dtype=np.float64)
    mask = np.array([[True, False], [True, False]])
    with pytest.raises(ValueError, match="no contributing client"):
        Sum(2).aggregate_rows(_info(), np.zeros((2, 1)), values, mask, _pio())


def _stub_server(param_io, table, aggregator, selected):
    server = object.__new__(BaseFederatedServer)
    server.param_ios = {"items": param_io}
    server.global_parameters = {"items": np.array(table, copy=True)}
    server.client_states = {}
    server.aggregation_strategies = {"items": aggregator}
    server.training_framework = "numpy"
    server.device = None
    server.current_round_statistics = _info(selected)
    return server


def _response(client_id, value, param_io):
    return ClientTrainingResponse(
        client_id=client_id,
        updated_client_state=ClientStateBase(client_id=client_id, local_parameters={}),
        updates_packages={
            "items": UpdatePackage(
                param_name="items",
                param_io=param_io,
                param_value=value,
                param_metadata={},
            )
        },
        training_losses=[],
        training_duration=0.0,
        num_training_examples=1,
    )


def test_sum_row_gradient_subtracts_the_contributor_sum():
    pio = ParamIO(
        residency="server",
        exchange="gradient",
        apply_to="server",
        layout="rows",
    )
    prev = np.array([[1.0], [2.0]], dtype=np.float32)
    server = _stub_server(pio, prev, Sum(2), [0, 1])
    responses = [
        _response(0, {"rows": np.array([0]), "values": np.array([[0.2]])}, pio),
        _response(1, {"rows": np.array([0]), "values": np.array([[0.3]])}, pio),
    ]
    server.aggregate_parameters(responses)
    np.testing.assert_allclose(server.global_parameters["items"][0], [0.5], atol=1e-6)
    np.testing.assert_allclose(server.global_parameters["items"][1], [2.0], atol=1e-6)
