"""Row-layout aggregation: no 1/n dilution, layout checks, Krum fallback counts."""

import numpy as np
import pytest

from fedreclab.federated.aggregation import FedAvg, Krum, Median, SimpleAvg
from fedreclab.federated.base_client import ClientStateBase, ClientTrainingResponse
from fedreclab.federated.base_server import BaseFederatedServer
from fedreclab.federated.utils import ParamIO, RoundStatistics, UpdatePackage


def _round_info(selected, examples=None):
    return RoundStatistics(
        current_round=0,
        selected=list(selected),
        responded=list(selected),
        dropped=[],
        num_training_examples={cid: (examples or {}).get(cid, 1) for cid in selected},
        training_losses={cid: [] for cid in selected},
        training_durations={cid: 0.0 for cid in selected},
    )


def _stub_server(param_io, table, aggregator, selected):
    server = object.__new__(BaseFederatedServer)
    server.param_ios = {"items": param_io}
    server.global_parameters = {"items": np.array(table, copy=True)}
    server.client_states = {}
    server.aggregation_strategies = {"items": aggregator}
    server.training_framework = "numpy"
    server.device = None
    server.current_round_statistics = _round_info(selected)
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


def test_simpleavg_row_is_not_diluted_by_clients_that_did_not_touch_it():
    """Paper example: one of two clients updates a row; the row must keep that update."""
    pio = ParamIO(residency="server", exchange="params", apply_to="server", layout="rows")
    prev = np.array([[0.04], [0.10]], dtype=np.float32)
    agg = SimpleAvg(num_clients=2)
    server = _stub_server(pio, prev, agg, [0, 1])
    responses = [
        _response(0, {"rows": np.array([0]), "values": np.array([[0.047]])}, pio),
        _response(1, {"rows": np.array([1]), "values": np.array([[0.20]])}, pio),
    ]

    server.aggregate_parameters(responses)

    np.testing.assert_allclose(server.global_parameters["items"][0], [0.047], atol=1e-7)
    np.testing.assert_allclose(server.global_parameters["items"][1], [0.20], atol=1e-7)


def test_dense_layout_rejects_sparse_upload():
    pio = ParamIO(residency="server", exchange="params", apply_to="server", layout="dense")
    server = _stub_server(pio, np.zeros((2, 1), dtype=np.float32), SimpleAvg(2), [0])
    responses = [
        _response(0, {"rows": np.array([0]), "values": np.array([[1.0]])}, pio),
    ]
    with pytest.raises(ValueError, match="layout='dense'"):
        server.aggregate_parameters(responses)


def test_rows_layout_rejects_full_table_upload():
    pio = ParamIO(residency="server", exchange="params", apply_to="server", layout="rows")
    server = _stub_server(pio, np.zeros((2, 1), dtype=np.float32), SimpleAvg(2), [0])
    responses = [_response(0, np.ones((2, 1), dtype=np.float32), pio)]
    with pytest.raises(ValueError, match="layout='rows'"):
        server.aggregate_parameters(responses)


def test_delta_exchange_leaves_untouched_rows():
    pio = ParamIO(residency="server", exchange="delta", apply_to="server", layout="rows")
    prev = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    server = _stub_server(pio, prev, SimpleAvg(2), [0])
    responses = [
        _response(0, {"rows": np.array([0]), "values": np.array([[0.5, 0.5]])}, pio),
    ]
    server.aggregate_parameters(responses)
    np.testing.assert_allclose(server.global_parameters["items"][0], [1.5, 1.5])
    np.testing.assert_allclose(server.global_parameters["items"][1], [2.0, 2.0])


def test_fedavg_row_weights_by_example_counts():
    pio = ParamIO(residency="server", exchange="params", apply_to="server")
    values = np.array([[[1.0]], [[3.0]]], dtype=np.float64)
    mask = np.array([[True], [True]])
    prev = np.array([[0.0]])
    info = _round_info([0, 1], examples={0: 1, 1: 3})
    out = FedAvg(2).aggregate_rows(info, prev, values, mask, pio)
    np.testing.assert_allclose(out[0], [2.5])


def test_median_rows_ignores_clients_without_the_row():
    pio = ParamIO(residency="server", exchange="params", apply_to="server")
    values = np.array([[[1.0]], [[100.0]], [[3.0]]], dtype=np.float64)
    mask = np.array([[True], [False], [True]])
    out = Median(3).aggregate_rows(_round_info([0, 1, 2]), np.zeros((1, 1)), values, mask, pio)
    np.testing.assert_allclose(out[0], [2.0])


def test_krum_row_fallback_is_counted():
    pio = ParamIO(residency="server", exchange="params", apply_to="server", layout="rows")
    prev = np.zeros((1, 2), dtype=np.float32)
    agg = Krum(num_clients=2, num_byzantine=1)
    server = _stub_server(pio, prev, agg, [0, 1])
    responses = [
        _response(0, {"rows": np.array([0]), "values": np.array([[1.0, 1.0]])}, pio),
        _response(1, {"rows": np.array([0]), "values": np.array([[3.0, 3.0]])}, pio),
    ]
    server.aggregate_parameters(responses)
    assert server.current_round_statistics.row_aggregator_fallbacks["items"] == 1
    np.testing.assert_allclose(server.global_parameters["items"][0], [2.0, 2.0])
