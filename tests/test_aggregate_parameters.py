"""Aggregation loop: full participation, missing groups, unknown names."""

import numpy as np
import pytest
import torch

from fedreclab.federated.aggregation import BaseAggregation, Sum
from fedreclab.federated.base_client import ClientStateBase, ClientTrainingResponse
from fedreclab.federated.base_server import BaseFederatedServer
from fedreclab.federated.utils import ParamIO, RoundStatistics, UpdatePackage


class RecordingAggregator(BaseAggregation):
    """Record ``round_info.selected`` and ``updates``; return the mean."""

    def __init__(self):
        super().__init__(num_clients=2)
        self.calls = []

    def aggregate(self, round_info, previous_parameters, updates, param_io):
        self.calls.append(
            {
                "selected": list(round_info.selected),
                "updates": updates,
            }
        )
        if isinstance(updates, np.ndarray):
            return updates.mean(axis=0)
        return updates[0]


def _stub_server(param_ios, global_parameters, aggregators, selected):
    server = object.__new__(BaseFederatedServer)
    server.param_ios = param_ios
    server.global_parameters = global_parameters
    server.client_states = {}
    server.aggregation_strategies = aggregators
    server.training_framework = "numpy"
    server.device = None
    server.current_round_statistics = RoundStatistics(
        current_round=0,
        selected=list(selected),
        responded=list(selected),
        dropped=[],
        num_training_examples={cid: 1 for cid in selected},
        training_losses={cid: [] for cid in selected},
        training_durations={cid: 0.0 for cid in selected},
    )
    return server


def _response(client_id, packages, param_ios):
    dummy_io = ParamIO(residency="server", exchange="params", apply_to="server")
    return ClientTrainingResponse(
        client_id=client_id,
        updated_client_state=ClientStateBase(
            client_id=client_id, local_parameters={}
        ),
        updates_packages={
            name: UpdatePackage(
                param_name=name,
                param_io=param_ios.get(name, dummy_io),
                param_value=value,
                param_metadata={},
            )
            for name, value in packages.items()
        },
        training_losses=[],
        training_duration=0.0,
        num_training_examples=1,
    )


def test_full_participation_passes_selected_and_updates_in_response_order():
    """Every client sends every group: same selected list and stacked updates as the old loop."""
    w_io = ParamIO(residency="server", exchange="params", apply_to="server")
    v_io = ParamIO(residency="server", exchange="params", apply_to="server")
    param_ios = {"w": w_io, "v": v_io}
    agg_w = RecordingAggregator()
    agg_v = RecordingAggregator()
    w0 = np.array([1.0, 2.0], dtype=np.float32)
    w1 = np.array([3.0, 4.0], dtype=np.float32)
    v0 = np.array([5.0, 6.0], dtype=np.float32)
    v1 = np.array([7.0, 8.0], dtype=np.float32)
    server = _stub_server(
        param_ios=param_ios,
        global_parameters={
            "w": np.zeros(2, dtype=np.float32),
            "v": np.zeros(2, dtype=np.float32),
        },
        aggregators={"w": agg_w, "v": agg_v},
        selected=[0, 1],
    )
    responses = [
        _response(0, {"w": w0, "v": v0}, param_ios),
        _response(1, {"w": w1, "v": v1}, param_ios),
    ]

    server.aggregate_parameters(responses)

    assert agg_w.calls[0]["selected"] == [0, 1]
    assert agg_v.calls[0]["selected"] == [0, 1]
    np.testing.assert_array_equal(agg_w.calls[0]["updates"], np.stack([w0, w1]))
    np.testing.assert_array_equal(agg_v.calls[0]["updates"], np.stack([v0, v1]))


def test_missing_group_aggregates_only_clients_that_sent_it():
    param_io = ParamIO(residency="server", exchange="params", apply_to="server")
    param_ios = {"w": param_io}
    agg = RecordingAggregator()
    w0 = np.array([1.0, 2.0], dtype=np.float32)
    server = _stub_server(
        param_ios=param_ios,
        global_parameters={"w": np.zeros(2, dtype=np.float32)},
        aggregators={"w": agg},
        selected=[0, 1],
    )
    responses = [
        _response(0, {"w": w0}, param_ios),
        _response(1, {}, param_ios),
    ]

    server.aggregate_parameters(responses)

    assert agg.calls[0]["selected"] == [0]
    np.testing.assert_array_equal(agg.calls[0]["updates"], np.stack([w0]))


def test_group_sent_by_nobody_is_skipped():
    param_io = ParamIO(residency="server", exchange="params", apply_to="server")
    param_ios = {"w": param_io}
    agg = RecordingAggregator()
    server = _stub_server(
        param_ios=param_ios,
        global_parameters={"w": np.zeros(2, dtype=np.float32)},
        aggregators={"w": agg},
        selected=[0, 1],
    )
    responses = [
        _response(0, {}, param_ios),
        _response(1, {}, param_ios),
    ]

    server.aggregate_parameters(responses)

    assert agg.calls == []


def test_unknown_package_name_raises():
    param_io = ParamIO(residency="server", exchange="params", apply_to="server")
    param_ios = {"w": param_io}
    agg = RecordingAggregator()
    w0 = np.array([1.0, 2.0], dtype=np.float32)
    server = _stub_server(
        param_ios=param_ios,
        global_parameters={"w": np.zeros(2, dtype=np.float32)},
        aggregators={"w": agg},
        selected=[0],
    )
    responses = [_response(0, {"w": w0, "typo": w0}, param_ios)]

    with pytest.raises(KeyError, match="typo"):
        server.aggregate_parameters(responses)


def _dense_dict_server(exchange, server_learning_rate=1.0):
    param_io = ParamIO(
        residency="server", exchange=exchange, apply_to="server", layout="dense"
    )
    server = _stub_server(
        param_ios={"g": param_io},
        global_parameters={
            "g": {"w": torch.ones(2, 2), "b": torch.zeros(2)},
        },
        aggregators={"g": Sum(num_clients=2)},
        selected=[0, 1],
    )
    server.training_framework = "torch"
    server.server_learning_rate = server_learning_rate
    return server, param_io


def _dense_dict_updates():
    return [
        {"w": torch.full((2, 2), 0.5), "b": torch.ones(2)},
        {"w": torch.full((2, 2), 0.25), "b": torch.ones(2)},
    ]


def test_dense_dict_group_gradient_exchange_steps_each_key():
    server, param_io = _dense_dict_server("gradient", server_learning_rate=2.0)

    server._aggregate_group("g", param_io, _dense_dict_updates(), [0, 1])

    new = server.global_parameters["g"]
    assert set(new) == {"w", "b"}
    torch.testing.assert_close(new["w"], torch.full((2, 2), 1.0 - 2.0 * 0.75))
    torch.testing.assert_close(new["b"], torch.full((2,), -4.0))


def test_dense_dict_group_delta_exchange_adds_each_key():
    server, param_io = _dense_dict_server("delta")

    server._aggregate_group("g", param_io, _dense_dict_updates(), [0, 1])

    new = server.global_parameters["g"]
    torch.testing.assert_close(new["w"], torch.full((2, 2), 1.75))
    torch.testing.assert_close(new["b"], torch.full((2,), 2.0))


def test_dense_dict_group_mismatched_keys_raise():
    server, param_io = _dense_dict_server("delta")

    with pytest.raises(ValueError, match="do not match parameter keys"):
        server._apply_exchange_to_value(
            old_value=server.global_parameters["g"],
            aggregated_result={"w": torch.zeros(2, 2)},
            param_io=param_io,
            param_name="g",
        )


def test_dense_dict_group_against_tensor_result_raises():
    server, param_io = _dense_dict_server("gradient")

    with pytest.raises(TypeError, match="cannot apply exchange='gradient'"):
        server._apply_exchange_to_value(
            old_value=server.global_parameters["g"],
            aggregated_result=torch.zeros(2, 2),
            param_io=param_io,
            param_name="g",
        )


def test_convert_to_framework_compatible_is_protected():
    param_io = ParamIO(residency="server", exchange="params", apply_to="server")
    server = _stub_server(
        param_ios={"w": param_io},
        global_parameters={"w": np.zeros(2, dtype=np.float32)},
        aggregators={"w": RecordingAggregator()},
        selected=[0],
    )

    stacked = server._convert_to_framework_compatible(
        [np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    )

    assert stacked.shape == (2, 2)
    np.testing.assert_array_equal(
        stacked, np.stack([np.ones(2, dtype=np.float32), np.zeros(2)])
    )
