"""Sparse user-embedding uploads go through the generic row path."""

import numpy as np
import pytest

from src.fedreclab.federated.aggregation import SimpleAvg
from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingResponse
from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import RoundStatistics, UpdatePackage
from src.fedreclab.methods.fedpergnn.fedpergnn import default_param_ios


def _stub(table):
    server = object.__new__(BaseFederatedServer)
    server.param_ios = default_param_ios()
    server.global_parameters = {
        "model_state": {},
        "item_embeddings": np.zeros((2, 4), dtype=np.float32),
        "user_embeddings": np.array(table, copy=True),
    }
    server.client_states = {}
    n = 2
    server.aggregation_strategies = {
        name: SimpleAvg(n) for name in server.param_ios
    }
    server.training_framework = "numpy"
    server.device = None
    server.current_round_statistics = RoundStatistics(
        current_round=0,
        selected=[0],
        responded=[0],
        dropped=[],
        num_training_examples={0: 1},
        training_losses={0: []},
        training_durations={0: 0.0},
    )
    return server


def _user_response(client_id, rows, values):
    pio = default_param_ios()["user_embeddings"]
    return ClientTrainingResponse(
        client_id=client_id,
        updated_client_state=ClientStateBase(client_id=client_id, local_parameters={}),
        updates_packages={
            "user_embeddings": UpdatePackage(
                param_name="user_embeddings",
                param_io=pio,
                param_value={"rows": np.asarray(rows), "values": np.asarray(values)},
                param_metadata={},
            )
        },
        training_losses=[],
        training_duration=0.0,
        num_training_examples=1,
    )


def test_malformed_user_embedding_update_raises():
    server = _stub(np.zeros((2, 4), dtype=np.float32))
    resp = _user_response(0, [0], np.ones(4, dtype=np.float32))
    resp.updates_packages["user_embeddings"].param_value = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="layout='rows'"):
        server.aggregate_parameters([resp])


def test_out_of_range_user_id_raises():
    server = _stub(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(IndexError, match="outside"):
        server.aggregate_parameters([_user_response(0, [5], np.ones((1, 4), dtype=np.float32))])


def test_two_clients_equal_mix_on_a_shared_user_row():
    table = np.zeros((2, 2), dtype=np.float32)
    server = _stub(table)
    server.current_round_statistics = RoundStatistics(
        current_round=0,
        selected=[0, 1],
        responded=[0, 1],
        dropped=[],
        num_training_examples={0: 1, 1: 1},
        training_losses={0: [], 1: []},
        training_durations={0: 0.0, 1: 0.0},
    )
    responses = [
        _user_response(0, [0], [[1.0, 1.0]]),
        _user_response(1, [0], [[3.0, 3.0]]),
    ]
    server.aggregate_parameters(responses)
    np.testing.assert_allclose(server.global_parameters["user_embeddings"][0], [2.0, 2.0])
    np.testing.assert_allclose(server.global_parameters["user_embeddings"][1], [0.0, 0.0])
