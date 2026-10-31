"""Sparse user-embedding uploads must be well-formed and in-range."""

import numpy as np
import pytest

from src.fedreclab.federated.aggregation import SimpleAvg
from src.fedreclab.federated.utils import ParamIO, RoundStatistics
from src.fedreclab.methods.fedpergnn.fedpergnn import FedPerGNN


def _stub_fedpergnn(table, aggregate=False):
    server = object.__new__(FedPerGNN)
    server.global_parameters = {"user_embeddings": table}
    server.param_ios = {
        "user_embeddings": ParamIO(
            residency="server", exchange="params", apply_to="custom"
        ),
    }
    server.aggregation_strategies = {
        "user_embeddings": SimpleAvg(num_clients=2),
    }
    server.user_embeddings_aggregate = aggregate
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


def test_malformed_user_embedding_update_raises():
    server = _stub_fedpergnn(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Client 0"):
        server._aggregate_user_embeddings([np.ones(4, dtype=np.float32)], [0])


def test_out_of_range_user_id_raises():
    server = _stub_fedpergnn(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(IndexError, match="user_id=5"):
        server._aggregate_user_embeddings(
            [
                {
                    "user_ids": np.array([5]),
                    "embeddings": np.ones(4, dtype=np.float32),
                }
            ],
            [0],
        )
