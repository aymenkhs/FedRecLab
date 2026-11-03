"""Shared FedPerGNN test fixtures: one client, one user, a tiny rating table."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from src.fedreclab.methods.fedpergnn.fedpergnn import default_param_ios
from src.fedreclab.methods.fedpergnn.fedpergnn_client import FedPerGNNClientAlgorithm


def _expansion(hidden, hist, neigh):
    return {
        0: {
            "history_item_ids": np.array([1, 2] + [-1] * max(hist - 2, 0), dtype=np.int64)[:hist],
            "neighbor_embeddings": np.zeros((hist, neigh, hidden), dtype=np.float32),
            "neighbor_user_ids": np.full((hist, neigh), -1, dtype=np.int64),
        }
    }


@pytest.fixture
def fedpergnn_client_and_request():
    """Build a CPU FedPerGNN client with one local user and two positives.

    Returns a factory: ``make(task=..., **kwargs) -> (client, request)``.
    """

    def make(task="rating", **kwargs):
        hidden, hist, neigh = 8, 4, 3
        client = FedPerGNNClientAlgorithm(
            hidden_dim=hidden,
            history_len=hist,
            neighbor_len=neigh,
            local_epochs=1,
            lr=0.01,
            batch_size=32,
            task=task,
            **kwargs,
        )
        client.device = torch.device("cpu")
        local, opt_state = client.initialize_local_parameters(
            client_id=0,
            total_num_users=1,
            num_users=1,
            num_items=6,
            client_train_data_view=None,
            users_in_client=[0],
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=opt_state
        )
        data = pd.DataFrame(
            {"user_id": [0, 0], "item_id": [1, 2], "rating": [4.0, 1.0]}
        )
        request = ClientTrainingRequest(
            client_id=0,
            current_round=0,
            users_in_client=[0],
            client_state=state,
            global_payload={
                "model_state": client._skeleton.get_model_state(),
                "item_embeddings": client._skeleton.get_item_embeddings(),
            },
            training_data=data,
            param_ios=default_param_ios(),
            extra_payload={"expanded_user_data": _expansion(hidden, hist, neigh)},
            seed=0,
        )
        return client, request

    return make
