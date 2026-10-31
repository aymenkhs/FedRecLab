"""Shared FedNCF test fixtures: one client, one user, a tiny rating table."""

import pandas as pd
import pytest
import torch

from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from src.fedreclab.methods.fedncf.fedncf import default_param_ios
from src.fedreclab.methods.fedncf.fedncf_client import FedNCFClientAlgorithm


def _clone_groups(groups):
    return {g: {k: v.clone() for k, v in tensors.items()} for g, tensors in groups.items()}


@pytest.fixture
def fedncf_client_and_request():
    """Build a CPU FedNCF client with one local user and two positive ratings.

    Returns a factory: ``make(optimizer_state_scope=...) -> (client, request)``.
    ``neg_samples=0`` so a training step does not draw random negatives.
    """

    def make(optimizer_state_scope="client_resident", **kwargs):
        client = FedNCFClientAlgorithm(
            predictive_factor=8,
            local_epochs=1,
            lr=0.01,
            batch_size=32,
            neg_samples=0,
            optimizer_state_scope=optimizer_state_scope,
            **kwargs,
        )
        client._device = torch.device("cpu")
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
        payload = _clone_groups(client._skeleton.get_server_state_groups())
        data = pd.DataFrame(
            {"user_id": [0, 0], "item_id": [1, 2], "rating": [1.0, 0.5]}
        )
        request = ClientTrainingRequest(
            client_id=0,
            current_round=0,
            users_in_client=[0],
            client_state=state,
            global_payload=payload,
            training_data=data,
            param_ios=default_param_ios(),
        )
        return client, request

    return make
