"""Shared FedNCF test fixtures: one client and a tiny rating table."""

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
    """Build a CPU FedNCF client.

    Returns a factory: ``make(...) -> (client, request)``. The default is
    one local user and two positive ratings. ``neg_samples=0`` so a
    training step does not draw random negatives.

    Pass ``users_in_client`` and ``training_data`` for more than one user.
    Every id in the training table must appear in ``users_in_client``.
    Embedding rows follow that list's order, not the global id.
    """

    def make(
        optimizer_state_scope="client_resident",
        training_data=None,
        users_in_client=None,
        num_items=6,
        seed=None,
        **kwargs,
    ):
        client = FedNCFClientAlgorithm(
            predictive_factor=8,
            local_epochs=1,
            lr=0.01,
            batch_size=32,
            neg_samples=0,
            optimizer_state_scope=optimizer_state_scope,
            **kwargs,
        )
        client.device = torch.device("cpu")
        if users_in_client is None:
            users_in_client = [0]
        users_in_client = [int(uid) for uid in users_in_client]
        if training_data is None:
            training_data = pd.DataFrame(
                {"user_id": [0, 0], "item_id": [1, 2], "rating": [1.0, 0.5]}
            )
        local, opt_state = client.initialize_local_parameters(
            client_id=0,
            total_num_users=len(users_in_client),
            num_users=len(users_in_client),
            num_items=num_items,
            client_train_data_view=None,
            users_in_client=users_in_client,
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=opt_state
        )
        payload = _clone_groups(client._skeleton.get_server_state_groups())
        request = ClientTrainingRequest(
            client_id=0,
            current_round=0,
            users_in_client=users_in_client,
            client_state=state,
            global_payload=payload,
            training_data=training_data,
            param_ios=default_param_ios(),
            seed=seed,
        )
        return client, request

    return make
