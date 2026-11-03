"""Shared FedMF fixtures: one client, one user, a tiny rating table."""

import numpy as np
import pandas as pd
import pytest
import torch

from fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from fedreclab.methods.fedmf.fedmf import default_param_ios
from fedreclab.methods.fedmf.fedmf_client import FedMFClientAlgorithm


def _payload(num_items, latent_dim, fill=0.01):
    return {
        "item_embeddings": {
            "item_embeddings.weight": torch.full(
                (num_items, latent_dim), fill, dtype=torch.float32
            )
        }
    }


@pytest.fixture
def fedmf_client_and_request():
    """Build a CPU FedMF client; by default one local user and two ratings.

    Returns a factory: ``make(**kwargs) -> (client, request)``. The client
    owns every user present in ``training_data`` unless ``users_in_client``
    is given.
    """

    def make(**kwargs):
        latent_dim = kwargs.pop("latent_dim", 4)
        num_items = kwargs.pop("num_items", 6)
        fill = kwargs.pop("item_fill", 0.01)
        extra_payload = kwargs.pop("extra_payload", {"n_train_ratings": 2})
        data = kwargs.pop(
            "training_data",
            pd.DataFrame(
                {"user_id": [0, 0], "item_id": [1, 2], "rating": [4.0, 5.0]}
            ),
        )
        param_ios = kwargs.pop("param_ios", default_param_ios())
        users = kwargs.pop(
            "users_in_client", sorted(int(u) for u in data["user_id"].unique())
        )
        client = FedMFClientAlgorithm(latent_dim=latent_dim, **kwargs)
        client.device = torch.device("cpu")
        local = client.initialize_local_parameters(
            client_id=0,
            total_num_users=len(users),
            num_users=len(users),
            num_items=num_items,
            client_train_data_view=data,
            users_in_client=users,
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=None
        )
        request = ClientTrainingRequest(
            client_id=0,
            current_round=0,
            users_in_client=users,
            client_state=state,
            global_payload=_payload(num_items, latent_dim, fill),
            training_data=data,
            param_ios=param_ios,
            extra_payload=extra_payload,
        )
        return client, request

    return make
