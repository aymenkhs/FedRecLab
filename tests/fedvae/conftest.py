"""Shared FedVAE fixtures: tiny catalogue, one-user client, CPU device."""

import pandas as pd
import pytest
import torch

from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from src.fedreclab.methods.fedvae.fedvae import default_param_ios
from src.fedreclab.methods.fedvae.fedvae_client import FedVAEClientAlgorithm
from src.fedreclab.methods.fedvae.fedvae_model import MultVAE

HIDDEN = (4,)
LATENT = 2
NUM_ITEMS = 6


def tiny_dataset(n_users: int = 3) -> InteractionDataset:
    """Build a fully-train interaction table, one client per user.

    Parameters
    ----------
    n_users : int, optional
        Number of users (and clients). Default is 3.

    Returns
    -------
    InteractionDataset
        Partitioned dataset with ``n_users`` clients and ``NUM_ITEMS`` items.
    """
    rows = []
    for user_id in range(n_users):
        rows.append(
            {
                "user_id": user_id,
                "item_id": user_id % NUM_ITEMS,
                "rating": 1.0,
                "split": "train",
            }
        )
        rows.append(
            {
                "user_id": user_id,
                "item_id": (user_id + 1) % NUM_ITEMS,
                "rating": 1.0,
                "split": "train",
            }
        )
    df = pd.DataFrame(rows)
    dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=False
        ),
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
        k_core=0,
        include_validation=False,
        include_test=False,
        splits_included=True,
        time_col=None,
    )
    dataset.partition_clients()
    return dataset


@pytest.fixture
def make_tiny_dataset():
    """Return the ``tiny_dataset`` factory for server tests."""
    return tiny_dataset


def _payload(num_items=NUM_ITEMS, hidden_dims=HIDDEN, latent_dim=LATENT, seed=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    model = MultVAE(
        num_items=num_items,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=0.5,
        generator=generator,
    )
    return model.get_server_state_groups()


@pytest.fixture
def fedvae_client_and_request():
    """Build a CPU FedVAE client with one local user and two positives.

    Returns a factory: ``make(**kwargs) -> (client, request)``.
    """

    def make(**kwargs):
        hidden_dims = kwargs.pop("hidden_dims", HIDDEN)
        latent_dim = kwargs.pop("latent_dim", LATENT)
        num_items = kwargs.pop("num_items", NUM_ITEMS)
        extra_payload = kwargs.pop("extra_payload", {})
        data = kwargs.pop(
            "training_data",
            pd.DataFrame(
                {"user_id": [0, 0], "item_id": [1, 2], "rating": [1.0, 1.0]}
            ),
        )
        users_in_client = kwargs.pop("users_in_client", [0])
        param_ios = kwargs.pop("param_ios", default_param_ios())
        seed = kwargs.pop("seed", 0)
        payload = kwargs.pop("global_payload", None)
        client = FedVAEClientAlgorithm(
            hidden_dims=hidden_dims, latent_dim=latent_dim, **kwargs
        )
        client.device = torch.device("cpu")
        local = client.initialize_local_parameters(
            client_id=0,
            total_num_users=1,
            num_users=len(users_in_client),
            num_items=num_items,
            client_train_data_view=data,
            users_in_client=users_in_client,
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=None
        )
        if payload is None:
            payload = _payload(num_items, hidden_dims, latent_dim, seed=seed)
        request = ClientTrainingRequest(
            client_id=0,
            current_round=0,
            users_in_client=users_in_client,
            client_state=state,
            global_payload=payload,
            training_data=data,
            param_ios=param_ios,
            extra_payload=extra_payload,
            seed=seed,
        )
        return client, request

    return make
