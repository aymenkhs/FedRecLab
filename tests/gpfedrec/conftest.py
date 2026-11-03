"""Shared GPFedRec fixtures: tiny catalogue, one user per client, CPU device."""

import pandas as pd
import pytest
import torch

from src.fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.aggregation import SimpleAvg
from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from src.fedreclab.federated.client_selection import FullClientSelection
from src.fedreclab.methods.gpfedrec.gpfedrec import GPFedRec, default_param_ios
from src.fedreclab.methods.gpfedrec.gpfedrec_client import GPFedRecClientAlgorithm
from src.fedreclab.methods.gpfedrec.gpfedrec_model import initial_item_table

LATENT = 4
HIDDEN = (8, 4)
NUM_ITEMS = 8


def tiny_dataset(n_users: int = 3) -> InteractionDataset:
    """Build a fully-train table with two positives per user, one client per user.

    User ``u`` has items ``2u`` and ``2u + 1`` modulo ``NUM_ITEMS``, so no
    user has every item and negatives always exist.

    Parameters
    ----------
    n_users : int, optional
        Number of users (and clients). Default is 3.

    Returns
    -------
    InteractionDataset
        Partitioned dataset with ``n_users`` clients and
        ``min(2 * n_users, NUM_ITEMS)`` items.
    """
    rows = [
        {
            "user_id": user_id,
            "item_id": (2 * user_id + offset) % NUM_ITEMS,
            "rating": 1.0,
            "split": "train",
        }
        for user_id in range(n_users)
        for offset in (0, 1)
    ]
    dataset = InteractionDataset(
        interaction_data=pd.DataFrame(rows),
        splitting_strategy=RatioSplitStrategy(include_validation=False, include_test=False),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
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
    """Return the ``tiny_dataset`` factory."""
    return tiny_dataset


@pytest.fixture
def make_server():
    """Return a factory for a CPU GPFedRec server on a tiny dataset.

    ``make(dataset=None, client=None, client_kwargs=None, server_class=GPFedRec,
    **server_kwargs)``. Defaults: ``FullClientSelection``, ``SimpleAvg``,
    ``seed=0``.
    """

    def make(dataset=None, client=None, client_kwargs=None, server_class=GPFedRec, **server_kwargs):
        if dataset is None:
            dataset = tiny_dataset()
        if client is None:
            client = GPFedRecClientAlgorithm(
                latent_dim=LATENT, hidden_layers=HIDDEN, **(client_kwargs or {})
            )
        kwargs = dict(
            latent_dim=LATENT,
            interaction_dataset=dataset,
            client_algorithm=client,
            num_rounds=1,
            seed=0,
            client_selection_strategy=FullClientSelection(dataset.num_clients),
            aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
            device="cpu",
        )
        kwargs.update(server_kwargs)
        return server_class(**kwargs)

    return make


@pytest.fixture
def make_client_and_request():
    """Return a factory for a CPU client with one user and its training request.

    ``make(global_payload=None, seed=0, current_round=0, **client_kwargs)
    -> (client, request)``. The default payload is round 0: one seeded
    table sent as both ``item_embeddings`` and
    ``personalized_item_embeddings`` (the same object).
    """

    def make(global_payload=None, seed=0, current_round=0, training_data=None, **client_kwargs):
        client_kwargs.setdefault("init_seed", 0)
        client = GPFedRecClientAlgorithm(
            latent_dim=LATENT, hidden_layers=HIDDEN, **client_kwargs
        )
        client.device = torch.device("cpu")
        if training_data is None:
            training_data = pd.DataFrame(
                {"user_id": [0, 0], "item_id": [1, 2], "rating": [1.0, 1.0]}
            )
        local, optimizer_state = client.initialize_local_parameters(
            client_id=0,
            total_num_users=1,
            num_users=1,
            num_items=NUM_ITEMS,
            client_train_data_view=training_data,
            users_in_client=[0],
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=optimizer_state
        )
        if global_payload is None:
            q = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=0)
            global_payload = {"item_embeddings": q, "personalized_item_embeddings": q}
        request = ClientTrainingRequest(
            client_id=0,
            current_round=current_round,
            users_in_client=[0],
            client_state=state,
            global_payload=global_payload,
            training_data=training_data,
            param_ios=default_param_ios(),
            extra_payload={},
            seed=seed,
        )
        return client, request

    return make
