"""Shared SemiDFEGL fixtures: tiny catalogue, one user per client, CPU."""

import pandas as pd
import pytest

from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from fedreclab.data.dataset import InteractionDataset
from fedreclab.data.split import RatioSplitStrategy
from fedreclab.federated.aggregation import FedAvg
from fedreclab.federated.client_selection import FullClientSelection
from fedreclab.methods.semidfegl.semidfegl import SemiDFEGL
from fedreclab.methods.semidfegl.semidfegl_client import SemiDFEGLClientAlgorithm

EMBEDDING_DIM = 4
NUM_ITEMS = 8
NUM_LAYERS = 1


def tiny_dataset(n_users: int = 4) -> InteractionDataset:
    """Build a fully-train table with two positives per user.

    User ``u`` has items ``2u`` and ``2u + 1`` modulo ``NUM_ITEMS``.

    Parameters
    ----------
    n_users : int, optional
        Number of users (and clients). Default 4.

    Returns
    -------
    InteractionDataset
        Partitioned dataset.
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
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=False
        ),
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
    """Return a factory for a CPU SemiDFEGL server on a tiny dataset.

    ``make(dataset=None, client=None, client_kwargs=None, **server_kwargs)``.
    Defaults: ``FullClientSelection``, ``FedAvg``, ``seed=0``,
    ``num_groups=2``, ``n_fake=1``, ``num_layers=1``, ``embedding_dim=4``.
    """

    def make(dataset=None, client=None, client_kwargs=None, **server_kwargs):
        if dataset is None:
            dataset = tiny_dataset()
        if client is None:
            client = SemiDFEGLClientAlgorithm(
                embedding_dim=EMBEDDING_DIM,
                num_layers=NUM_LAYERS,
                **(client_kwargs or {}),
            )
        kwargs = dict(
            embedding_dim=EMBEDDING_DIM,
            num_groups=2,
            n_fake=1,
            num_layers=NUM_LAYERS,
            interaction_dataset=dataset,
            client_algorithm=client,
            num_rounds=2,
            seed=0,
            client_selection_strategy=FullClientSelection(dataset.num_clients),
            aggregation_strategies=FedAvg(num_clients=dataset.num_clients),
            device="cpu",
        )
        kwargs.update(server_kwargs)
        return SemiDFEGL(**kwargs)

    return make
