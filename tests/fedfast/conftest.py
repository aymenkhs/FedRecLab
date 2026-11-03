"""Shared FedFast fixtures: tiny multi-client train table."""

import pandas as pd
import pytest
import torch

from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from fedreclab.data.dataset import InteractionDataset
from fedreclab.data.split import RatioSplitStrategy
from fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from fedreclab.federated.client_selection import ActvSampClientSelection
from fedreclab.methods.fedfast.fedfast import FedFast, default_param_ios
from fedreclab.methods.fedfast.fedfast_client import FedFastClientAlgorithm


def _clone_groups(groups):
    return {
        g: {k: v.clone() for k, v in tensors.items()} for g, tensors in groups.items()
    }


@pytest.fixture
def fedfast_client_and_request():
    """Build a CPU FedFast client with one local user and two positives.

    Returns a factory: ``make(**kwargs) -> (client, request)``.
    """

    def make(**kwargs):
        embedding_dim = kwargs.pop("embedding_dim", 4)
        num_items = kwargs.pop("num_items", 6)
        client = FedFastClientAlgorithm(
            embedding_dim=embedding_dim,
            local_epochs=1,
            lr=0.01,
            batch_size=32,
            neg_samples=0,
            **kwargs,
        )
        client.device = torch.device("cpu")
        local, opt_state = client.initialize_local_parameters(
            client_id=0,
            total_num_users=1,
            num_users=1,
            num_items=num_items,
            client_train_data_view=None,
            users_in_client=[0],
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=opt_state
        )
        payload = _clone_groups(client._skeleton.get_server_state_groups())
        data = pd.DataFrame(
            {"user_id": [0, 0], "item_id": [1, 2], "rating": [1.0, 1.0]}
        )
        request = ClientTrainingRequest(
            client_id=0,
            current_round=0,
            users_in_client=[0],
            client_state=state,
            global_payload=payload,
            training_data=data,
            param_ios=default_param_ios(),
            seed=0,
        )
        return client, request

    return make


def make_fedfast_dataset(n_clients: int = 4):
    """Cross-device dataset with varied train profiles for ``G0`` clustering."""
    # Multiple rows per some clients so count/entropy/mean rating differ.
    rows = []
    for uid in range(n_clients):
        n_pos = uid + 1
        for j in range(n_pos):
            rows.append(
                {
                    "user_id": uid,
                    "item_id": (uid * 3 + j) % 8,
                    "rating": float(1 + (uid + j) % 5),
                    "split": "train",
                }
            )
    df = pd.DataFrame(rows)
    dataset = InteractionDataset(
        interaction_data=df,
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


def make_fedfast_server(dataset, **kwargs):
    """FedFast server with ActvSAMP and ``actv_agg=True`` by default."""
    n = dataset.num_clients
    p = kwargs.pop("num_clusters", min(2, n))
    m = kwargs.pop("number_selected_clients", p)
    embedding_dim = kwargs.pop("embedding_dim", 4)
    client = FedFastClientAlgorithm(
        embedding_dim=embedding_dim, local_epochs=1, neg_samples=0, lr=0.05
    )
    selector = ActvSampClientSelection(n, m, seed=0)
    return FedFast(
        embedding_dim=embedding_dim,
        num_clusters=p,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=kwargs.pop("num_rounds", 1),
        seed=0,
        client_selection_strategy=selector,
        device="cpu",
        **kwargs,
    )
