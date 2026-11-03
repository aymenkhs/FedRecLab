"""Shared MetaMF fixtures: tiny rated dataset and a CPU client factory."""

import pandas as pd
import pytest
import torch

from src.fedreclab.data.client_partitioning import (
    ClientPartition,
    CrossDevicePartitionStrategy,
)
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from src.fedreclab.federated.aggregation import FedAvg
from src.fedreclab.federated.client_selection import FullClientSelection
from src.fedreclab.methods.metamf.metamf import MetaMF, default_param_ios
from src.fedreclab.methods.metamf.metamf_client import MetaMFClientAlgorithm
from src.fedreclab.methods.metamf.metamf_model import MetaRecommender

TINY = dict(
    user_emb_dim=4,
    item_emb_dim=4,
    item_mem_num=2,
    mem_size=8,
    hidden_size=8,
)
NUM_ITEMS = 6


def tiny_dataset(n_users: int = 3) -> InteractionDataset:
    """Build a fully-train rated table, one client per user by default.

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
                "rating": 4.0 + 0.1 * user_id,
                "split": "train",
            }
        )
        rows.append(
            {
                "user_id": user_id,
                "item_id": (user_id + 1) % NUM_ITEMS,
                "rating": 3.0,
                "split": "train",
            }
        )
    present = {row["item_id"] for row in rows}
    for item_id in range(NUM_ITEMS):
        if item_id not in present:
            rows.append(
                {
                    "user_id": 0,
                    "item_id": item_id,
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
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=0,
        include_validation=False,
        include_test=False,
        splits_included=True,
        time_col=None,
    )
    dataset.partition_clients()
    return dataset


def two_user_client_dataset() -> InteractionDataset:
    """Two users on client 0, one user on client 1."""
    dataset = tiny_dataset(n_users=3)
    previous = dataset.client_partition
    train_0 = list(previous.client_to_train_indices[0]) + list(
        previous.client_to_train_indices[1]
    )
    dataset.client_partition = ClientPartition(
        num_clients=2,
        client_ids_2_user_ids={0: [0, 1], 1: [2]},
        user_ids_2_client_ids={0: 0, 1: 0, 2: 1},
        client_to_train_indices={
            0: train_0,
            1: list(previous.client_to_train_indices[2]),
        },
        client_train_sizes={
            0: len(train_0),
            1: int(previous.client_train_sizes[2]),
        },
    )
    dataset.interaction_data["client_id"] = dataset.interaction_data["user_id"].map(
        {0: 0, 1: 0, 2: 1}
    )
    return dataset


@pytest.fixture
def make_tiny_dataset():
    """Return the ``tiny_dataset`` factory."""
    return tiny_dataset


@pytest.fixture
def make_two_user_client_dataset():
    """Return the two-user-on-one-client dataset factory."""
    return two_user_client_dataset


def _generated_payload(
    num_users=1,
    num_items=NUM_ITEMS,
    users=None,
    item_payload="factors",
    seed=0,
    **arch,
):
    cfg = dict(TINY)
    cfg.update(arch)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    model = MetaRecommender(
        num_users=num_users, num_items=num_items, generator=generator, **cfg
    )
    if users is None:
        users = list(range(num_users))
    user_t = torch.tensor(users, dtype=torch.long)
    with torch.no_grad():
        batched = model.generate(user_t, item_payload=item_payload)
    generated = {}
    for index, uid in enumerate(users):
        generated[int(uid)] = {
            key: tensor[index].detach().cpu().clone()
            for key, tensor in batched.items()
        }
    return {"generated_parameters": generated}


@pytest.fixture
def metamf_client_and_request():
    """Build a CPU MetaMF client with one local user and two ratings.

    Returns a factory: ``make(**kwargs) -> (client, request)``.
    """

    def make(**kwargs):
        arch = {key: kwargs.pop(key, TINY[key]) for key in TINY}
        item_payload = kwargs.pop("item_payload", "factors")
        num_items = kwargs.pop("num_items", NUM_ITEMS)
        extra_payload = kwargs.pop("extra_payload", {})
        data = kwargs.pop(
            "training_data",
            pd.DataFrame(
                {"user_id": [0, 0], "item_id": [1, 2], "rating": [4.0, 5.0]}
            ),
        )
        users_in_client = kwargs.pop(
            "users_in_client",
            sorted(int(u) for u in data["user_id"].unique()),
        )
        param_ios = kwargs.pop("param_ios", default_param_ios())
        seed = kwargs.pop("seed", 0)
        payload = kwargs.pop("global_payload", None)
        client = MetaMFClientAlgorithm(
            item_payload=item_payload, **arch, **kwargs
        )
        client.device = torch.device("cpu")
        local = client.initialize_local_parameters(
            client_id=0,
            total_num_users=max(users_in_client) + 1,
            num_users=len(users_in_client),
            num_items=num_items,
            client_train_data_view=data,
            users_in_client=users_in_client,
        )
        state = ClientStateBase(
            client_id=0, local_parameters=local, optimizer_state=None
        )
        if payload is None:
            payload = _generated_payload(
                num_users=max(users_in_client) + 1,
                num_items=num_items,
                users=users_in_client,
                item_payload=item_payload,
                seed=seed,
                **arch,
            )
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


def make_server(
    dataset,
    *,
    aggregator=None,
    selector=None,
    item_payload="factors",
    param_io_overrides=None,
    seed=0,
    **kwargs,
):
    """Build a CPU MetaMF server with the tiny architecture."""
    arch = {key: kwargs.pop(key, TINY[key]) for key in TINY}
    client_kwargs = {
        key: kwargs.pop(key) for key in ("local_batch_size", "local_batch_replacement") if key in kwargs
    }
    client = MetaMFClientAlgorithm(
        item_payload=item_payload, **arch, **client_kwargs
    )
    if aggregator is None:
        aggregator = FedAvg(num_clients=dataset.num_clients)
    if selector is None:
        selector = FullClientSelection(dataset.num_clients)
    return MetaMF(
        item_payload=item_payload,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=seed,
        client_selection_strategy=selector,
        aggregation_strategies=aggregator,
        param_io_overrides=param_io_overrides,
        device="cpu",
        **arch,
        **kwargs,
    )
