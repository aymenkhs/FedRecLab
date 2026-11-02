"""FedMF init, extra_payload M, and one Sum round."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.aggregation import Sum
from src.fedreclab.federated.client_selection import FullClientSelection
from src.fedreclab.methods.fedmf.fedmf import FedMF
from src.fedreclab.methods.fedmf.fedmf_client import FedMFClientAlgorithm
from src.fedreclab.methods.fedmf.fedmf_model import MatrixFactorization


def _tiny_dataset():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 2],
            "rating": [4.0, 5.0, 3.0, 2.0],
            "split": ["train", "train", "train", "train"],
        }
    )
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


def test_constant_init_is_0_01():
    m = MatrixFactorization(num_items=3, latent_dim=2, init_mode="constant", init_value=0.01)
    np.testing.assert_allclose(m.item_embeddings.weight.detach().numpy(), 0.01)


def test_user_table_constant_init(fedmf_client_and_request):
    client, request = fedmf_client_and_request()
    u = request.client_state.local_parameters["user_embeddings"]
    np.testing.assert_allclose(u.numpy(), 0.01)


def test_normal_init_without_seed_fails():
    client = FedMFClientAlgorithm(init_mode="normal", latent_dim=2)
    client.device = torch.device("cpu")
    with pytest.raises(ValueError, match="seed"):
        client.initialize_local_parameters(
            client_id=0,
            total_num_users=1,
            num_users=1,
            num_items=3,
            client_train_data_view=None,
            users_in_client=[0],
        )


def test_server_publishes_train_m_and_torch_item_table():
    dataset = _tiny_dataset()
    client = FedMFClientAlgorithm(latent_dim=4, lr=1e-3)
    server = FedMF(
        latent_dim=4,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=0,
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=Sum(num_clients=dataset.num_clients),
        device="cpu",
    )
    assert server.extra_payload["n_train_ratings"] == 4
    table = server.global_parameters["item_embeddings"]["item_embeddings.weight"]
    assert isinstance(table, torch.Tensor)
    assert table.shape == (dataset.num_items, 4)
    np.testing.assert_allclose(table.detach().cpu().numpy(), 0.01)


def test_one_sum_round_moves_rated_item_rows():
    dataset = _tiny_dataset()
    client = FedMFClientAlgorithm(latent_dim=4, lr=0.01)
    server = FedMF(
        latent_dim=4,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=0,
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=Sum(num_clients=dataset.num_clients),
        device="cpu",
    )
    v0 = server.global_parameters["item_embeddings"]["item_embeddings.weight"].clone()
    server.run_round(0)
    v1 = server.global_parameters["item_embeddings"]["item_embeddings.weight"]
    # Items 0, 1, 2 are rated; every row that was uploaded should move.
    assert not torch.allclose(v1[0], v0[0])
    assert not torch.allclose(v1[1], v0[1])
    assert not torch.allclose(v1[2], v0[2])


def test_latent_dim_mismatch_fails():
    dataset = _tiny_dataset()
    client = FedMFClientAlgorithm(latent_dim=8)
    with pytest.raises(ValueError, match="latent_dim"):
        FedMF(
            latent_dim=4,
            interaction_dataset=dataset,
            client_algorithm=client,
            num_rounds=1,
            client_selection_strategy=FullClientSelection(dataset.num_clients),
            aggregation_strategies=Sum(num_clients=dataset.num_clients),
            device="cpu",
        )


def test_item_table_stays_torch_after_a_round():
    dataset = _tiny_dataset()
    client = FedMFClientAlgorithm(latent_dim=3, lr=0.01)
    server = FedMF(
        latent_dim=3,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=1,
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=Sum(num_clients=dataset.num_clients),
        device="cpu",
    )
    server.run_round(0)
    table = server.global_parameters["item_embeddings"]["item_embeddings.weight"]
    assert isinstance(table, torch.Tensor)
    assert table.device.type == "cpu"
