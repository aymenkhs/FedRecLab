"""Components are single-use per server; the dataset may be shared."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.evaluator import FullRankingEvaluator
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.aggregation import SimpleAvg, Sum
from src.fedreclab.federated.base_client import BaseClientAlgorithm
from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.client_selection import FullClientSelection
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedmf.fedmf import FedMF
from src.fedreclab.methods.fedmf.fedmf_client import FedMFClientAlgorithm


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


class _DummyClient(BaseClientAlgorithm):
    def initialize_local_parameters(
        self,
        client_id,
        total_num_users,
        num_users,
        num_items,
        client_train_data_view,
        users_in_client,
    ):
        return {}


class _DummyServer(BaseFederatedServer):
    def _initialize_server_parameters(self):
        return {
            "a": np.zeros(1, dtype=np.float32),
            "b": np.zeros(1, dtype=np.float32),
        }


def _param_ios():
    return {
        "a": ParamIO(residency="server", exchange="params", apply_to="server"),
        "b": ParamIO(residency="server", exchange="params", apply_to="server"),
    }


def _make_server(dataset, **overrides):
    kwargs = dict(
        interaction_dataset=dataset,
        client_algorithm=_DummyClient(),
        num_rounds=1,
        param_ios=_param_ios(),
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=SimpleAvg(dataset.num_clients),
        device="cpu",
        training_framework="numpy",
    )
    kwargs.update(overrides)
    return _DummyServer(**kwargs)


def _assert_reuse_rejected(dataset, class_name, **shared):
    _make_server(dataset, **shared)
    with pytest.raises(ValueError, match=f"{class_name} instance is already used"):
        _make_server(dataset, **shared)


def test_reused_client_algorithm_is_rejected():
    dataset = _tiny_dataset()
    _assert_reuse_rejected(dataset, "_DummyClient", client_algorithm=_DummyClient())


def test_reused_selector_is_rejected():
    dataset = _tiny_dataset()
    selector = FullClientSelection(dataset.num_clients)
    _assert_reuse_rejected(
        dataset, "FullClientSelection", client_selection_strategy=selector
    )


def test_reused_aggregator_is_rejected():
    dataset = _tiny_dataset()
    aggregator = SimpleAvg(dataset.num_clients)
    _assert_reuse_rejected(dataset, "SimpleAvg", aggregation_strategies=aggregator)


def test_aggregator_reused_inside_a_group_dict_is_rejected():
    dataset = _tiny_dataset()
    aggregator = SimpleAvg(dataset.num_clients)
    _make_server(dataset, aggregation_strategies=aggregator)
    with pytest.raises(ValueError, match="SimpleAvg instance is already used"):
        _make_server(
            dataset,
            aggregation_strategies={
                "a": SimpleAvg(dataset.num_clients),
                "b": aggregator,
            },
        )


def test_reused_evaluator_is_rejected():
    dataset = _tiny_dataset()
    evaluator = FullRankingEvaluator(
        metrics=["ndcg"],
        topk=1,
        interaction_dataset=dataset,
        client_algorithm=_DummyClient(),
    )
    _assert_reuse_rejected(dataset, "FullRankingEvaluator", evaluator=evaluator)


def test_rejected_construction_leaves_fresh_components_unbound():
    dataset = _tiny_dataset()
    selector = FullClientSelection(dataset.num_clients)
    _make_server(dataset, client_selection_strategy=selector)
    fresh_client = _DummyClient()
    with pytest.raises(ValueError, match="FullClientSelection"):
        _make_server(
            dataset,
            client_algorithm=fresh_client,
            client_selection_strategy=selector,
        )
    _make_server(dataset, client_algorithm=fresh_client)


def test_deleting_the_first_server_does_not_unbind_components():
    dataset = _tiny_dataset()
    client = _DummyClient()
    server = _make_server(dataset, client_algorithm=client)
    del server
    with pytest.raises(ValueError, match="_DummyClient instance is already used"):
        _make_server(dataset, client_algorithm=client)


def test_one_aggregator_for_several_groups_of_one_server_is_accepted():
    dataset = _tiny_dataset()
    shared = SimpleAvg(dataset.num_clients)
    server = _make_server(dataset, aggregation_strategies={"a": shared, "b": shared})
    assert server.aggregation_strategies["a"] is server.aggregation_strategies["b"]


def test_servers_with_fresh_components_can_share_one_dataset():
    dataset = _tiny_dataset()
    first = _make_server(dataset)
    second = _make_server(dataset)
    assert first.interaction_dataset is second.interaction_dataset


def _fedmf_server(dataset, seed):
    return FedMF(
        latent_dim=4,
        init_mode="normal",
        interaction_dataset=dataset,
        client_algorithm=FedMFClientAlgorithm(latent_dim=4, init_mode="normal"),
        num_rounds=1,
        seed=seed,
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=Sum(num_clients=dataset.num_clients),
        device="cpu",
    )


def test_fedmf_fresh_clients_follow_each_server_seed():
    dataset = _tiny_dataset()
    first = _fedmf_server(dataset, seed=0)
    second = _fedmf_server(dataset, seed=1)

    assert first.client_algorithm.init_seed == 0
    assert second.client_algorithm.init_seed == 1
    v_first = first.global_parameters["item_embeddings"]["item_embeddings.weight"]
    v_second = second.global_parameters["item_embeddings"]["item_embeddings.weight"]
    assert not torch.allclose(v_first, v_second)
    u_first = first.client_states[0].local_parameters["user_embeddings"]
    u_second = second.client_states[0].local_parameters["user_embeddings"]
    assert not torch.allclose(u_first, u_second)
