"""FedFast server: actv_agg wiring, round-end subordinates, selector sync."""

import numpy as np
import pytest
import torch

from fedreclab.federated.aggregation import ChangeMagnitudeAvg, FedAvg, SimpleAvg
from fedreclab.federated.client_selection import ActvSampClientSelection
from fedreclab.methods.fedfast.fedfast import FedFast
from fedreclab.methods.fedfast.fedfast_client import FedFastClientAlgorithm
from tests.fedfast.conftest import make_fedfast_dataset, make_fedfast_server


def test_actv_agg_true_installs_change_magnitude_and_fedavg():
    dataset = make_fedfast_dataset(4)
    server = make_fedfast_server(dataset, num_clusters=2, number_selected_clients=2)
    assert isinstance(
        server.aggregation_strategies["item_embeddings"], ChangeMagnitudeAvg
    )
    assert isinstance(server.aggregation_strategies["output_weights"], FedAvg)
    assert isinstance(server.client_selection_module, ActvSampClientSelection)
    server._before_training()
    groups = server.client_selection_module.groups
    assert groups is not None
    assert sum(len(g) for g in groups) == 4
    assert len(groups) >= 1


def test_actv_agg_false_requires_explicit_strategies():
    dataset = make_fedfast_dataset(4)
    with pytest.raises(ValueError, match="aggregation_strategies"):
        make_fedfast_server(dataset, actv_agg=False)


def test_one_round_writes_subordinate_and_refreshes_groups():
    dataset = make_fedfast_dataset(4)
    server = make_fedfast_server(
        dataset, num_clusters=2, number_selected_clients=2, num_rounds=1
    )
    server.run()

    groups = server.client_selection_module.groups
    assert groups is not None
    assert sum(len(g) for g in groups) == 4
    assert all(len(g) >= 1 for g in groups)
    assert server._cluster_state.groups is not None
    assert server._cluster_state.last_gamma is not None


def test_ablation_simpleavg_skips_subordinate_when_actv_agg_false():
    dataset = make_fedfast_dataset(4)
    n = dataset.num_clients
    client = FedFastClientAlgorithm(embedding_dim=4, local_epochs=1, neg_samples=0)
    server = FedFast(
        embedding_dim=4,
        num_clusters=2,
        actv_agg=False,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=0,
        client_selection_strategy=ActvSampClientSelection(n, 2, seed=0),
        aggregation_strategies={
            "item_embeddings": SimpleAvg(n),
            "output_weights": SimpleAvg(n),
        },
        device="cpu",
    )
    # Manual G0 so ActvSAMP can select; actv_agg=False skips round-end writes.
    server._before_training()
    before = {
        cid: torch.as_tensor(state.local_parameters["user_embeddings"]).clone()
        for cid, state in server.client_states.items()
    }
    server.run_round(0)
    for cid, state in server.client_states.items():
        if cid in (server.selected_client_ids or []):
            continue
        assert torch.allclose(
            before[cid], torch.as_tensor(state.local_parameters["user_embeddings"])
        )
