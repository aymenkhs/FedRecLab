"""EFVAE server: one round Adam, anneal, row updates touch support only."""

import copy

import numpy as np
import torch

from fedreclab.federated.aggregation import SimpleAvg
from fedreclab.federated.client_selection import FullClientSelection
from fedreclab.methods.efvae.efvae import EFVAE
from fedreclab.methods.efvae.efvae_client import EFVAEClientAlgorithm
from tests.efvae.conftest import HIDDEN, LATENT, NUM_CLUSTER, NUM_SAMPLE


def _make_server(dataset, *, seed=0, total_anneal_steps=10, anneal_cap=0.2):
    client = EFVAEClientAlgorithm(
        hidden_dims=HIDDEN, latent_dim=LATENT, dropout=0.5
    )
    aggregator = SimpleAvg(num_clients=dataset.num_clients)
    selector = FullClientSelection(dataset.num_clients)
    return EFVAE(
        hidden_dims=HIDDEN,
        latent_dim=LATENT,
        dropout=0.5,
        lr=1e-3,
        anneal_cap=anneal_cap,
        total_anneal_steps=total_anneal_steps,
        num_cluster=NUM_CLUSTER,
        num_sample=NUM_SAMPLE,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=seed,
        client_selection_strategy=selector,
        aggregation_strategies=aggregator,
        device="cpu",
    )


def test_one_round_changes_parameters(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    server = _make_server(dataset)
    before = copy.deepcopy(server.global_parameters)
    server.run_round(0)
    after = server.global_parameters
    # Middle dense should move; at least one embedding row should move.
    mid_changed = False
    for name in before["middle"]:
        if not torch.allclose(
            torch.as_tensor(before["middle"][name]),
            torch.as_tensor(after["middle"][name]),
        ):
            mid_changed = True
            break
    assert mid_changed
    enc_before = torch.as_tensor(before["enc_embeddings"]["enc_embeddings.weight"])
    enc_after = torch.as_tensor(after["enc_embeddings"]["enc_embeddings.weight"])
    row_delta = (enc_before - enc_after).abs().sum(dim=-1)
    assert float(row_delta.max()) > 0.0


def test_untouched_enc_rows_remain_with_one_client(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    server = _make_server(dataset, seed=1)
    # Train only client 0 (owns half the catalogue under the fixture).
    before = torch.as_tensor(
        server.global_parameters["enc_embeddings"]["enc_embeddings.weight"]
    ).clone()
    server.current_round = 0
    server.selected_client_ids = [0]
    server._on_round_start_server()
    response = server._train_client(0, 0)
    server.current_round_statistics = server._update_statistics(0, [0], [response])
    server.aggregate_parameters([response])
    after = torch.as_tensor(
        server.global_parameters["enc_embeddings"]["enc_embeddings.weight"]
    )
    unchanged = (before - after).abs().sum(dim=-1) == 0
    assert bool(unchanged.any())


def test_anneal_schedule(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=1)
    server = _make_server(dataset, total_anneal_steps=10, anneal_cap=0.2)
    server.current_round = 0
    server._on_round_start_server()
    assert server.extra_payload["anneal"] == 0.0
    server.update_count = 1
    server._on_round_start_server()
    np.testing.assert_allclose(server.extra_payload["anneal"], 0.1)
    server.update_count = 100
    server._on_round_start_server()
    np.testing.assert_allclose(server.extra_payload["anneal"], 0.2)


def test_adam_step_once_per_round(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    server = _make_server(dataset)
    assert server.adam_step == 0
    server.run_round(0)
    assert server.adam_step == 1
