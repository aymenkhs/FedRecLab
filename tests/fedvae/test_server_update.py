"""FedVAE server Adam step, FedVAEdec schedule, and aggregator compatibility."""

import math

import numpy as np
import pytest
import torch

from fedreclab.federated.aggregation import Krum, Median, SimpleAvg
from fedreclab.federated.client_selection import (
    FullClientSelection,
    RandomClientSelection,
)
from fedreclab.methods.fedvae.fedvae import FedVAE, default_param_ios
from fedreclab.methods.fedvae.fedvae_client import FedVAEClientAlgorithm
from fedreclab.methods.fedvae.fedvae_model import MultVAE

HIDDEN = (4,)
LATENT = 2


def _make_server(
    dataset,
    *,
    aggregator=None,
    selector=None,
    lr=1e-3,
    lr_multiplier=0.0,
    lr_decay=0.9,
    decay_every_rounds=1,
    extrapolate="all",
    param_io_overrides=None,
    server_learning_rate=None,
    seed=0,
):
    client = FedVAEClientAlgorithm(
        hidden_dims=HIDDEN, latent_dim=LATENT, dropout=0.5, beta=0.2
    )
    if aggregator is None:
        aggregator = SimpleAvg(num_clients=dataset.num_clients)
    if selector is None:
        selector = FullClientSelection(dataset.num_clients)
    server = FedVAE(
        hidden_dims=HIDDEN,
        latent_dim=LATENT,
        dropout=0.5,
        lr=lr,
        lr_multiplier=lr_multiplier,
        lr_decay=lr_decay,
        decay_every_rounds=decay_every_rounds,
        extrapolate=extrapolate,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        seed=seed,
        client_selection_strategy=selector,
        aggregation_strategies=aggregator,
        param_io_overrides=param_io_overrides,
        device="cpu",
    )
    if server_learning_rate is not None:
        server.server_learning_rate = server_learning_rate
    return server


def _clone_group(group):
    return {name: tensor.detach().clone() for name, tensor in group.items()}


def _hand_adam(old, grads, *, lr, betas=(0.9, 0.999), eps=1e-8, step=1, eta=None):
    """One bias-corrected Adam step from zero moments."""
    if eta is None:
        eta = lr
    beta1, beta2 = betas
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    out = {}
    for name, w in old.items():
        g = grads[name]
        m = (1.0 - beta1) * g
        v = (1.0 - beta2) * (g * g)
        m_hat = m / bc1
        v_hat = v / bc2
        step_size = eta[name] if isinstance(eta, dict) else eta
        out[name] = w - step_size * m_hat / (torch.sqrt(v_hat) + eps)
    return out


def _mean_grads(server, client_ids, round_idx):
    """Average uploaded gradients at the current global weights."""
    acc = None
    n = 0
    for client_id in client_ids:
        response = server._train_client(client_id, round_idx)
        grad = response.updates_packages["vae"].param_value
        if acc is None:
            acc = {name: tensor.clone() for name, tensor in grad.items()}
        else:
            for name, tensor in grad.items():
                acc[name] = acc[name] + tensor
        n += 1
    return {name: tensor / n for name, tensor in acc.items()}


def test_one_simpleavg_round_matches_hand_adam(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    server = _make_server(dataset, lr=1e-3, decay_every_rounds=1)
    w0 = _clone_group(server.global_parameters["vae"])
    client_ids = list(range(dataset.num_clients))
    mean_grad = _mean_grads(server, client_ids, 0)
    server.run_round(0)
    expected = _hand_adam(w0, mean_grad, lr=1e-3)
    got = server.global_parameters["vae"]
    for name in w0:
        np.testing.assert_allclose(
            got[name].detach().cpu().numpy(),
            expected[name].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_first_round_schedule_uses_full_gamma(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    lr = 1e-3
    gamma = 3.0
    client_ids = list(range(dataset.num_clients))
    server = _make_server(
        dataset, lr=lr, lr_multiplier=gamma, lr_decay=0.9, decay_every_rounds=1
    )
    w0 = _clone_group(server.global_parameters["vae"])
    mean_grad = _mean_grads(server, client_ids, 0)
    server.run_round(0)
    expected = _hand_adam(w0, mean_grad, lr=lr, eta=lr * (1.0 + gamma))
    got = server.global_parameters["vae"]
    for name in w0:
        np.testing.assert_allclose(
            got[name].detach().cpu().numpy(),
            expected[name].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_second_epoch_uses_decayed_gamma(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    lr, gamma, delta, period = 1e-3, 2.0, 0.5, 2
    server = _make_server(
        dataset,
        lr=lr,
        lr_multiplier=gamma,
        lr_decay=delta,
        decay_every_rounds=period,
    )
    server.current_round = 2
    w0 = _clone_group(server.global_parameters["vae"])
    fake_grad = {name: torch.ones_like(tensor) for name, tensor in w0.items()}
    server._apply_aggregated_result_custom(
        "vae", fake_grad, server.param_ios["vae"], [0]
    )
    eta_t = lr * (1.0 + gamma * (delta ** (2 // period)))
    expected = _hand_adam(w0, fake_grad, lr=lr, eta=eta_t)
    got = server.global_parameters["vae"]
    for name in w0:
        np.testing.assert_allclose(
            got[name].detach().cpu().numpy(),
            expected[name].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_default_decay_every_rounds_is_one_epoch(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=4)
    selector = RandomClientSelection(dataset.num_clients, 2, "num", seed=0)
    server = _make_server(
        dataset, selector=selector, decay_every_rounds=None, seed=0
    )
    assert server.decay_every_rounds is None
    server.run_round(0)
    r = len(server.selected_client_ids)
    assert server.decay_every_rounds == int(math.ceil(dataset.num_clients / r))
    assert server.decay_every_rounds == 2


def test_extrapolate_weights_leaves_biases_on_the_plain_step(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    lr = 1e-3
    gamma = 5.0
    client_ids = list(range(dataset.num_clients))
    server = _make_server(
        dataset,
        lr=lr,
        lr_multiplier=gamma,
        lr_decay=0.9,
        decay_every_rounds=1,
        extrapolate="weights",
    )
    w0 = _clone_group(server.global_parameters["vae"])
    mean_grad = _mean_grads(server, client_ids, 0)
    server.run_round(0)
    eta = {
        name: lr * (1.0 + gamma) if name.endswith(".weight") else lr for name in w0
    }
    expected = _hand_adam(w0, mean_grad, lr=lr, eta=eta)
    got = server.global_parameters["vae"]
    for name in w0:
        np.testing.assert_allclose(
            got[name].detach().cpu().numpy(),
            expected[name].detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    expected_all = _hand_adam(w0, mean_grad, lr=lr, eta=lr * (1.0 + gamma))
    bias_names = [name for name in w0 if name.endswith(".bias")]
    assert bias_names
    for name in bias_names:
        assert not torch.allclose(got[name], expected_all[name])


def test_krum_and_median_run_without_method_special_casing(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=3)
    krum_server = _make_server(
        dataset,
        aggregator=Krum(num_clients=dataset.num_clients, num_byzantine=0),
    )
    w0 = _clone_group(krum_server.global_parameters["vae"])
    krum_server.run_round(0)
    krum_w = krum_server.global_parameters["vae"]
    assert any(not torch.allclose(krum_w[n], w0[n]) for n in w0)

    median_server = _make_server(
        dataset,
        aggregator=Median(num_clients=dataset.num_clients),
        seed=1,
    )
    w0m = _clone_group(median_server.global_parameters["vae"])
    median_server.run_round(0)
    median_w = median_server.global_parameters["vae"]
    assert any(not torch.allclose(median_w[n], w0m[n]) for n in w0m)


def test_plain_sgd_ablation_subtracts_lr_times_mean_grad(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    lr = 0.25
    client_ids = list(range(dataset.num_clients))
    server = _make_server(
        dataset,
        param_io_overrides={"vae": {"apply_to": "server"}},
        server_learning_rate=lr,
        decay_every_rounds=1,
    )
    assert default_param_ios()["vae"].apply_to == "custom"
    assert server.param_ios["vae"].apply_to == "server"
    w0 = _clone_group(server.global_parameters["vae"])
    mean_grad = _mean_grads(server, client_ids, 0)
    server.run_round(0)
    got = server.global_parameters["vae"]
    for name, old in w0.items():
        expected = old - lr * mean_grad[name]
        np.testing.assert_allclose(
            torch.as_tensor(got[name]).detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_architecture_mismatch_fails(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    client = FedVAEClientAlgorithm(hidden_dims=(8,), latent_dim=LATENT)
    with pytest.raises(ValueError, match="hidden_dims"):
        FedVAE(
            hidden_dims=HIDDEN,
            latent_dim=LATENT,
            interaction_dataset=dataset,
            client_algorithm=client,
            num_rounds=1,
            client_selection_strategy=FullClientSelection(dataset.num_clients),
            aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
            device="cpu",
        )


def test_seeded_server_init_matches_model_generator(make_tiny_dataset):
    dataset = make_tiny_dataset(n_users=2)
    server = _make_server(dataset, seed=11)
    generator = torch.Generator()
    generator.manual_seed(11)
    model = MultVAE(
        num_items=dataset.num_items,
        hidden_dims=HIDDEN,
        latent_dim=LATENT,
        dropout=0.5,
        generator=generator,
    )
    expected = model.get_server_state_groups()["vae"]
    got = server.global_parameters["vae"]
    for name in expected:
        assert torch.equal(got[name], expected[name])
