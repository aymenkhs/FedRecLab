"""Construction rejects overrides that break the Phi-to-Theta translation."""

import numpy as np
import pytest
import torch

from src.fedreclab.federated.aggregation import SimpleAvg
from src.fedreclab.federated.client_selection import FullClientSelection
from src.fedreclab.methods.metamf.metamf import MetaMF
from src.fedreclab.methods.metamf.metamf_client import MetaMFClientAlgorithm
from tests.metamf.conftest import TINY, make_server, tiny_dataset


def _fresh_components(dataset):
    client = MetaMFClientAlgorithm(**TINY)
    selector = FullClientSelection(dataset.num_clients)
    aggregator = SimpleAvg(num_clients=dataset.num_clients)
    return client, selector, aggregator


def test_layout_rows_raises_before_binding():
    dataset = tiny_dataset(n_users=2)
    client, selector, aggregator = _fresh_components(dataset)
    with pytest.raises(ValueError, match="layout"):
        MetaMF(
            interaction_dataset=dataset,
            client_algorithm=client,
            num_rounds=1,
            client_selection_strategy=selector,
            aggregation_strategies=aggregator,
            param_io_overrides={"meta_network": {"layout": "rows"}},
            device="cpu",
            **TINY,
        )
    assert getattr(client, "_bound_server", False) is False
    assert getattr(selector, "_bound_server", False) is False
    assert getattr(aggregator, "_bound_server", False) is False
    MetaMF(
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        client_selection_strategy=selector,
        aggregation_strategies=aggregator,
        device="cpu",
        **TINY,
    )
    assert getattr(client, "_bound_server", False) is True


@pytest.mark.parametrize(
    "exchange",
    ["params", "delta", "encrypted_params", "encrypted_delta", "none"],
)
def test_disallowed_exchange_raises(exchange):
    dataset = tiny_dataset(n_users=2)
    with pytest.raises(ValueError, match="exchange"):
        make_server(
            dataset,
            param_io_overrides={"meta_network": {"exchange": exchange}},
        )


def test_encrypted_gradient_is_accepted():
    server = make_server(
        tiny_dataset(n_users=2),
        param_io_overrides={"meta_network": {"exchange": "encrypted_gradient"}},
    )
    assert server.param_ios["meta_network"].exchange == "encrypted_gradient"


def test_client_residency_and_clients_apply_to_raise():
    dataset = tiny_dataset(n_users=2)
    with pytest.raises(ValueError, match="residency"):
        make_server(
            dataset,
            param_io_overrides={"meta_network": {"residency": "client"}},
        )
    with pytest.raises(ValueError, match="apply_to"):
        make_server(
            dataset,
            param_io_overrides={"meta_network": {"apply_to": "clients"}},
        )
    with pytest.raises(ValueError, match="apply_to"):
        make_server(
            dataset,
            param_io_overrides={"meta_network": {"apply_to": "both"}},
        )


def test_apply_to_server_without_lr_raises_at_round_start():
    server = make_server(
        tiny_dataset(n_users=2),
        param_io_overrides={"meta_network": {"apply_to": "server"}},
    )
    with pytest.raises(ValueError, match="server_learning_rate"):
        server.run_round(0)


def test_apply_to_server_is_plain_sgd_without_weight_decay_or_clip():
    dataset = tiny_dataset(n_users=2)
    eta = 0.25
    server = make_server(
        dataset,
        aggregator=SimpleAvg(num_clients=dataset.num_clients),
        param_io_overrides={"meta_network": {"apply_to": "server"}},
        weight_decay=1e-3,
        max_grad_norm=5.0,
        seed=0,
    )
    server.server_learning_rate = eta
    w0 = {
        name: tensor.detach().clone()
        for name, tensor in server.global_parameters["meta_network"].items()
    }
    server.current_round = 0
    server.selected_client_ids = list(range(dataset.num_clients))
    responses = [
        server._train_client(cid, 0) for cid in server.selected_client_ids
    ]
    server.current_round_statistics = server._update_statistics(
        0, server.selected_client_ids, responses
    )
    updates, _ = server._collect_updates(
        "meta_network", server.param_ios["meta_network"], responses
    )
    mean = {}
    keys = updates[0].keys()
    for name in keys:
        mean[name] = torch.stack(
            [torch.as_tensor(u[name], dtype=torch.float32) for u in updates]
        ).mean(dim=0)
    server.aggregate_parameters(responses)
    got = server.global_parameters["meta_network"]
    for name in w0:
        expected = w0[name] - eta * mean[name]
        np.testing.assert_allclose(
            got[name].detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
