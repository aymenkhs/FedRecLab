"""Chain-rule Theta-gradients, Adam parity, aggregator plumbing."""

import numpy as np
import pytest
import torch
from torch import nn

from src.fedreclab.federated.aggregation import FedAvg, Median
from src.fedreclab.methods.metamf.metamf_model import (
    MetaRecommender,
    predict_ratings,
    unbatch_phi,
)
from tests.metamf.conftest import TINY, make_server, tiny_dataset, two_user_client_dataset


def _clone_group(group):
    return {name: tensor.detach().clone() for name, tensor in group.items()}


def _centralized_theta_grad(model, item_payload, user_ids, item_ids, ratings):
    """Mean-squared-error gradient of Theta on a rating table."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    item_ids = np.asarray(item_ids, dtype=np.int64)
    ratings = np.asarray(ratings, dtype=np.float64)
    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    user_t = torch.as_tensor(unique_users, dtype=torch.long, device=model.user_embedding.weight.device)
    phi = model.generate(user_t, item_payload=item_payload)
    preds = []
    for row, user_index in enumerate(inverse):
        phi_u = unbatch_phi(phi, int(user_index))
        item = torch.tensor(
            [int(item_ids[row])], dtype=torch.long, device=user_t.device
        )
        preds.append(predict_ratings(phi_u, item))
    pred = torch.cat(preds, dim=0)
    target = torch.as_tensor(ratings, dtype=pred.dtype, device=pred.device)
    loss = torch.mean((pred - target) ** 2)
    grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    out = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        if grad is None:
            out[name] = torch.zeros_like(param, device="cpu")
        else:
            out[name] = grad.detach().cpu().clone()
    return out


def _client_ratings(server, client_id):
    view = server.interaction_dataset.get_client_view(
        client_id, include_validation=False, include_test=False
    )
    return (
        view["user_id"].to_numpy(),
        view["item_id"].to_numpy(),
        view["rating"].to_numpy(dtype=np.float64),
    )


@pytest.mark.parametrize("item_payload", ["factors", "full"])
def test_chain_rule_matches_centralized_one_user(item_payload):
    server = make_server(tiny_dataset(n_users=2), item_payload=item_payload, seed=0)
    response = server._train_client(0, 0)
    updates, ids = server._collect_updates(
        "meta_network", server.param_ios["meta_network"], [response]
    )
    assert ids == [0]
    users, items, ratings = _client_ratings(server, 0)
    expected = _centralized_theta_grad(
        server._meta_model, item_payload, users, items, ratings
    )
    for name in expected:
        np.testing.assert_allclose(
            updates[0][name].numpy(),
            expected[name].numpy(),
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parametrize("item_payload", ["factors", "full"])
def test_chain_rule_matches_centralized_two_user_client(item_payload):
    server = make_server(
        two_user_client_dataset(), item_payload=item_payload, seed=0
    )
    response = server._train_client(0, 0)
    updates, _ = server._collect_updates(
        "meta_network", server.param_ios["meta_network"], [response]
    )
    users, items, ratings = _client_ratings(server, 0)
    assert set(users.tolist()) == {0, 1}
    expected = _centralized_theta_grad(
        server._meta_model, item_payload, users, items, ratings
    )
    for name in expected:
        np.testing.assert_allclose(
            updates[0][name].numpy(),
            expected[name].numpy(),
            rtol=1e-4,
            atol=1e-5,
        )


def test_fedavg_matches_centralized_per_rating_mean():
    dataset = tiny_dataset(n_users=2)
    server = make_server(
        dataset, aggregator=FedAvg(num_clients=dataset.num_clients), seed=0
    )
    responses = [server._train_client(cid, 0) for cid in range(dataset.num_clients)]
    server.current_round = 0
    server.current_round_statistics = server._update_statistics(
        0, list(range(dataset.num_clients)), responses
    )
    updates, ids = server._collect_updates(
        "meta_network", server.param_ios["meta_network"], responses
    )
    aggregated = server.aggregation_strategies["meta_network"].aggregate(
        server._round_statistics_for_participants(ids),
        server.global_parameters["meta_network"],
        updates,
        server.param_ios["meta_network"],
    )
    all_users, all_items, all_ratings = [], [], []
    for cid in range(dataset.num_clients):
        users, items, ratings = _client_ratings(server, cid)
        all_users.append(users)
        all_items.append(items)
        all_ratings.append(ratings)
    expected = _centralized_theta_grad(
        server._meta_model,
        "factors",
        np.concatenate(all_users),
        np.concatenate(all_items),
        np.concatenate(all_ratings),
    )
    for name in expected:
        np.testing.assert_allclose(
            torch.as_tensor(aggregated[name]).cpu().numpy(),
            expected[name].numpy(),
            rtol=1e-4,
            atol=1e-5,
        )


def test_median_round_completes_with_matching_theta_shapes():
    dataset = tiny_dataset(n_users=3)
    captured = []
    aggregator = Median(num_clients=dataset.num_clients)
    orig = aggregator.aggregate

    def wrapped(round_info, previous, updates, param_io):
        captured.append(updates)
        return orig(round_info, previous, updates, param_io)

    aggregator.aggregate = wrapped
    server = make_server(dataset, aggregator=aggregator, seed=0)
    server.run_round(0)
    assert captured
    theta = server.global_parameters["meta_network"]
    for update in captured[0]:
        assert set(update) == set(theta)
        for name in theta:
            assert tuple(torch.as_tensor(update[name]).shape) == tuple(theta[name].shape)


def test_payload_never_contains_theta():
    server = make_server(tiny_dataset(n_users=2), seed=0)
    payload = server._build_server_payload(0)
    assert "meta_network" not in payload
    assert "generated_parameters" in payload
    phi = next(iter(payload["generated_parameters"].values()))
    assert "user_embedding.weight" not in phi
    assert "item_left" in phi


def test_item_rows_vjp_matches_full_generation():
    server = make_server(tiny_dataset(n_users=2), seed=0)
    model = server._meta_model
    model.load_server_weights(server.global_parameters)
    user_t = torch.tensor([0], dtype=torch.long, device=model.user_embedding.weight.device)
    rows = torch.tensor([0, 1], dtype=torch.long, device=user_t.device)
    sliced = model.item_rows(user_t, rows, item_payload="factors")
    one_s = unbatch_phi(sliced, 0)
    g_left = torch.randn_like(one_s["item_left"])
    g_right = torch.randn_like(one_s["item_right"])
    g_w1 = torch.randn_like(one_s["w1"])
    g_b1 = torch.randn_like(one_s["b1"])
    g_w2 = torch.randn_like(one_s["w2"])
    g_b2 = torch.randn_like(one_s["b2"])
    params = list(model.parameters())
    sliced_grads = torch.autograd.grad(
        [one_s["item_left"], one_s["item_right"], one_s["w1"], one_s["b1"], one_s["w2"], one_s["b2"]],
        params,
        grad_outputs=[g_left, g_right, g_w1, g_b1, g_w2, g_b2],
        allow_unused=True,
    )
    full = model.generate(user_t, item_payload="factors")
    one_f = unbatch_phi(full, 0)
    full_left_rows = one_f["item_left"].index_select(0, rows)
    full_grads = torch.autograd.grad(
        [full_left_rows, one_f["item_right"], one_f["w1"], one_f["b1"], one_f["w2"], one_f["b2"]],
        params,
        grad_outputs=[g_left, g_right, g_w1, g_b1, g_w2, g_b2],
        allow_unused=True,
    )
    for g_s, g_f in zip(sliced_grads, full_grads):
        if g_s is None and g_f is None:
            continue
        np.testing.assert_allclose(
            g_s.detach().cpu().numpy(),
            g_f.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def _synthetic_grads(model, scale_second=False):
    grads = []
    for step in range(3):
        g = {}
        for name, param in model.named_parameters():
            torch.manual_seed(10 + step)
            noise = torch.randn_like(param)
            if name == "user_embedding.weight":
                noise = noise.clone()
                noise[1].zero_()
            g[name] = noise
        if step == 1 and scale_second:
            first = next(iter(g))
            g[first] = g[first] * 1.0e3
        grads.append(g)
    return grads


def test_multi_step_matches_clip_then_adam_and_keeps_state_attached():
    server = make_server(
        tiny_dataset(n_users=3), lr=1e-3, weight_decay=1e-3, max_grad_norm=5.0, seed=0
    )
    ref = MetaRecommender(
        num_users=server.interaction_dataset.num_users,
        num_items=server.interaction_dataset.num_items,
        **TINY,
    )
    ref.load_server_weights(server.global_parameters)
    opt = torch.optim.Adam(
        ref.parameters(),
        lr=server.lr,
        betas=server.betas,
        eps=server.eps,
        weight_decay=server.weight_decay,
    )
    pio = server.param_ios["meta_network"]
    grads = _synthetic_grads(server._meta_model, scale_second=True)
    norms = [
        float(torch.sqrt(sum((t.float() ** 2).sum() for t in g.values())))
        for g in grads
    ]
    assert norms[1] > 5.0
    row_before = ref.user_embedding.weight[1].detach().clone()
    for g in grads:
        cpu_g = {name: tensor.detach().cpu().clone() for name, tensor in g.items()}
        server._apply_aggregated_result_custom("meta_network", cpu_g, pio, [0])
        replaced = {
            name: tensor.detach().cpu().clone()
            for name, tensor in server.global_parameters["meta_network"].items()
        }
        server.global_parameters["meta_network"] = replaced
        for name, param in ref.named_parameters():
            param.grad = g[name].to(device=param.device, dtype=param.dtype)
        nn.utils.clip_grad_norm_(ref.parameters(), server.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
    got = server.global_parameters["meta_network"]
    for name, param in ref.named_parameters():
        np.testing.assert_allclose(
            got[name].detach().cpu().numpy(),
            param.detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
    param_ids = {id(p) for p in server._meta_model.parameters()}
    state_ids = {id(p) for p in server._meta_optimizer.state}
    assert state_ids == param_ids
    steps = [int(state["step"]) for state in server._meta_optimizer.state.values()]
    assert steps and all(s == 3 for s in steps)
    row_after = ref.user_embedding.weight[1].detach()
    assert not torch.equal(row_before, row_after)
