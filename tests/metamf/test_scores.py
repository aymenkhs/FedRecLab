"""Evaluator-agnostic predicted-rating scores and skeleton caching."""

import numpy as np
import torch

from src.fedreclab.methods.metamf.metamf_model import predict_ratings, unbatch_phi
from tests.metamf.conftest import make_server, tiny_dataset


def test_scores_align_with_candidate_arrays():
    server = make_server(tiny_dataset(n_users=2), seed=0)
    client = server.client_algorithm
    state = server.client_states[0]
    params = server.global_parameters
    candidates = {
        0: np.array([2, 0, 5]),
        1: np.array([], dtype=np.int64),
    }
    scores = client.compute_scores(0, candidates, params, state, {})
    assert scores[0].shape == (3,)
    assert scores[1].shape == (0,)
    model = client._skeleton
    with torch.no_grad():
        phi = unbatch_phi(
            model.generate(torch.tensor([0]), item_payload="factors"), 0
        )
        expected = predict_ratings(phi, torch.tensor([2, 0, 5]))
    np.testing.assert_allclose(
        scores[0], expected.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )


def test_sliced_and_full_paths_agree():
    server = make_server(tiny_dataset(n_users=1), seed=0)
    client = server.client_algorithm
    state = server.client_states[0]
    params = server.global_parameters
    small = np.array([0, 2])
    large = np.arange(client.num_items)
    scores_small = client.compute_scores(0, {0: small}, params, state, {})
    assert small.size < client.num_items // 2
    phi = unbatch_phi(
        client._skeleton.generate(torch.tensor([0]), item_payload="factors"), 0
    )
    expected_small = predict_ratings(phi, torch.as_tensor(small))
    np.testing.assert_allclose(
        scores_small[0], expected_small.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    scores_large = client.compute_scores(0, {0: large}, params, state, {})
    expected_large = predict_ratings(phi, torch.as_tensor(large))
    np.testing.assert_allclose(
        scores_large[0], expected_large.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )


def test_skeleton_reloads_only_when_theta_identity_changes():
    server = make_server(tiny_dataset(n_users=3), seed=0)
    client = server.client_algorithm
    params = server.global_parameters
    state = server.client_states[0]
    candidates = {uid: np.array([0, 1]) for uid in range(3)}
    client.compute_scores(0, candidates, params, state, {})
    loads = {"n": 0}
    orig = client._skeleton.load_server_weights

    def spy(payload):
        loads["n"] += 1
        return orig(payload)

    client._skeleton.load_server_weights = spy
    client.compute_scores(0, candidates, params, state, {})
    client.compute_scores(1, candidates, params, server.client_states[1], {})
    assert loads["n"] == 0
    new_theta = {
        name: tensor.detach().clone()
        for name, tensor in params["meta_network"].items()
    }
    client.compute_scores(
        0, candidates, {"meta_network": new_theta}, state, {}
    )
    assert loads["n"] == 1
    client._skeleton.load_server_weights = orig


def test_new_theta_after_server_step_reloads_and_changes_scores():
    server = make_server(tiny_dataset(n_users=2), seed=0, lr=1e-2)
    client = server.client_algorithm
    state = server.client_states[0]
    candidates = {0: np.array([1, 2])}
    before = client.compute_scores(
        0, candidates, server.global_parameters, state, {}
    )[0].copy()
    loaded = client._loaded_theta
    server.run_round(0)
    after_params = server.global_parameters
    assert after_params["meta_network"] is not loaded
    after = client.compute_scores(0, candidates, after_params, state, {})[0]
    assert client._loaded_theta is after_params["meta_network"]
    assert not np.allclose(before, after)


def test_best_global_parameters_snapshot_triggers_reload():
    server = make_server(tiny_dataset(n_users=2), seed=0)
    client = server.client_algorithm
    state = server.client_states[0]
    candidates = {0: np.array([1])}
    client.compute_scores(0, candidates, server.global_parameters, state, {})
    server._snapshot_best_state(0)
    loads = {"n": 0}
    orig = client._skeleton.load_server_weights

    def spy(payload):
        loads["n"] += 1
        return orig(payload)

    client._skeleton.load_server_weights = spy
    client.compute_scores(
        0, candidates, server.best_global_parameters, state, {}
    )
    assert loads["n"] == 1
    client._skeleton.load_server_weights = orig
