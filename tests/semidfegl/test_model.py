"""Unit tests for SemiDFEGL LightGCN helpers and fuzzy c-means."""

import numpy as np
import torch

from fedreclab.methods.semidfegl.semidfegl_model import (
    fuzzy_cmeans,
    lightgcn_layer_combine,
    propagate_semidfegl,
    score_dot,
    xavier_uniform_table,
)


def test_xavier_table_is_seeded():
    g0 = torch.Generator().manual_seed(0)
    g1 = torch.Generator().manual_seed(0)
    a = xavier_uniform_table(5, 3, generator=g0)
    b = xavier_uniform_table(5, 3, generator=g1)
    assert a.shape == (5, 3)
    assert torch.allclose(a, b)


def test_layer_combine_and_score_dot():
    layers = [torch.ones(2), torch.zeros(2)]
    assert torch.allclose(lightgcn_layer_combine(layers), torch.tensor([0.5, 0.5]))
    scores = score_dot(torch.ones(2), torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    assert torch.allclose(scores, torch.tensor([1.0, 2.0]))


def test_propagate_single_user_no_fake():
    u = torch.ones(2, requires_grad=True)
    items = torch.ones(2, 2, requires_grad=True)
    fake = torch.tensor([False, False])
    uc, ic = propagate_semidfegl(
        u, items, fake, torch.zeros(0, 2), 2.0, torch.zeros(0), 1, 1
    )
    assert uc.shape == (2,)
    assert ic.shape == (2, 2)
    loss = uc.sum() + ic.sum()
    loss.backward()
    assert u.grad is not None
    assert items.grad is not None


def test_fuzzy_cmeans_shapes():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.05, 0.0], [1.05, 1.0]], dtype=np.float64)
    membership, centres = fuzzy_cmeans(
        pts, num_clusters=2, fuzziness=2.0, max_iter=50, tol=1e-5,
        rng=np.random.default_rng(0),
    )
    assert membership.shape == (4, 2)
    assert centres.shape == (2, 2)
    np.testing.assert_allclose(membership.sum(axis=1), np.ones(4), atol=1e-6)
