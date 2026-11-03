"""FIMS-Uni sampler rebuild and sample shapes."""

import torch

from fedreclab.methods.efvae.efvae_sampler import FIMSUniSampler, kmeans


def test_kmeans_shapes():
    pts = torch.randn(20, 4)
    gen = torch.Generator().manual_seed(0)
    labels, centres = kmeans(pts, 3, generator=gen)
    assert labels.shape == (20,)
    assert centres.shape == (3, 4)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 3


def test_fims_update_and_sample():
    emb = torch.randn(8, 4)
    gen = torch.Generator().manual_seed(1)
    s = FIMSUniSampler(num_items=8, num_cluster=2, num_neg=3, generator=gen)
    s.update(emb, generator=gen)
    pos_lp, neg, neg_lp = s(torch.randn(2, 4), torch.tensor([[0, 1], [2, 3]]))
    assert pos_lp.shape == (2, 2)
    assert neg.shape == (2, 3)
    assert neg_lp.shape == (2, 3)
    assert int(neg.min()) >= 0
    assert int(neg.max()) < 8


def test_fims_state_dict_roundtrip():
    emb = torch.randn(8, 4)
    gen = torch.Generator().manual_seed(2)
    s = FIMSUniSampler(num_items=8, num_cluster=2, num_neg=3, generator=gen)
    s.update(emb, generator=gen)
    state = s.state_dict()
    s2 = FIMSUniSampler(num_items=8, num_cluster=2, num_neg=3)
    s2.load_state_dict(state)
    q = torch.randn(1, 4)
    gen2 = torch.Generator().manual_seed(3)
    s.generator = gen2
    s2.generator = torch.Generator().manual_seed(3)
    _, neg_a, _ = s(q)
    _, neg_b, _ = s2(q)
    assert torch.equal(neg_a, neg_b)
