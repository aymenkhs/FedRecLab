"""MetaRecommender shapes, Table 4 width, shared hidden_layer_3, item_rows."""

import pytest
import torch
import torch.nn.functional as F

from fedreclab.methods.metamf.metamf_model import (
    MetaRecommender,
    predict_ratings,
    resolve_rp_hidden_dim,
    unbatch_phi,
)
from tests.metamf.conftest import TINY


def test_resolve_rp_hidden_dim_follows_item_emb_over_four():
    assert resolve_rp_hidden_dim(32, None) == 8
    assert resolve_rp_hidden_dim(8, None) == 2
    assert resolve_rp_hidden_dim(32, 3) == 3
    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_rp_hidden_dim(3, None)
    with pytest.raises(ValueError, match="must be >= 1"):
        MetaRecommender(num_users=2, num_items=4, item_emb_dim=3, rp_hidden_dim=None)


def test_table4_scale_builds_rp_width_two():
    model = MetaRecommender(
        num_users=3,
        num_items=5,
        user_emb_dim=8,
        item_emb_dim=8,
        item_mem_num=8,
        mem_size=32,
        hidden_size=128,
        rp_hidden_dim=None,
    )
    assert model.rp_hidden_dim == 2
    phi = model.generate(torch.tensor([0]))
    assert phi["w1"].shape == (1, 8, 2)
    assert phi["w2"].shape == (1, 2, 1)


def test_generated_shapes_match_tiny_config():
    model = MetaRecommender(num_users=3, num_items=6, **TINY)
    phi = model.generate(torch.tensor([0, 2]), item_payload="factors")
    assert phi["item_left"].shape == (2, 6, 2)
    assert phi["item_right"].shape == (2, 2, 4)
    assert phi["w1"].shape == (2, 4, 1)
    assert phi["b1"].shape == (2, 1, 1)
    assert phi["w2"].shape == (2, 1, 1)
    assert phi["b2"].shape == (2, 1, 1)
    full = model.generate(torch.tensor([0]), item_payload="full")
    assert full["item_embeddings"].shape == (1, 6, 4)


def test_hidden_layer_3_feeds_both_item_heads():
    model = MetaRecommender(num_users=2, num_items=5, **TINY)
    assert model.emb_layer_1.in_features == model.hidden_size
    assert model.emb_layer_2.in_features == model.hidden_size
    assert model.hidden_layer_3.out_features == model.hidden_size
    named = dict(model.named_modules())
    assert "hidden_layer_3" in named
    assert "hidden_layer_4" not in named


def test_predict_ratings_matches_reference_formula():
    model = MetaRecommender(num_users=2, num_items=5, **TINY)
    user = torch.tensor([0])
    items = torch.tensor([1, 3])
    phi = unbatch_phi(model.generate(user, item_payload="full"), 0)
    got = predict_ratings(phi, items)
    item_emb = phi["item_embeddings"].index_select(0, items)
    hidden = F.relu(item_emb @ phi["w1"] + phi["b1"])
    expected = (hidden @ phi["w2"] + phi["b2"]).reshape(-1)
    assert torch.allclose(got, expected)


def test_factor_and_full_predict_identically():
    model = MetaRecommender(num_users=2, num_items=5, **TINY)
    user = torch.tensor([1])
    items = torch.tensor([0, 2, 4])
    factors = unbatch_phi(model.generate(user, item_payload="factors"), 0)
    full = unbatch_phi(model.generate(user, item_payload="full"), 0)
    assert torch.allclose(
        predict_ratings(factors, items), predict_ratings(full, items), atol=1e-6
    )


def _cases():
    return [
        torch.tensor([1, 3, 4]),
        torch.tensor([2, 2, 0]),
        torch.tensor([4, 1, 0, 3]),
        torch.tensor([2]),
        torch.tensor([], dtype=torch.long),
        torch.tensor([0, 5]),
    ]


@pytest.mark.parametrize("candidates", _cases())
def test_item_rows_matches_full_generation(candidates):
    model = MetaRecommender(num_users=2, num_items=6, **TINY)
    user = torch.tensor([0])
    sliced = model.item_rows(user, candidates, item_payload="factors")
    full = model.generate(user, item_payload="factors")
    if candidates.numel() == 0:
        assert sliced["item_left"].shape == (1, 0, model.item_mem_num)
        return
    expected_left = full["item_left"][0].index_select(0, candidates)
    expected_right = full["item_right"][0]
    assert torch.allclose(sliced["item_left"][0], expected_left, atol=1e-6)
    assert torch.allclose(sliced["item_right"][0], expected_right, atol=1e-6)
    for key in ("w1", "b1", "w2", "b2"):
        assert torch.allclose(sliced[key], full[key], atol=1e-6)
    sliced_full = model.item_rows(user, candidates, item_payload="full")
    expected_emb = full["item_left"][0].index_select(0, candidates) @ full["item_right"][0]
    assert torch.allclose(sliced_full["item_embeddings"][0], expected_emb, atol=1e-6)


def test_state_round_trip_copies_in_place():
    src = MetaRecommender(num_users=2, num_items=4, **TINY)
    dst = MetaRecommender(num_users=2, num_items=4, **TINY)
    param_ids = [id(p) for p in dst.parameters()]
    dst.load_server_weights(src.get_server_state_groups())
    assert [id(p) for p in dst.parameters()] == param_ids
    for p1, p2 in zip(src.parameters(), dst.parameters()):
        assert torch.equal(p1, p2)


def test_seeded_init_is_reproducible():
    g1 = torch.Generator()
    g1.manual_seed(7)
    a = MetaRecommender(num_users=2, num_items=4, generator=g1, **TINY)
    g2 = torch.Generator()
    g2.manual_seed(7)
    b = MetaRecommender(num_users=2, num_items=4, generator=g2, **TINY)
    for p1, p2 in zip(a.parameters(), b.parameters()):
        assert torch.equal(p1, p2)
