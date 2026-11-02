"""EFMultVAE shapes, groups, and eval path."""

import torch

from src.fedreclab.methods.efvae.efvae_fcis import EFVAEFCISServer
from src.fedreclab.methods.efvae.efvae import EFVAE
from src.fedreclab.methods.efvae.efvae_client import (
    EFVAEClientAlgorithm,
    EFVAEFCISClientAlgorithm,
)
from src.fedreclab.methods.efvae.efvae_model import EFMultVAE


def test_forward_shapes():
    m = EFMultVAE(num_items=5, hidden_dims=(4,), latent_dim=2, dropout=0.0)
    m.eval()
    logits, mu, logvar = m(torch.zeros(1, 5))
    assert logits.shape == (1, 5)
    assert mu.shape == (1, 2)
    assert logvar.shape == (1, 2)
    assert torch.equal(logits, m.decode_hidden(mu) @ m.dec_embeddings.weight.T + m.dec_bias.weight.T)


def test_server_state_groups_cover_named_parameters():
    m = EFMultVAE(num_items=5, hidden_dims=(4,), latent_dim=2)
    groups = m.get_server_state_groups()
    assert set(groups) == {
        "enc_embeddings",
        "dec_embeddings",
        "dec_bias",
        "middle",
    }
    flat = {k: v for g in groups.values() for k, v in g.items()}
    named = {n for n, _ in m.named_parameters()}
    assert set(flat) == named


def test_load_roundtrip():
    g = torch.Generator().manual_seed(0)
    m = EFMultVAE(num_items=5, hidden_dims=(4,), latent_dim=2, generator=g)
    payload = m.get_server_state_groups()
    m2 = EFMultVAE(num_items=5, hidden_dims=(4,), latent_dim=2)
    m2.load_server_weights(payload)
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_extension_surface_subclasses_fcis_bases():
    assert issubclass(EFVAE, EFVAEFCISServer)
    assert issubclass(EFVAEClientAlgorithm, EFVAEFCISClientAlgorithm)
