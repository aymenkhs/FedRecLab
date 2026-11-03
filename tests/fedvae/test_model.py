"""MultVAE shapes, state round-trip, eval uses mu, seeded dropout/eps."""

import torch

from fedreclab.methods.fedvae.fedvae_model import MultVAE


def test_layer_shapes_match_hidden_and_latent():
    m = MultVAE(num_items=6, hidden_dims=(4,), latent_dim=2, dropout=0.5)
    assert m.enc_layers[0].in_features == 6
    assert m.enc_layers[0].out_features == 4
    assert m.enc_layers[1].in_features == 4
    assert m.enc_layers[1].out_features == 4
    assert m.dec_layers[0].in_features == 2
    assert m.dec_layers[0].out_features == 4
    assert m.dec_layers[1].in_features == 4
    assert m.dec_layers[1].out_features == 6
    x = torch.zeros(3, 6)
    x[0, 1] = 1.0
    logits, mu, logvar = m(x)
    assert logits.shape == (3, 6)
    assert mu.shape == (3, 2)
    assert logvar.shape == (3, 2)


def test_state_round_trip_restores_named_parameters():
    src = MultVAE(num_items=5, hidden_dims=(3,), latent_dim=2)
    dst = MultVAE(num_items=5, hidden_dims=(3,), latent_dim=2)
    dst.load_server_weights(src.get_server_state_groups())
    for (n1, p1), (n2, p2) in zip(src.named_parameters(), dst.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_eval_is_deterministic_and_uses_mu():
    m = MultVAE(num_items=5, hidden_dims=(3,), latent_dim=2, dropout=0.5)
    m.eval()
    x = torch.zeros(2, 5)
    x[0, 0] = 1.0
    x[1, 2] = 1.0
    logits_a, mu_a, logvar_a = m(x)
    logits_b, mu_b, logvar_b = m(x)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(mu_a, mu_b)
    assert torch.equal(logvar_a, logvar_b)
    assert torch.equal(logits_a, m.decode(mu_a))


def test_same_generator_seed_gives_same_dropout_and_eps():
    m = MultVAE(num_items=5, hidden_dims=(3,), latent_dim=2, dropout=0.5)
    m.train()
    x = torch.ones(2, 5)
    gen_a = torch.Generator()
    gen_a.manual_seed(123)
    out_a = m(x, generator=gen_a)
    gen_b = torch.Generator()
    gen_b.manual_seed(123)
    out_b = m(x, generator=gen_b)
    for a, b in zip(out_a, out_b):
        assert torch.equal(a, b)


def test_seeded_init_is_reproducible():
    g1 = torch.Generator()
    g1.manual_seed(7)
    a = MultVAE(num_items=4, hidden_dims=(3,), latent_dim=2, generator=g1)
    g2 = torch.Generator()
    g2.manual_seed(7)
    b = MultVAE(num_items=4, hidden_dims=(3,), latent_dim=2, generator=g2)
    for p1, p2 in zip(a.parameters(), b.parameters()):
        assert torch.equal(p1, p2)
