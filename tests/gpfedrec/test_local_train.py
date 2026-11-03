"""GPFedRec local training: regulariser, upload, noise, learning rates, seeding."""

import numpy as np
import pytest
import torch

from fedreclab.methods.gpfedrec.gpfedrec_client import GPFedRecClientAlgorithm
from fedreclab.methods.gpfedrec.gpfedrec_model import initial_item_table

LATENT = 4
HIDDEN = (8, 4)
NUM_ITEMS = 8


def _upload(response):
    return response.updates_packages["item_embeddings"].param_value


def test_upload_is_a_dense_full_table(make_client_and_request):
    client, request = make_client_and_request()
    response = client.local_train(request)
    upload = _upload(response)
    assert isinstance(upload, torch.Tensor)
    assert upload.shape == (NUM_ITEMS, LATENT)
    assert upload.dtype == torch.float32
    assert response.num_training_examples == 2 * (1 + client.neg_samples)


def test_reg_pulls_the_table_toward_r_i(make_client_and_request):
    q = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=0)
    r = torch.zeros_like(q)

    def distance(reg):
        client, request = make_client_and_request(
            global_payload={"item_embeddings": q, "personalized_item_embeddings": r},
            reg=reg,
            item_lr=1.0,
            local_epochs=3,
        )
        table = client.local_train(request).updated_client_state.local_parameters[
            "local_item_embeddings"
        ]
        return float(torch.linalg.norm(table - r))

    assert distance(5.0) < distance(0.0)


def test_state_keeps_the_noise_free_table_when_noise_is_on(make_client_and_request):
    client, request = make_client_and_request(ldp_laplace_scale=0.5)
    noisy = client.local_train(request)
    client_clean, request_clean = make_client_and_request(ldp_laplace_scale=0.0)
    clean = client_clean.local_train(request_clean)

    stored_noisy = noisy.updated_client_state.local_parameters["local_item_embeddings"]
    stored_clean = clean.updated_client_state.local_parameters["local_item_embeddings"]
    # Noise uses its own RNG stream, so the trained tables are identical.
    assert torch.equal(stored_noisy, stored_clean)
    assert torch.equal(_upload(clean), stored_clean)
    assert not torch.allclose(_upload(noisy), stored_noisy)


def test_learning_rate_formulas_and_overrides(make_client_and_request):
    client, _ = make_client_and_request()
    assert client.effective_user_lr() == pytest.approx(0.1 / 1.0 * 80 - 0.1)
    assert client.effective_item_lr() == pytest.approx(0.1 * NUM_ITEMS * 80 - 0.1)
    half, _ = make_client_and_request(participation_ratio=0.5)
    assert half.effective_user_lr() == pytest.approx(0.1 / 0.5 * 80 - 0.1)
    fixed, _ = make_client_and_request(user_lr=2.0, item_lr=3.0)
    assert fixed.effective_user_lr() == 2.0
    assert fixed.effective_item_lr() == 3.0


def test_same_seed_gives_identical_training(make_client_and_request):
    first, request_a = make_client_and_request()
    second, request_b = make_client_and_request()
    assert torch.equal(_upload(first.local_train(request_a)), _upload(second.local_train(request_b)))
    third, request_c = make_client_and_request(seed=1)
    assert not torch.equal(_upload(first.local_train(request_a)), _upload(third.local_train(request_c)))


def test_negatives_never_hit_training_items(make_client_and_request):
    client, _ = make_client_and_request()
    rng = np.random.default_rng(0)
    positives = {0: np.array([0, 2, 5], dtype=np.int64)}
    users, items, labels = client._build_epoch_samples(positives, rng)
    negatives = items[labels == 0]
    assert negatives.size == 3 * client.neg_samples
    assert not np.isin(negatives, positives[0]).any()
    assert ((negatives >= 0) & (negatives < NUM_ITEMS)).all()


def test_non_finite_loss_raises(make_client_and_request):
    q = initial_item_table(NUM_ITEMS, LATENT, 1.0, seed=0)
    q[1, 0] = float("nan")
    client, request = make_client_and_request(
        global_payload={"item_embeddings": q, "personalized_item_embeddings": q}
    )
    with pytest.raises(ValueError, match="Non-finite"):
        client.local_train(request)


# --- template seed ----------------------------------------------------------


def test_server_copies_seed_into_unset_init_seed(make_server):
    server = make_server(seed=7)
    assert server.client_algorithm.init_seed == 7


def test_explicit_client_init_seed_takes_precedence(make_server):
    client = GPFedRecClientAlgorithm(latent_dim=LATENT, hidden_layers=HIDDEN, init_seed=3)
    server = make_server(client=client, seed=7)
    assert server.client_algorithm.init_seed == 3


def test_missing_seed_raises(make_server):
    with pytest.raises(ValueError, match="init_seed"):
        make_server(seed=None)


def test_every_client_starts_from_the_same_template(make_server):
    server = make_server()
    first = server.client_states[0].local_parameters
    for state in server.client_states.values():
        local = state.local_parameters
        assert torch.equal(local["user_embeddings"], first["user_embeddings"])
        for name, value in local["score_function"].items():
            assert torch.equal(value, first["score_function"][name])


def _template(make_client_and_request, before=None):
    if before is not None:
        before()
    client, request = make_client_and_request()
    local = request.client_state.local_parameters
    return local["user_embeddings"], local["score_function"]


def test_template_is_unaffected_by_earlier_global_draws(make_client_and_request):
    user_a, score_a = _template(make_client_and_request)
    user_b, score_b = _template(make_client_and_request, before=lambda: torch.rand(100))
    assert torch.equal(user_a, user_b)
    for name in score_a:
        assert torch.equal(score_a[name], score_b[name])


def test_template_leaves_the_cpu_rng_state_unchanged(make_client_and_request):
    before = torch.get_rng_state()
    make_client_and_request()
    assert torch.equal(torch.get_rng_state(), before)


def test_template_does_not_reseed_cuda(make_client_and_request, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: calls.append("all"))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *a, **k: calls.append("one"))
    make_client_and_request()
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_template_leaves_the_cuda_rng_state_unchanged(make_client_and_request):
    before = torch.cuda.get_rng_state_all()
    make_client_and_request()
    after = torch.cuda.get_rng_state_all()
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_template_user_vector_differs_from_the_item_table_start(make_server):
    server = make_server()
    user_vector = server.client_states[0].local_parameters["user_embeddings"][0]
    item_row = server.global_parameters["item_embeddings"][0]
    assert not torch.allclose(user_vector, item_row)


def test_fresh_clients_follow_each_server_seed(make_server, make_tiny_dataset):
    dataset = make_tiny_dataset()
    first = make_server(dataset=dataset, seed=0)
    second = make_server(dataset=dataset, seed=1)
    assert first.client_algorithm.init_seed == 0
    assert second.client_algorithm.init_seed == 1
    assert not torch.allclose(
        first.global_parameters["item_embeddings"],
        second.global_parameters["item_embeddings"],
    )
    assert not torch.allclose(
        first.client_states[0].local_parameters["user_embeddings"],
        second.client_states[0].local_parameters["user_embeddings"],
    )
