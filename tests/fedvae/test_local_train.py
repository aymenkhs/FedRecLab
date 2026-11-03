"""FedVAE local_train uploads a dense autograd gradient of the whole VAE."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from fedreclab.methods.fedvae.fedvae_model import MultVAE


def test_upload_keys_and_shapes_match_server_state(fedvae_client_and_request):
    client, request = fedvae_client_and_request()
    resp = client.local_train(request)
    uploaded = resp.updates_packages["vae"].param_value
    server = request.global_payload["vae"]
    assert set(uploaded) == set(server)
    for name, tensor in uploaded.items():
        assert tensor.shape == server[name].shape
        assert tensor.device.type == "cpu"


def test_upload_equals_autograd_gradient_for_the_same_seed(fedvae_client_and_request):
    client, request = fedvae_client_and_request()
    resp = client.local_train(request)
    uploaded = resp.updates_packages["vae"].param_value

    ref = MultVAE(
        num_items=client.num_items,
        hidden_dims=client.hidden_dims,
        latent_dim=client.latent_dim,
        dropout=client.dropout,
    )
    ref.load_server_weights(request.global_payload)
    ref.train()
    batch = client._user_item_batch(
        request.users_in_client, request.training_data, device=client.device
    )
    generator = client._round_generator(
        request.seed, request.client_id, request.current_round
    )
    ref.zero_grad(set_to_none=True)
    logits, mu, logvar = ref(batch, generator=generator)
    loss = client._mult_vae_loss(logits, batch, mu, logvar)
    loss.backward()
    for name, param in ref.named_parameters():
        np.testing.assert_allclose(
            uploaded[name].numpy(),
            param.grad.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


def test_payload_is_not_modified(fedvae_client_and_request):
    client, request = fedvae_client_and_request()
    before = copy.deepcopy(request.global_payload["vae"])
    client.local_train(request)
    after = request.global_payload["vae"]
    for name, tensor in before.items():
        assert torch.equal(torch.as_tensor(after[name]), torch.as_tensor(tensor))


def test_num_training_examples_is_the_interaction_count(fedvae_client_and_request):
    data = pd.DataFrame(
        {"user_id": [0, 0, 0], "item_id": [1, 2, 3], "rating": [1.0, 1.0, 1.0]}
    )
    client, request = fedvae_client_and_request(training_data=data)
    resp = client.local_train(request)
    assert resp.num_training_examples == 3


def test_non_finite_loss_raises(fedvae_client_and_request):
    client, request = fedvae_client_and_request()

    def exploding(*_args, **_kwargs):
        return torch.tensor(float("nan"))

    client._mult_vae_loss = exploding
    with pytest.raises(ValueError, match="Non-finite"):
        client.local_train(request)


def test_empty_history_user_is_a_zero_vector(fedvae_client_and_request):
    data = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    client, request = fedvae_client_and_request(
        training_data=data, users_in_client=[0]
    )
    resp = client.local_train(request)
    assert resp.num_training_examples == 0
    uploaded = resp.updates_packages["vae"].param_value
    for tensor in uploaded.values():
        assert torch.isfinite(tensor).all()
