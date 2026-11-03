"""FedVAE scoring: train-history input, no masking, finite empty-history scores."""

import numpy as np
import pandas as pd
import torch

from fedreclab.federated.base_client import ClientStateBase
from fedreclab.methods.fedvae.fedvae_model import MultVAE


def test_scores_align_with_candidates_and_use_train_items(
    fedvae_client_and_request,
):
    client, request = fedvae_client_and_request()
    resp = client.local_train(request)
    candidates = np.array([0, 2, 5])
    scores = client.compute_scores(
        0,
        {0: candidates},
        request.global_payload,
        resp.updated_client_state,
        {},
    )
    assert list(scores) == [0]
    assert scores[0].shape == (3,)

    ref = MultVAE(
        num_items=client.num_items,
        hidden_dims=client.hidden_dims,
        latent_dim=client.latent_dim,
        dropout=client.dropout,
    )
    ref.load_server_weights(request.global_payload)
    ref.eval()
    x = client._binary_vector(np.array([1, 2], dtype=np.int64), client.device)
    with torch.no_grad():
        logits, mu, _ = ref(x.unsqueeze(0))
        expected = logits[0, candidates].cpu().numpy()
        assert torch.equal(logits, ref.decode(mu))
    np.testing.assert_allclose(scores[0], expected, rtol=1e-5, atol=1e-5)


def test_scores_do_not_mask_training_items(fedvae_client_and_request):
    client, request = fedvae_client_and_request()
    state = request.client_state
    mixed = np.array([1, 3, 2])
    scores = client.compute_scores(
        0, {0: mixed}, request.global_payload, state, {}
    )
    assert np.isfinite(scores[0]).all()
    # Training items in the candidate list are scored, not set to -inf.
    assert not np.isneginf(scores[0][0])
    assert not np.isneginf(scores[0][2])


def test_empty_history_user_gives_finite_scores(fedvae_client_and_request):
    empty = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    client, request = fedvae_client_and_request(
        training_data=empty, users_in_client=[0]
    )
    state = ClientStateBase(
        client_id=0,
        local_parameters={"train_items_per_user": {0: np.array([], dtype=np.int64)}},
        optimizer_state=None,
    )
    scores = client.compute_scores(
        0,
        {0: np.array([0, 1, 4])},
        request.global_payload,
        state,
        {},
    )
    assert scores[0].shape == (3,)
    assert np.isfinite(scores[0]).all()
