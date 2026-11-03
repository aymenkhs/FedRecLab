"""Trainable neighbour rows take the same local Adam steps as the owner.

The uploaded row is the parameter after those steps. It is not one
plain SGD step on a gradient averaged over batches.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


def _install_neighbours(request, embeddings, user_ids):
    expansion = request.extra_payload["expanded_user_data"][0]
    expansion["neighbor_embeddings"] = np.asarray(embeddings, dtype=np.float32)
    expansion["neighbor_user_ids"] = np.asarray(user_ids, dtype=np.int64)
    return expansion


def _one_neighbour(hidden, history_len, neighbor_len, user_id=4):
    embeddings = np.zeros((history_len, neighbor_len, hidden), dtype=np.float32)
    embeddings[0, 0] = np.linspace(0.1, 0.8, hidden, dtype=np.float32)
    user_ids = np.full((history_len, neighbor_len), -1, dtype=np.int64)
    user_ids[0, 0] = user_id
    return embeddings, user_ids


def _adam_reference(client, request, epochs):
    """Replay ``epochs`` Adam steps on a fresh copy of the round's weights.

    One batch holds every training row. With the fixture batch size that
    is also what ``local_train`` does, and the mean loss does not depend
    on row order.
    """
    model = FedPerGNNModel(
        num_items=client._num_items,
        hidden_dim=client.hidden_dim,
        history_len=client.history_len,
        neighbor_len=client.neighbor_len,
    )
    payload = request.global_payload
    model.load_server_state(payload["model_state"])
    model.load_item_embeddings(payload["item_embeddings"])
    model.train()
    weight = request.client_state.local_parameters["user_embedding.weight"]
    user = torch.nn.Parameter(torch.as_tensor(weight).float().clone())
    expansion = request.extra_payload["expanded_user_data"][request.client_id]
    owner_id = int(request.users_in_client[0])
    param, user_ids, index, frozen = client._bind_neighbour_parameter(
        expansion["neighbor_embeddings"],
        expansion["neighbor_user_ids"],
        owner_id,
        torch.device("cpu"),
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [user, param], lr=client.lr
    )
    history = torch.as_tensor(expansion["history_item_ids"], dtype=torch.long)
    data = request.training_data
    item_ids = torch.as_tensor(data["item_id"].to_numpy(dtype=np.int64))
    labels = torch.as_tensor(data["rating"].to_numpy(dtype=np.float32))
    plain = None
    for _ in range(epochs):
        optimizer.zero_grad()
        neighbours = client._neighbor_embeddings_from_parameter(
            param, index, frozen
        )
        user_repr = model.user_representation(history, user, neighbours)
        loss = torch.nn.functional.mse_loss(model.score(user_repr, item_ids), labels)
        loss.backward()
        if plain is None:
            plain = (param.detach() - client.lr * param.grad.detach()).clone()
        optimizer.step()
    return user_ids, param.detach().cpu().numpy(), plain.cpu().numpy()


def _uploaded_neighbour(response, user_id):
    upload = response.updates_packages["user_embeddings"].param_value
    rows = [int(row) for row in upload["rows"]]
    return rows, upload["values"][rows.index(user_id)]


def test_trainable_neighbour_matches_adam_not_one_sgd_step(
    fedpergnn_client_and_request,
):
    """One local epoch uploads the Adam row, not ``start - lr * grad``.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If the upload misses the neighbour, matches plain SGD, or
        disagrees with Adam.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    request.training_data = pd.DataFrame(
        {"user_id": [0], "item_id": [1], "rating": [4.0]}
    )
    embeddings, user_ids = _one_neighbour(
        client.hidden_dim, client.history_len, client.neighbor_len
    )
    _install_neighbours(request, embeddings, user_ids)

    response = client.local_train(request)
    rows, updated = _uploaded_neighbour(response, 4)
    _, expected, plain = _adam_reference(client, request, epochs=1)

    assert rows == [0, 4]
    assert updated.shape == (client.hidden_dim,)
    np.testing.assert_allclose(updated, expected[0], rtol=1e-5, atol=1e-6)
    assert not np.allclose(updated, plain[0])
    assert not np.allclose(updated, embeddings[0, 0])


def test_two_local_epochs_take_two_adam_steps(fedpergnn_client_and_request):
    """Neighbour rows move on every epoch, with the same optimiser.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If two epochs collapse to a single Adam step.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    client.local_epochs = 2
    request.training_data = pd.DataFrame(
        {"user_id": [0], "item_id": [1], "rating": [4.0]}
    )
    embeddings, user_ids = _one_neighbour(
        client.hidden_dim, client.history_len, client.neighbor_len
    )
    _install_neighbours(request, embeddings, user_ids)

    response = client.local_train(request)
    _, updated = _uploaded_neighbour(response, 4)
    _, two_steps, _ = _adam_reference(client, request, epochs=2)
    _, one_step, _ = _adam_reference(client, request, epochs=1)

    np.testing.assert_allclose(updated, two_steps[0], rtol=1e-5, atol=1e-6)
    assert not np.allclose(updated, one_step[0])


def test_repeated_neighbour_id_is_one_shared_row(fedpergnn_client_and_request):
    """Two slots of one user stay one vector and upload one row.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If the id is uploaded twice or the row disagrees with Adam on
        the shared parameter.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    request.training_data = pd.DataFrame(
        {"user_id": [0], "item_id": [1], "rating": [4.0]}
    )
    hidden = client.hidden_dim
    embeddings, user_ids = _one_neighbour(hidden, client.history_len, client.neighbor_len)
    embeddings[1, 0] = embeddings[0, 0]
    user_ids[1, 0] = 4
    _install_neighbours(request, embeddings, user_ids)

    param, bound_ids, index, _ = client._bind_neighbour_parameter(
        embeddings, user_ids, owner_id=0, device=torch.device("cpu")
    )
    assert bound_ids == [4]
    assert int((index == 1).sum()) == 2
    assert param.shape == (1, hidden)

    response = client.local_train(request)
    rows, updated = _uploaded_neighbour(response, 4)
    _, expected, _ = _adam_reference(client, request, epochs=1)

    assert rows.count(4) == 1
    np.testing.assert_allclose(updated, expected[0], rtol=1e-5, atol=1e-6)


def test_owner_and_padding_are_not_extra_user_rows(fedpergnn_client_and_request):
    """The owner is uploaded once. Empty slots are not user ids.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If the owner id is repeated or a negative id is uploaded.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    hidden = client.hidden_dim
    embeddings = np.zeros((client.history_len, client.neighbor_len, hidden), dtype=np.float32)
    embeddings[0, 0] = np.linspace(-0.4, 0.4, hidden, dtype=np.float32)
    embeddings[0, 1] = np.linspace(0.2, 0.9, hidden, dtype=np.float32)
    user_ids = np.full((client.history_len, client.neighbor_len), -1, dtype=np.int64)
    user_ids[0, 0] = 0
    user_ids[0, 1] = 4
    _install_neighbours(request, embeddings, user_ids)

    response = client.local_train(request)
    rows, updated = _uploaded_neighbour(response, 4)
    assert rows == [0, 4]
    assert not np.allclose(updated, embeddings[0, 1])


def test_conflicting_payload_rows_for_one_user_raise(fedpergnn_client_and_request):
    """One user id cannot start from two payload vectors.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    ValueError
        If the same neighbour id is paired with two different rows.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    hidden = client.hidden_dim
    embeddings = np.zeros((client.history_len, client.neighbor_len, hidden), dtype=np.float32)
    embeddings[0, 0] = 0.2
    embeddings[1, 0] = -0.2
    user_ids = np.full((client.history_len, client.neighbor_len), -1, dtype=np.int64)
    user_ids[0, 0] = 4
    user_ids[1, 0] = 4
    _install_neighbours(request, embeddings, user_ids)

    with pytest.raises(ValueError, match="two different payload rows"):
        client.local_train(request)


def test_neighbour_moments_are_not_stored(fedpergnn_client_and_request):
    """Adam moments for neighbour rows do not enter client state.

    The next round re-initialises those rows from the server mirror, so
    moments from this round's parameter would not match it.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If a neighbour moment is stored, or the owner moment is dropped
        under ``optimizer_state_scope="all"``.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True,
        optimizer_state_scope="all",
    )
    embeddings, user_ids = _one_neighbour(
        client.hidden_dim, client.history_len, client.neighbor_len
    )
    _install_neighbours(request, embeddings, user_ids)

    response = client.local_train(request)
    keys = set(response.updated_client_state.optimizer_state or {})
    # GAT layers such as neighbor_proj are assembled parameters and may
    # be stored under scope "all". The round-local neighbour rows are not
    # among them.
    assembled = {name for name, _ in client._skeleton.named_parameters()}
    assembled.add("user_embedding.weight")
    assert "user_embedding.weight" in keys
    assert keys <= assembled


def test_no_local_batch_does_not_upload_neighbour_rows(
    fedpergnn_client_and_request,
):
    """A client with no training rows does not vote the mirror back.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If a neighbour id is uploaded without a local step.
    """
    client, request = fedpergnn_client_and_request(
        neighbor_embeddings_trainable=True
    )
    request.training_data = pd.DataFrame(
        {"user_id": [], "item_id": [], "rating": []}
    )
    embeddings, user_ids = _one_neighbour(
        client.hidden_dim, client.history_len, client.neighbor_len
    )
    _install_neighbours(request, embeddings, user_ids)

    response = client.local_train(request)
    rows = [
        int(row)
        for row in response.updates_packages["user_embeddings"].param_value["rows"]
    ]
    assert rows == [0]
