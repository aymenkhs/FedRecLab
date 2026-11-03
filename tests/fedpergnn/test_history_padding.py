"""Catalogue item 0 is a real item. Empty history slots are -1."""

import numpy as np
import pandas as pd
import torch

from src.fedreclab.methods.fedpergnn.fedpergnn import FedPerGNN
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


def test_score_reads_catalogue_item_zero_at_embedding_row_one():
    """Id 0 is embedding row 1. The padding row is not that item.

    Raises
    ------
    AssertionError
        If scoring id 0 reads the padding row, or if the exchanged
        table still contains that row.
    """
    model = FedPerGNNModel(num_items=3, hidden_dim=4, history_len=1, neighbor_len=1)
    assert tuple(model.item_embeddings.weight.shape) == (4, 4)
    assert tuple(model.get_item_embeddings().shape) == (3, 4)
    with torch.no_grad():
        model.item_embeddings.weight.zero_()
        model.item_embeddings.weight[1, 0] = 1.0
        model.item_embeddings.weight[0, 1] = 1.0
    scores = model.score(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0, 1]))
    assert torch.allclose(scores, torch.tensor([1.0, 0.0]))


def test_empty_history_slot_reads_the_padding_row_not_item_zero():
    """A -1 slot follows the padding row. Filling catalogue item 0 does not.

    One empty slot still receives the softmax mass, so the value stored
    on the padding row is visible. Catalogue item 0 is a different row.

    Raises
    ------
    AssertionError
        If the padding row is ignored, or if catalogue item 0 changes
        the representation of an empty slot.
    """
    model = FedPerGNNModel(num_items=3, hidden_dim=4, history_len=1, neighbor_len=1)
    user = torch.zeros(4)
    user[0] = 1.0
    history = torch.tensor([-1])
    with torch.no_grad():
        model.item_embeddings.weight.zero_()
        model.item_embeddings.weight[1].fill_(5.0)
        ignored = model.user_representation(history, user).clone()
        model.item_embeddings.weight[0].fill_(5.0)
        used = model.user_representation(history, user).clone()
        model.item_embeddings.weight[0].zero_()
        restored = model.user_representation(history, user).clone()
    assert not torch.allclose(ignored, used)
    assert torch.allclose(ignored, restored)


def test_real_item_zero_in_history_reads_its_own_row():
    """History id 0 follows embedding row 1, not the padding row.

    Raises
    ------
    AssertionError
        If changing catalogue item 0 leaves the user state unchanged.
    """
    model = FedPerGNNModel(num_items=3, hidden_dim=4, history_len=1, neighbor_len=1)
    user = torch.zeros(4)
    user[0] = 1.0
    history = torch.tensor([0])
    with torch.no_grad():
        model.item_embeddings.weight.zero_()
        model.item_embeddings.weight[0].fill_(5.0)
        baseline = model.user_representation(history, user).clone()
        model.item_embeddings.weight[1].fill_(5.0)
        moved = model.user_representation(history, user).clone()
    assert not torch.allclose(baseline, moved)


def test_item_zero_expands_to_its_neighbours():
    """A one-item history of catalogue id 0 still yields that item's co-user.

    Raises
    ------
    AssertionError
        If id 0 is stored as an empty slot, or if its co-user is missing
        while the empty slots gain a neighbour.
    """
    server = FedPerGNN.__new__(FedPerGNN)
    server.history_len = 3
    server.neighbor_len = 2
    server.expansion_seed = 0
    server.seed = 0
    server._provided_history_item_ids = None
    server._provided_neighbor_user_ids = None
    server._expansion_cache = {}
    server.num_clients = 1
    server._user_to_items = {7: np.array([0], dtype=np.int64)}
    server._item_to_users = {0: np.array([7, 8], dtype=np.int64)}

    class _Partition:
        client_ids_2_user_ids = {0: [7]}

    class _Dataset:
        client_partition = _Partition()

    server.interaction_dataset = _Dataset()
    server._construct_expanded_users()
    cached = server._expansion_cache[0]
    assert cached["history_item_ids"].tolist() == [0, -1, -1]
    neighbours = cached["neighbor_user_ids"]
    assert neighbours.shape == (3, 2)
    assert 8 in neighbours[0]
    assert np.all(neighbours[1] == -1)
    assert np.all(neighbours[2] == -1)


def test_padded_history_does_not_upload_item_zero(fedpergnn_client_and_request):
    """Empty slots are not item rows. The trained items 1 and 2 are.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory. Its history is ``[1, 2, -1, -1]``.

    Raises
    ------
    AssertionError
        If row 0 or a negative id is uploaded.
    """
    client, request = fedpergnn_client_and_request()
    upload = client.local_train(request).updates_packages["item_embeddings"].param_value
    assert set(int(row) for row in upload["rows"]) == {1, 2}


def test_trained_item_zero_is_uploaded_and_the_padding_row_stays_zero(
    fedpergnn_client_and_request,
):
    """Training catalogue item 0 updates exchanged row 0, not the pad row.

    Parameters
    ----------
    fedpergnn_client_and_request : callable
        Fixture factory for one client and its training request.

    Raises
    ------
    AssertionError
        If item 0 is missing from the upload, if that vector matches
        the downloaded row, or if the padding row moves.
    """
    client, request = fedpergnn_client_and_request()
    hidden = client.hidden_dim
    expansion = request.extra_payload["expanded_user_data"][0]
    expansion["history_item_ids"] = np.array([0, -1, -1, -1], dtype=np.int64)
    request.client_state.local_parameters["user_embedding.weight"] = torch.ones(
        (1, hidden)
    )
    request.training_data = pd.DataFrame(
        {"user_id": [0], "item_id": [0], "rating": [5.0]}
    )
    downloaded = np.asarray(request.global_payload["item_embeddings"], dtype=np.float32)
    response = client.local_train(request)
    upload = response.updates_packages["item_embeddings"].param_value
    rows = [int(row) for row in upload["rows"]]
    assert set(rows) == {0}
    uploaded = np.asarray(upload["values"][rows.index(0)])
    assert not np.allclose(uploaded, downloaded[0])
    catalogue = client._skeleton.get_item_embeddings().numpy()
    np.testing.assert_allclose(uploaded, catalogue[0])
    pad = client._skeleton.item_embeddings.weight[0].detach().cpu().numpy()
    assert np.all(pad == 0)
