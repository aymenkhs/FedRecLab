"""FedPerGNN takes its widths from the client and rejects the duplicate knobs."""

import pandas as pd
import pytest

from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from fedreclab.data.dataset import InteractionDataset
from fedreclab.data.split import RatioSplitStrategy
from fedreclab.federated.aggregation import SimpleAvg
from fedreclab.federated.client_selection import FullClientSelection
from fedreclab.methods.fedpergnn.fedpergnn import FedPerGNN
from fedreclab.methods.fedpergnn.fedpergnn_client import FedPerGNNClientAlgorithm

NUM_ITEMS = 6


def _dataset(n_users: int = 2) -> InteractionDataset:
    rows = []
    for user_id in range(n_users):
        rows.append(
            {
                "user_id": user_id,
                "item_id": user_id % NUM_ITEMS,
                "rating": 1.0,
                "split": "train",
            }
        )
    df = pd.DataFrame(rows)
    dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=False
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=0,
        include_validation=False,
        include_test=False,
        splits_included=True,
        time_col=None,
    )
    dataset.partition_clients()
    return dataset


def test_server_copies_widths_from_the_client():
    """Expansion and the global table follow the client, not a second set of defaults.

    Raises
    ------
    AssertionError
        If the server keeps its own widths, stores the trainable flag,
        or builds a user table or history of a different size.
    """
    hidden, hist, neigh = 5, 2, 1
    client = FedPerGNNClientAlgorithm(
        hidden_dim=hidden,
        history_len=hist,
        neighbor_len=neigh,
        neighbor_embeddings_trainable=True,
    )
    dataset = _dataset()
    server = FedPerGNN(
        warmup_rounds=0,
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
        device="cpu",
        seed=0,
    )
    assert server.hidden_dim == hidden
    assert server.history_len == hist
    assert server.neighbor_len == neigh
    assert not hasattr(server, "neighbor_embeddings_trainable")
    assert client.neighbor_embeddings_trainable is True
    assert server.global_parameters["user_embeddings"].shape == (dataset.num_users, hidden)
    assert server.global_parameters["item_embeddings"].shape == (dataset.num_items, hidden)
    weight = server.global_parameters["model_state"]["history_item_proj.weight"]
    assert tuple(weight.shape) == (hidden, hidden)
    owner = server.client_states[0].local_parameters["user_embedding.weight"]
    assert tuple(owner.shape) == (1, hidden)

    server._before_training()
    cached = server._expansion_cache[0]
    assert cached["history_item_ids"].shape == (hist,)
    assert cached["neighbor_user_ids"].shape == (hist, neigh)


def test_server_rejects_duplicated_client_settings():
    """The old server knobs name the client as the place to set them.

    Raises
    ------
    AssertionError
        If constructing with those arguments succeeds, or if the error
        omits one of the four names.
    """
    with pytest.raises(TypeError, match="FedPerGNNClientAlgorithm") as exc:
        FedPerGNN(
            hidden_dim=4,
            history_len=2,
            neighbor_len=1,
            neighbor_embeddings_trainable=True,
        )
    message = str(exc.value)
    for name in (
        "hidden_dim",
        "history_len",
        "neighbor_len",
        "neighbor_embeddings_trainable",
    ):
        assert name in message


def test_server_requires_widths_on_the_client():
    """A client algorithm without the three widths fails at construction.

    Raises
    ------
    AssertionError
        If a stand-in client is accepted.
    """

    class _Bare:
        pass

    with pytest.raises(TypeError, match="hidden_dim"):
        FedPerGNN(client_algorithm=_Bare())
