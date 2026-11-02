"""FedPerGNN skeleton tensors vs the payload the server aggregates.

The module is server-only: item table, GAT layers, and the history-with-
neighbours projection. The owner's user embedding is not on the module.
"""

from src.fedreclab.federated.utils import assert_skeleton_server_coverage
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


def test_history_with_neighbors_is_in_the_server_state():
    model = FedPerGNNModel(num_items=5, hidden_dim=8, history_len=4, neighbor_len=3)
    state = model.get_server_state()
    assert "history_with_neighbors.weight" in state
    assert "history_with_neighbors.bias" in state
    assert "item_embeddings.weight" in state
    assert "item_embeddings.weight" not in model.get_model_state()
    assert "user_embedding.weight" not in state
    assert "output_layer.weight" not in state
    assert "output_layer.bias" not in state
    assert_skeleton_server_coverage(model, state)
