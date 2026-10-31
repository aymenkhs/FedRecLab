"""The owner's user embedding lives in ClientState, not in the server table.

The server table is an expansion mirror. Training must start from the
client's own row even when the mirror holds a different vector for that user.
"""

import numpy as np
import pandas as pd
import torch

from src.fedreclab.federated.base_client import ClientStateBase, ClientTrainingRequest
from src.fedreclab.methods.fedpergnn.fedpergnn_client import FedPerGNNClientAlgorithm


def test_local_train_uses_client_state_row_not_the_server_table():
    hidden, hist, neigh = 8, 4, 3
    client = FedPerGNNClientAlgorithm(
        hidden_dim=hidden,
        history_len=hist,
        neighbor_len=neigh,
        local_epochs=1,
        lr=0.01,
        batch_size=4,
        optimizer_state_scope="client_resident",
    )
    client._device = torch.device("cpu")
    local, opt_state = client.initialize_local_parameters(
        client_id=0,
        total_num_users=2,
        num_users=1,
        num_items=5,
        client_train_data_view=None,
        users_in_client=[0],
    )
    local["user_embedding.weight"] = torch.full((1, hidden), 0.5)
    # Mirror row for user 0 is deliberately the opposite value.
    server_table = np.stack(
        [np.full(hidden, -0.5, dtype=np.float32), np.zeros(hidden, dtype=np.float32)]
    )

    state = ClientStateBase(client_id=0, local_parameters=local, optimizer_state=opt_state)
    request = ClientTrainingRequest(
        client_id=0,
        current_round=0,
        users_in_client=[0],
        client_state=state,
        global_payload={
            "model_state": client._skeleton.get_server_state(),
            "user_embeddings": server_table,
        },
        training_data=pd.DataFrame(
            {"user_id": [0], "item_id": [1], "rating": [1.0]}
        ),
        extra_payload={
            "expanded_user_data": {
                0: {
                    "history_item_ids": np.zeros(hist, dtype=np.int64),
                    "neighbor_embeddings": np.zeros(
                        (hist, neigh, hidden), dtype=np.float32
                    ),
                    "neighbor_user_ids": np.full((hist, neigh), -1, dtype=np.int64),
                }
            }
        },
    )
    resp = client.local_train(request)

    assert "model" not in resp.updated_client_state.local_parameters
    uploaded = resp.updates_packages["user_embeddings"].param_value
    assert uploaded["user_ids"][0] == 0
    assert not np.allclose(uploaded["embeddings"][0], -0.5)

    keys = set(resp.updated_client_state.optimizer_state or {})
    assert keys <= {"user_embedding.weight"}
    assert "item_embeddings.weight" not in keys
