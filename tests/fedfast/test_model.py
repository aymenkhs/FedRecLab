"""FedFast GMF skeleton groups and forward shape."""

import torch

from fedreclab.methods.fedfast.fedfast_model import GeneralMatrixFactorization


def test_server_groups_and_forward_shape():
    m = GeneralMatrixFactorization(num_items=5, embedding_dim=4)
    groups = m.get_server_state_groups()
    assert set(groups) == {"item_embeddings", "output_weights"}
    assert groups["item_embeddings"]["item_embeddings.weight"].shape == (5, 4)

    user_table = torch.randn(2, 4)
    x = torch.tensor([[0, 1], [1, 3]], dtype=torch.long)
    scores = m(x, user_table)
    assert scores.shape == (2,)
