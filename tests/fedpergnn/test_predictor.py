"""FedPerGNN scores must depend on the user, not only on the item table."""

import torch

from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


def _scores_for_user(model, item_ids, history, user_embedding):
    """Repeat one user's history across a candidate batch and score it.

    Parameters
    ----------
    model : FedPerGNNModel
        Skeleton used for scoring.
    item_ids : torch.Tensor
        Candidate item ids, shape ``(B,)``.
    history : torch.Tensor
        One history row, shape ``(L,)``.
    user_embedding : torch.Tensor
        Owner row, shape ``(H,)``.

    Returns
    -------
    torch.Tensor
        Scores of shape ``(B,)``.
    """
    batch = item_ids.size(0)
    hist = history.unsqueeze(0).expand(batch, -1)
    return model(item_ids, hist, user_embedding)


def test_rank_order_differs_across_users_with_the_same_candidates():
    """Two users with different embeddings and histories must rank items differently.

    A linear layer on ``cat([u_final, item_emb])`` yields one global item
    order for every user. The SI inner-product predictor does not.

    Raises
    ------
    AssertionError
        If the descending rank orders match.

    Examples
    --------
    >>> test_rank_order_differs_across_users_with_the_same_candidates()
    """
    torch.manual_seed(0)
    hidden_dim = 8
    model = FedPerGNNModel(
        num_items=8, hidden_dim=hidden_dim, history_len=3, neighbor_len=2
    )
    with torch.no_grad():
        model.item_embeddings.weight.copy_(torch.eye(8, hidden_dim))

    candidates = torch.arange(1, 8)
    user_a = torch.zeros(hidden_dim)
    user_a[0] = 1.0
    user_b = torch.zeros(hidden_dim)
    user_b[1] = 1.0
    history_a = torch.tensor([1, 2, 3])
    history_b = torch.tensor([4, 5, 6])

    scores_a = _scores_for_user(model, candidates, history_a, user_a)
    scores_b = _scores_for_user(model, candidates, history_b, user_b)
    order_a = torch.argsort(scores_a, descending=True)
    order_b = torch.argsort(scores_b, descending=True)
    assert not torch.equal(order_a, order_b)
    # Additive concat-linear scores differ by a user constant, so pairwise
    # gaps would match even when the constant changes.
    assert not torch.allclose(scores_a - scores_a[0], scores_b - scores_b[0])
