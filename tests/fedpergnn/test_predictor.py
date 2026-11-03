"""FedPerGNN scores must depend on the user, not only on the item table."""

import pytest
import torch

from fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


def _scores_for_user(model, item_ids, history, user_embedding):
    """Encode one user once, then score every candidate item.

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
    user_repr = model.user_representation(history, user_embedding)
    return model.score(user_repr, item_ids)


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
        model.item_embeddings.weight.zero_()
        model.item_embeddings.weight[1:].copy_(torch.eye(8, hidden_dim))

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


def test_one_user_state_scores_every_candidate():
    """A shared user state matches scoring each candidate on its own.

    Raises
    ------
    AssertionError
        If the batched inner product disagrees with per-item scores, or
        if a per-candidate history batch is accepted.
    """
    torch.manual_seed(0)
    hidden = 4
    model = FedPerGNNModel(
        num_items=6, hidden_dim=hidden, history_len=3, neighbor_len=2
    )
    history = torch.tensor([1, 2, -1])
    user = torch.randn(hidden)
    neighbors = torch.randn(3, 2, hidden)
    neighbors[2] = 0
    items = torch.tensor([1, 3, 4, 5])

    with torch.no_grad():
        shared = model(items, history, user, neighbors)
        separate = torch.cat(
            [
                model(item.reshape(1), history, user, neighbors)
                for item in items
            ]
        )
    assert torch.allclose(shared, separate)

    with pytest.raises(ValueError, match="history_item_ids"):
        model.user_representation(history.unsqueeze(0).expand(4, -1), user)
    with pytest.raises(ValueError, match="neighbor_embeddings"):
        model.user_representation(
            history, user, neighbors.unsqueeze(0).expand(4, -1, -1, -1)
        )


def _loss_grads(model, history, user, neighbors, items, labels):
    """Return user, neighbour, and item-table gradients of a mean MSE.

    Parameters
    ----------
    model : FedPerGNNModel
        Skeleton whose item table receives gradient.
    history : torch.Tensor
        History ids, shape ``(L,)``.
    user : torch.Tensor
        Owner row, a parameter of shape ``(H,)``.
    neighbors : torch.Tensor
        Neighbour rows, a parameter of shape ``(L, N, H)``.
    items : torch.Tensor
        Target ids, shape ``(B,)``.
    labels : torch.Tensor
        Regression targets, shape ``(B,)``.

    Returns
    -------
    tuple of torch.Tensor
        Cloned gradients for ``user``, ``neighbors``, and
        ``model.item_embeddings.weight``.
    """
    model.zero_grad(set_to_none=True)
    user.grad = None
    neighbors.grad = None
    user_repr = model.user_representation(history, user, neighbors)
    preds = model.score(user_repr, items)
    torch.nn.functional.mse_loss(preds, labels).backward()
    return (
        user.grad.detach().clone(),
        neighbors.grad.detach().clone(),
        model.item_embeddings.weight.grad.detach().clone(),
    )


def test_shared_user_tower_matches_per_item_gradients():
    """Backprop through one user state matches a tower rerun per item.

    The loss is a mean over candidates. Gradients on the owner row,
    neighbour rows, and item table must agree with encoding the same
    user once per candidate and averaging those losses.

    Raises
    ------
    AssertionError
        If any of the three gradients disagree.
    """
    torch.manual_seed(1)
    hidden = 4
    model = FedPerGNNModel(
        num_items=6, hidden_dim=hidden, history_len=3, neighbor_len=2
    )
    history = torch.tensor([1, 2, -1])
    user = torch.nn.Parameter(torch.randn(hidden))
    neighbors = torch.nn.Parameter(torch.randn(3, 2, hidden))
    with torch.no_grad():
        neighbors[2].zero_()
    items = torch.tensor([1, 3, 4])
    labels = torch.tensor([1.0, 0.0, 0.5])

    shared = _loss_grads(model, history, user, neighbors, items, labels)

    model.zero_grad(set_to_none=True)
    user.grad = None
    neighbors.grad = None
    per_item = []
    for item, label in zip(items, labels):
        user_repr = model.user_representation(history, user, neighbors)
        pred = model.score(user_repr, item.reshape(1))
        per_item.append(
            torch.nn.functional.mse_loss(pred, label.reshape(1))
        )
    torch.stack(per_item).mean().backward()
    separate = (
        user.grad.detach().clone(),
        neighbors.grad.detach().clone(),
        model.item_embeddings.weight.grad.detach().clone(),
    )
    for got, expected in zip(shared, separate):
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-6)
