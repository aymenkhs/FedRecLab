"""LightGCN helpers for SemiDFEGL.

Source: Qu et al., WWW 2023, equations (2), (6) and (8), with the
simulation shortcut that neighbour ego vectors stay fixed across layers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


def xavier_uniform_table(
    num_rows: int,
    dim: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a ``(num_rows, dim)`` table filled with Xavier uniform values.

    Parameters
    ----------
    num_rows : int
        Number of embedding rows.
    dim : int
        Embedding width.
    generator : torch.Generator or None, optional
        CPU generator for the draw. ``None`` uses the default RNG.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(num_rows, dim)``.

    Raises
    ------
    ValueError
        If ``num_rows`` or ``dim`` is not a positive integer.

    Examples
    --------
    >>> g = torch.Generator().manual_seed(0)
    >>> t = xavier_uniform_table(3, 4, generator=g)
    >>> t.shape
    torch.Size([3, 4])
    """
    if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows < 1:
        raise ValueError(f"num_rows must be a positive int, got {num_rows!r}")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    table = torch.empty(num_rows, dim, dtype=torch.float32)
    if generator is None:
        nn.init.xavier_uniform_(table)
    else:
        # Match nn.init.xavier_uniform_ with a and gain 1.
        fan_in = float(dim)
        fan_out = float(num_rows)
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        bound = (3.0 ** 0.5) * std
        table.uniform_(-bound, bound, generator=generator)
    return table


def lightgcn_layer_combine(
    layers: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Average layer embeddings with equal LightGCN weights ``1/(K+1)``.

    Parameters
    ----------
    layers : sequence of torch.Tensor
        Embeddings at layers ``0 .. K``, same shape.

    Returns
    -------
    torch.Tensor
        Mean of the inputs.

    Raises
    ------
    ValueError
        If ``layers`` is empty.

    Examples
    --------
    >>> a = torch.ones(2)
    >>> b = torch.zeros(2)
    >>> torch.allclose(lightgcn_layer_combine([a, b]), torch.tensor([0.5, 0.5]))
    True
    """
    if len(layers) == 0:
        raise ValueError("layers must contain at least the layer-0 embedding")
    stacked = torch.stack(list(layers), dim=0)
    return stacked.mean(dim=0)


def propagate_semidfegl(
    user_emb0: torch.Tensor,
    item_emb0: torch.Tensor,
    item_is_fake: torch.Tensor,
    neighbor_egos: torch.Tensor,
    user_degree: float,
    neighbor_degrees: torch.Tensor,
    cluster_size: int,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run lagged LightGCN on one user's augmented ego graph.

    Layer 0 is ``(user_emb0, item_emb0)``. At every layer the user
    aggregates from its real and fake item states. Real-only items
    aggregate from the user with degree 1. Fake items aggregate from the
    live user state and the detached neighbour ego vectors, with item
    degree ``cluster_size``. Neighbour ego vectors do not advance through
    layers.

    Parameters
    ----------
    user_emb0 : torch.Tensor
        Shape ``(d,)``. Trainable layer-0 user embedding.
    item_emb0 : torch.Tensor
        Shape ``(n_items, d)``. Layer-0 embeddings of the items on this
        device (training positives and fake bridges).
    item_is_fake : torch.Tensor
        Boolean mask of shape ``(n_items,)``. True for fake common items.
    neighbor_egos : torch.Tensor
        Shape ``(n_neighbors, d)``. Detached ego vectors of the other
        users in the cluster. Empty when the cluster has size 1.
    user_degree : float
        Augmented degree of this user ``|N(u)|``.
    neighbor_degrees : torch.Tensor
        Shape ``(n_neighbors,)``. Augmented degrees of the neighbours.
    cluster_size : int
        Number of users in the cluster. Degree of every fake item.
    num_layers : int
        Number of propagation steps ``K``.

    Returns
    -------
    user_combined : torch.Tensor
        Shape ``(d,)``. Layer-combined user embedding.
    item_combined : torch.Tensor
        Shape ``(n_items, d)``. Layer-combined item embeddings.

    Raises
    ------
    ValueError
        If shapes disagree, degrees are non-positive, or ``num_layers``
        is negative.

    Examples
    --------
    >>> u = torch.ones(2)
    >>> items = torch.ones(1, 2)
    >>> fake = torch.tensor([False])
    >>> uc, ic = propagate_semidfegl(
    ...     u, items, fake, torch.zeros(0, 2), 1.0, torch.zeros(0), 1, 1
    ... )
    >>> uc.shape, ic.shape
    (torch.Size([2]), torch.Size([1, 2]))
    """
    if user_emb0.ndim != 1:
        raise ValueError(f"user_emb0 must be 1-d, got shape {tuple(user_emb0.shape)}")
    if item_emb0.ndim != 2:
        raise ValueError(f"item_emb0 must be 2-d, got shape {tuple(item_emb0.shape)}")
    if item_is_fake.shape != (item_emb0.shape[0],):
        raise ValueError(
            f"item_is_fake shape {tuple(item_is_fake.shape)} does not match "
            f"n_items={item_emb0.shape[0]}"
        )
    if neighbor_egos.ndim != 2 or neighbor_egos.shape[1] != user_emb0.shape[0]:
        raise ValueError(
            f"neighbor_egos must have shape (n, {user_emb0.shape[0]}), "
            f"got {tuple(neighbor_egos.shape)}"
        )
    if neighbor_degrees.shape != (neighbor_egos.shape[0],):
        raise ValueError(
            f"neighbor_degrees shape {tuple(neighbor_degrees.shape)} does not "
            f"match n_neighbors={neighbor_egos.shape[0]}"
        )
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 0:
        raise ValueError(f"num_layers must be a non-negative int, got {num_layers!r}")
    if user_degree <= 0.0:
        raise ValueError(f"user_degree must be > 0, got {user_degree}")
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}")
    if (neighbor_degrees <= 0).any():
        raise ValueError("neighbor_degrees must be positive")

    d = int(user_emb0.shape[0])
    n_items = int(item_emb0.shape[0])
    device = user_emb0.device
    dtype = user_emb0.dtype

    item_deg = torch.ones(n_items, dtype=dtype, device=device)
    item_deg = torch.where(
        item_is_fake,
        torch.full((), float(cluster_size), dtype=dtype, device=device),
        item_deg,
    )

    sqrt_user = float(user_degree) ** 0.5
    sqrt_items = item_deg.sqrt()
    # Coefficient from user to each item and back.
    user_to_item = 1.0 / (sqrt_user * sqrt_items)

    neighbor_egos = neighbor_egos.detach()
    if neighbor_egos.numel() > 0:
        neigh_coeff = 1.0 / (
            (float(cluster_size) ** 0.5) * neighbor_degrees.to(dtype=dtype).sqrt()
        )
    else:
        neigh_coeff = neighbor_egos.new_zeros((0,))

    user_layers = [user_emb0]
    item_layers = [item_emb0]
    user_k = user_emb0
    item_k = item_emb0

    for _ in range(num_layers):
        # User aggregates from all on-device items.
        user_next = (user_to_item.unsqueeze(1) * item_k).sum(dim=0)

        # Real-only items: only this user. Fake items: user + neighbour egos.
        item_next = user_to_item.unsqueeze(1) * user_k.unsqueeze(0)
        if item_is_fake.any() and neighbor_egos.shape[0] > 0:
            fake_from_neighbors = (neigh_coeff.unsqueeze(1) * neighbor_egos).sum(dim=0)
            item_next = item_next.clone()
            item_next[item_is_fake] = (
                item_next[item_is_fake]
                + fake_from_neighbors.unsqueeze(0).expand(
                    int(item_is_fake.sum().item()), d
                )
            )
        elif item_is_fake.any() and neighbor_egos.shape[0] == 0:
            # Cluster of one: fake item degree is 1, already in user_to_item.
            pass

        user_k = user_next
        item_k = item_next
        user_layers.append(user_k)
        item_layers.append(item_k)

    return lightgcn_layer_combine(user_layers), lightgcn_layer_combine(item_layers)


def score_dot(
    user_vec: torch.Tensor,
    item_rows: torch.Tensor,
) -> torch.Tensor:
    """Dot-product scores ``user_vec @ item_rows.T``.

    Parameters
    ----------
    user_vec : torch.Tensor
        Shape ``(d,)`` or ``(1, d)``.
    item_rows : torch.Tensor
        Shape ``(n, d)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``.

    Examples
    --------
    >>> score_dot(torch.ones(2), torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    tensor([1., 2.])
    """
    u = user_vec.reshape(-1)
    return item_rows @ u


def fuzzy_cmeans(
    points: np.ndarray,
    num_clusters: int,
    fuzziness: float,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy c-means with Euclidean distance.

    Parameters
    ----------
    points : ndarray
        Shape ``(n, d)`` float array of points to cluster.
    num_clusters : int
        Number of centres ``C``.
    fuzziness : float
        Weighting exponent ``l > 1``.
    max_iter : int
        Maximum number of alternating updates.
    tol : float
        Stop when the max absolute change in membership is below this.
    rng : numpy.random.Generator
        Source of the initial centres.

    Returns
    -------
    membership : ndarray
        Shape ``(n, C)``. Soft memberships.
    centres : ndarray
        Shape ``(C, d)``. Cluster centres.

    Raises
    ------
    ValueError
        If shapes or hyper-parameters are invalid, or a centre becomes
        empty / non-finite.

    Examples
    --------
    >>> pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [1.1, 1.0]])
    >>> P, C = fuzzy_cmeans(pts, 2, 2.0, 50, 1e-5, np.random.default_rng(0))
    >>> P.shape, C.shape
    ((4, 2), (2, 2))
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"points must have shape (n, d) with n >= 1, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("points contains non-finite values")
    n, d = x.shape
    c = int(num_clusters)
    if c < 1:
        raise ValueError(f"num_clusters must be >= 1, got {num_clusters!r}")
    if c > n:
        raise ValueError(
            f"num_clusters={c} exceeds number of points {n}"
        )
    if float(fuzziness) <= 1.0:
        raise ValueError(f"fuzziness must be > 1, got {fuzziness}")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter!r}")
    if float(tol) < 0.0:
        raise ValueError(f"tol must be >= 0, got {tol}")

    # Seed centres by sampling distinct rows.
    idx = rng.choice(n, size=c, replace=False)
    centres = x[idx].copy()
    membership = np.zeros((n, c), dtype=np.float64)
    exponent = 2.0 / (float(fuzziness) - 1.0)

    for _ in range(max_iter):
        # Squared Euclidean distances; protect exact centre hits.
        diff = x[:, None, :] - centres[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        dist2 = np.maximum(dist2, 1e-12)
        inv = dist2 ** (-exponent / 2.0)
        membership_new = inv / inv.sum(axis=1, keepdims=True)
        if not np.isfinite(membership_new).all():
            raise ValueError("fuzzy c-means membership became non-finite")

        weights = membership_new ** float(fuzziness)
        weight_sum = weights.sum(axis=0)
        if (weight_sum <= 0).any():
            raise ValueError("fuzzy c-means produced an empty centre")
        centres = (weights.T @ x) / weight_sum[:, None]
        if not np.isfinite(centres).all():
            raise ValueError("fuzzy c-means centres became non-finite")

        delta = float(np.max(np.abs(membership_new - membership)))
        membership = membership_new
        if delta < float(tol):
            break

    return membership.astype(np.float64), centres.astype(np.float64)
