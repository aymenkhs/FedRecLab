"""FedFast-private ActvAGG helpers (Muhammad et al., KDD 2020 Algorithm 3).

This module is semi-private: names are underscored so it is not confused
with the stock aggregators in ``federated.aggregation``. Power users may
still import it. Faithful FedFast wiring is normally enabled through the
``FedFast(actv_agg=True)`` constructor flag without importing this file.

Item-embedding ActvAGG uses the public ``ChangeMagnitudeAvg`` aggregator.
Non-embedding components use ``FedAvg``. Subordinate user-embedding
propagation and reclustering live here and are materialised by the
FedFast server on ``_on_round_end_server``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.fedreclab.federated.aggregation import ChangeMagnitudeAvg, FedAvg


@dataclass
class _FedFastClusterState:
    """Shared cluster / pending-write state for FedFast ActvAGG.

    Parameters
    ----------
    num_clusters : int
        Target number of clusters ``p``.
    seed : int or None
        Seed for k-means RNG. ``None`` leaves the generator unseeded.

    Attributes
    ----------
    groups : list of list of int or None
        Current partition ``G``.
    previous_user_embeddings : dict of int to ndarray or None
        Snapshot of each client's mean user embedding before local
        training in the current round (``w0``).
    pending_user_writes : dict of int to ndarray
        Subordinate embeddings to apply at round end.
    last_gamma : float or None
        Discount used for the most recent subordinate update.
    """

    num_clusters: int
    seed: int | None = None
    groups: list[list[int]] | None = None
    previous_user_embeddings: dict[int, np.ndarray] | None = None
    pending_user_writes: dict[int, np.ndarray] = field(default_factory=dict)
    last_gamma: float | None = None


class _ActvAgg:
    """Compute ActvAGG subordinate user updates and refresh clusters.

    Delegates' user embeddings are already stored in ``ClientState`` after
    local training. This helper:

    1. builds a full client embedding matrix from current states;
    2. reclusters with seeded k-means into ``p`` groups;
    3. for each subordinate, adds a discounted average of co-cluster
       delegate embedding deltas ``wk - w0``.

    Parameters
    ----------
    cluster_state : _FedFastClusterState
        Mutable state shared with the FedFast server and selector.
    change_eps : float, optional
        Forwarded when building the default item aggregator. Default 0.0.

    Examples
    --------
    >>> state = _FedFastClusterState(num_clusters=2, seed=0)
    >>> agg = _ActvAgg(state)
    >>> isinstance(agg, _ActvAgg)
    True
    """

    def __init__(
        self,
        cluster_state: _FedFastClusterState,
        change_eps: float = 0.0,
    ):
        self.cluster_state = cluster_state
        self.change_eps = float(change_eps)

    @staticmethod
    def default_aggregation_strategies(
        num_clients: int,
        change_eps: float = 0.0,
    ) -> dict[str, Any]:
        """Return the stock FedFast aggregators for item and output groups.

        Parameters
        ----------
        num_clients : int
            Population size ``K``.
        change_eps : float, optional
            Threshold for ``ChangeMagnitudeAvg``. Default is 0.0.

        Returns
        -------
        dict of str to BaseAggregation
            ``item_embeddings`` → ``ChangeMagnitudeAvg``,
            ``output_weights`` → ``FedAvg``.

        Examples
        --------
        >>> strats = _ActvAgg.default_aggregation_strategies(10)
        >>> sorted(strats)
        ['item_embeddings', 'output_weights']
        """
        return {
            "item_embeddings": ChangeMagnitudeAvg(
                num_clients, change_eps=change_eps
            ),
            "output_weights": FedAvg(num_clients),
        }

    def snapshot_user_embeddings(
        self,
        client_embeddings: Mapping[int, np.ndarray],
    ) -> None:
        """Store pre-training user embeddings as ``w0`` for this round.

        Parameters
        ----------
        client_embeddings : mapping of int to ndarray
            One embedding vector per client (mean over local users if a
            client hosts several).

        Returns
        -------
        None
        """
        self.cluster_state.previous_user_embeddings = {
            int(cid): np.asarray(vec, dtype=np.float64).reshape(-1).copy()
            for cid, vec in client_embeddings.items()
        }
        self.cluster_state.pending_user_writes = {}

    def compute_subordinate_updates(
        self,
        client_embeddings: Mapping[int, np.ndarray],
        selected_client_ids: Sequence[int],
        current_round: int,
    ) -> tuple[dict[int, np.ndarray], list[list[int]]]:
        """Recluster and compute discounted subordinate embeddings.

        Parameters
        ----------
        client_embeddings : mapping of int to ndarray
            Current per-client embeddings (delegates already updated).
        selected_client_ids : sequence of int
            Delegates ``S`` for this round.
        current_round : int
            Round index ``t`` used for ``gamma = exp(-t)``.

        Returns
        -------
        pending : dict of int to ndarray
            New embeddings for subordinate clients only.
        groups : list of list of int
            Updated partition ``G``.

        Raises
        ------
        ValueError
            If a pre-training snapshot is missing, embeddings are
            non-finite, or ``num_clusters`` is invalid for the population.

        Examples
        --------
        >>> state = _FedFastClusterState(num_clusters=2, seed=0)
        >>> helper = _ActvAgg(state)
        >>> emb = {i: np.ones(2) * i for i in range(4)}
        >>> helper.snapshot_user_embeddings(emb)
        >>> emb[0] = np.array([10.0, 10.0])
        >>> pending, groups = helper.compute_subordinate_updates(emb, [0], 0)
        >>> len(groups) == 2
        True
        """
        previous = self.cluster_state.previous_user_embeddings
        if previous is None:
            raise ValueError(
                "ActvAGG requires snapshot_user_embeddings before "
                "compute_subordinate_updates"
            )
        selected = {int(c) for c in selected_client_ids}
        client_ids = sorted(int(c) for c in client_embeddings)
        if set(client_ids) != set(previous):
            raise ValueError(
                "ActvAGG client_embeddings keys must match the pre-training "
                "snapshot"
            )

        matrix = np.stack(
            [
                np.asarray(client_embeddings[cid], dtype=np.float64).reshape(-1)
                for cid in client_ids
            ],
            axis=0,
        )
        if not np.isfinite(matrix).all():
            raise ValueError("ActvAGG user embeddings must be finite")

        p = int(self.cluster_state.num_clusters)
        labels = _kmeans_labels(
            matrix, num_clusters=p, seed=self.cluster_state.seed
        )
        groups: list[list[int]] = [[] for _ in range(p)]
        cid_to_cluster = {}
        for row, cid in enumerate(client_ids):
            c = int(labels[row])
            groups[c].append(cid)
            cid_to_cluster[cid] = c
        # Drop empty clusters while keeping p lists for set_groups cover:
        # reassign empty by merging into nearest non-empty via label remap.
        groups = _ensure_nonempty_groups(groups, client_ids, matrix, labels)

        gamma = float(np.exp(-float(current_round)))
        self.cluster_state.last_gamma = gamma

        # Accumulate delegate deltas onto subordinates in the same cluster.
        dim = matrix.shape[1]
        delta = {cid: np.zeros(dim, dtype=np.float64) for cid in client_ids}
        delegates_per_cluster: dict[int, list[int]] = {
            i: [] for i in range(len(groups))
        }
        for i, group in enumerate(groups):
            for cid in group:
                cid_to_cluster[cid] = i
                if cid in selected:
                    delegates_per_cluster[i].append(cid)

        for k in selected:
            ck = cid_to_cluster[k]
            d_k = np.asarray(client_embeddings[k], dtype=np.float64).reshape(
                -1
            ) - np.asarray(previous[k], dtype=np.float64).reshape(-1)
            if not np.isfinite(d_k).all():
                raise ValueError(
                    f"ActvAGG non-finite delegate delta for client {k}"
                )
            for s in groups[ck]:
                if s in selected:
                    continue
                delta[s] = delta[s] + d_k

        pending: dict[int, np.ndarray] = {}
        for c_idx, group in enumerate(groups):
            n_del = len(delegates_per_cluster[c_idx])
            if n_del == 0:
                continue
            scale = gamma / float(n_del)
            for s in group:
                if s in selected:
                    continue
                new_emb = np.asarray(previous[s], dtype=np.float64).reshape(
                    -1
                ) + scale * delta[s]
                if not np.isfinite(new_emb).all():
                    raise ValueError(
                        f"ActvAGG non-finite subordinate embedding for "
                        f"client {s}"
                    )
                pending[s] = new_emb.astype(np.float64, copy=False)

        self.cluster_state.groups = groups
        self.cluster_state.pending_user_writes = pending
        return pending, groups


def _kmeans_labels(
    x: np.ndarray,
    num_clusters: int,
    seed: int | None,
    max_iter: int = 50,
) -> np.ndarray:
    """Run a small seeded Lloyd k-means and return integer labels.

    Parameters
    ----------
    x : ndarray
        Data matrix, shape ``(n, d)``.
    num_clusters : int
        Number of clusters ``p``.
    seed : int or None
        RNG seed. ``None`` uses an unseeded generator.
    max_iter : int, optional
        Maximum Lloyd iterations. Default is 50.

    Returns
    -------
    ndarray
        Label in ``[0, p)`` for each row.

    Raises
    ------
    ValueError
        If ``num_clusters`` is outside ``[1, n]``.
    """
    n = int(x.shape[0])
    p = int(num_clusters)
    if p < 1 or p > n:
        raise ValueError(
            f"num_clusters must satisfy 1 <= p <= n, got p={p}, n={n}"
        )
    if p == 1:
        return np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    # k-means++ style init: first centre uniform, then distance-weighted.
    centres = np.empty((p, x.shape[1]), dtype=np.float64)
    centres[0] = x[int(rng.integers(0, n))]
    closest = np.full(n, np.inf)
    for j in range(1, p):
        dist = np.linalg.norm(x - centres[j - 1], axis=1)
        closest = np.minimum(closest, dist)
        weights = closest ** 2
        total = float(weights.sum())
        if total <= 0:
            centres[j] = x[int(rng.integers(0, n))]
        else:
            probs = weights / total
            centres[j] = x[int(rng.choice(n, p=probs))]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(p):
            members = x[labels == j]
            if members.shape[0] == 0:
                centres[j] = x[int(rng.integers(0, n))]
            else:
                centres[j] = members.mean(axis=0)
    return labels


def _ensure_nonempty_groups(
    groups: list[list[int]],
    client_ids: Sequence[int],
    matrix: np.ndarray,
    labels: np.ndarray,
) -> list[list[int]]:
    """Reassign clients from empty clusters so every group is non-empty.

    Empty k-means clusters are rare with ``p << n`` but break
    ``ActvSampClientSelection.set_groups``. Clients in empty labels are
    moved to the nearest non-empty centre's group; if that still leaves
    empties, remaining empty slots receive one client taken from the
    largest group.

    Parameters
    ----------
    groups : list of list of int
        Tentative clusters (may contain empties).
    client_ids : sequence of int
        Row alignment with ``matrix``.
    matrix : ndarray
        Embedding matrix, shape ``(n, d)``.
    labels : ndarray
        Label per row.

    Returns
    -------
    list of list of int
        ``p`` non-empty groups covering all clients.
    """
    p = len(groups)
    nonempty = [i for i, g in enumerate(groups) if len(g) > 0]
    if len(nonempty) == p:
        return groups

    centres = []
    for i in range(p):
        if groups[i]:
            rows = [client_ids.index(cid) for cid in groups[i]]
            centres.append(matrix[rows].mean(axis=0))
        else:
            centres.append(None)

    new_groups: list[list[int]] = [[] for _ in range(p)]
    for row, cid in enumerate(client_ids):
        lab = int(labels[row])
        if centres[lab] is not None:
            new_groups[lab].append(cid)
            continue
        # Assign to nearest non-empty centre.
        best = nonempty[0]
        best_dist = np.inf
        for j in nonempty:
            dist = float(np.linalg.norm(matrix[row] - centres[j]))
            if dist < best_dist:
                best_dist = dist
                best = j
        new_groups[best].append(cid)

    # Ensure every index has at least one client by stealing from the largest.
    for i in range(p):
        if new_groups[i]:
            continue
        donor = max(range(p), key=lambda j: len(new_groups[j]))
        if len(new_groups[donor]) <= 1:
            raise ValueError(
                "Cannot repair empty ActvAGG clusters: not enough clients "
                f"for p={p}"
            )
        new_groups[i].append(new_groups[donor].pop())
    return new_groups
