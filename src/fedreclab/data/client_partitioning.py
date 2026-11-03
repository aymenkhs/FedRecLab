"""Client partition strategies for federated recommendation simulation.

Partition strategies assign every training user to exactly one client and
return a :class:`ClientPartition`. Only the training split is partitioned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class ClientPartition:
    num_clients: int
    client_ids_2_user_ids: Dict[int, List[int]]
    user_ids_2_client_ids: Dict[int, int]
    client_to_train_indices: Dict[int, List[int]]
    client_train_sizes: Dict[int, int]
    client_to_val_indices: Dict[int, List[int]] | None = None
    client_to_test_indices: Dict[int, List[int]] | None = None
    metadata: Dict[str, Any] | None = None


class BaseClientPartitionStrategy:
    def __init__(
        self,
        num_clients: int,
        user_col: str = "user_id",
        metadata: Dict[str, Any] | None = None,
    ):
        self.num_clients = num_clients
        self.user_col = user_col
        self.metadata = metadata

    def partition(self, train_interactions: pd.DataFrame) -> ClientPartition:
        raise NotImplementedError("Subclasses must implement this method")


def _build_client_partition(
    train_interactions: pd.DataFrame,
    user_col: str,
    user_to_client: Mapping[int, int],
    num_clients: int,
    metadata: Dict[str, Any] | None = None,
) -> ClientPartition:
    """Build a :class:`ClientPartition` from a user-to-client assignment.

    Parameters
    ----------
    train_interactions : pandas.DataFrame
        Training interactions. Index labels are preserved in
        ``client_to_train_indices``.
    user_col : str
        Column that holds user identifiers.
    user_to_client : mapping of int to int
        Assignment of each training user id to a client id in
        ``0 … num_clients - 1``. Every user that appears in
        ``train_interactions`` must be present.
    num_clients : int
        Number of clients. Keys ``0 … num_clients - 1`` are always
        present in the returned maps, including empty clients.
    metadata : dict or None, optional
        Copied onto the returned partition. Default is None.

    Returns
    -------
    ClientPartition
        Partition with sorted user lists per client, the inverse user map,
        training index lists concatenated in user-id order, and their
        lengths as ``client_train_sizes``.

    Raises
    ------
    ValueError
        If ``user_col`` is missing from ``train_interactions``, or a
        training user has no entry in ``user_to_client``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [10, 11, 10]})
    >>> part = _build_client_partition(df, "user_id", {0: 0, 1: 0}, 1)
    >>> part.client_train_sizes[0]
    3
    """
    if user_col not in train_interactions.columns:
        raise ValueError(
            f"train_interactions must contain a '{user_col}' column with "
            "user identifiers"
        )

    client_ids_2_user_ids: Dict[int, List[int]] = {
        client_id: [] for client_id in range(num_clients)
    }
    for user_id, client_id in user_to_client.items():
        client_ids_2_user_ids[int(client_id)].append(int(user_id))
    for client_id in client_ids_2_user_ids:
        client_ids_2_user_ids[client_id].sort()

    user_ids_2_client_ids: Dict[int, int] = {
        int(user_id): int(client_id)
        for user_id, client_id in user_to_client.items()
    }

    grouped = train_interactions.groupby(user_col, sort=False).groups
    client_to_train_indices: Dict[int, List[int]] = {
        client_id: [] for client_id in range(num_clients)
    }
    for client_id, user_ids in client_ids_2_user_ids.items():
        index_list: List[int] = []
        for user_id in user_ids:
            if user_id not in grouped:
                continue
            index_list.extend(int(i) for i in grouped[user_id])
        client_to_train_indices[client_id] = index_list

    for user_id in train_interactions[user_col].unique():
        uid = int(user_id)
        if uid not in user_ids_2_client_ids:
            raise ValueError(
                f"training user {uid} has no client assignment in "
                "user_to_client"
            )

    client_train_sizes = {
        client_id: len(indices)
        for client_id, indices in client_to_train_indices.items()
    }

    return ClientPartition(
        num_clients=num_clients,
        client_ids_2_user_ids=client_ids_2_user_ids,
        user_ids_2_client_ids=user_ids_2_client_ids,
        client_to_train_indices=client_to_train_indices,
        client_train_sizes=client_train_sizes,
        client_to_val_indices=None,
        client_to_test_indices=None,
        metadata=metadata,
    )


def _log_partition_sizes(
    client_ids_2_user_ids: Mapping[int, List[int]],
    client_train_sizes: Mapping[int, int],
    *,
    target_interaction_proportion: float | Sequence[float] | None = None,
) -> None:
    """Log min, max and mean client sizes in users and interactions.

    Parameters
    ----------
    client_ids_2_user_ids : mapping of int to list of int
        Users assigned to each client.
    client_train_sizes : mapping of int to int
        Training interaction counts per client.
    target_interaction_proportion : float, sequence of float, or None, optional
        When set, also log the largest absolute gap between each
        client's achieved interaction proportion and this target.
        A scalar is compared to every client; a sequence is aligned to
        clients in ``client_ids_2_user_ids`` iteration order. Default
        is None.
    """
    user_counts = np.asarray(
        [len(users) for users in client_ids_2_user_ids.values()], dtype=float
    )
    interaction_counts = np.asarray(
        [client_train_sizes[cid] for cid in client_ids_2_user_ids],
        dtype=float,
    )
    logging.info(
        "client partition sizes: users min=%s max=%s mean=%.2f; "
        "interactions min=%s max=%s mean=%.2f",
        int(user_counts.min()) if len(user_counts) else 0,
        int(user_counts.max()) if len(user_counts) else 0,
        float(user_counts.mean()) if len(user_counts) else 0.0,
        int(interaction_counts.min()) if len(interaction_counts) else 0,
        int(interaction_counts.max()) if len(interaction_counts) else 0,
        float(interaction_counts.mean()) if len(interaction_counts) else 0.0,
    )
    if target_interaction_proportion is not None:
        total = float(interaction_counts.sum())
        if total > 0:
            proportions = interaction_counts / total
            targets = np.asarray(target_interaction_proportion, dtype=float)
            max_gap = float(np.max(np.abs(proportions - targets)))
        else:
            max_gap = 0.0
        if np.ndim(target_interaction_proportion) == 0:
            logging.info(
                "client partition interaction proportion max gap vs target "
                "%.6f: %.6f",
                float(target_interaction_proportion),
                max_gap,
            )
        else:
            logging.info(
                "client partition interaction proportion max gap vs "
                "Dirichlet targets: %.6f",
                max_gap,
            )


class CrossDevicePartitionStrategy(BaseClientPartitionStrategy):
    """
    Each unique user is mapped to a distinct client.

    The number of clients is determined from the data as the number of
    unique user ids in the `u` column of `train_interactions`.
    """

    def __init__(
        self, user_col: str = "user_id", metadata: Dict[str, Any] | None = None
    ) -> None:
        # num_clients is inferred during partition
        super().__init__(num_clients=0, user_col=user_col, metadata=metadata)

    def partition(self, train_interactions: pd.DataFrame) -> ClientPartition:
        if self.user_col not in train_interactions.columns:
            raise ValueError(
                f"train_interactions must contain a '{self.user_col}' "
                "column with user identifiers"
            )

        # Unique users define clients (one client per user)
        unique_users = train_interactions[self.user_col].unique()
        num_clients = len(unique_users)

        # Map user ids to client ids (0-indexed)
        user_ids_2_client_ids: Dict[int, int] = {
            int(user_id): client_id
            for client_id, user_id in enumerate(unique_users)
        }

        # Inverse mapping: each client id has a single associated user id
        client_ids_2_user_ids: Dict[int, List[int]] = {
            client_id: [int(user_id)]
            for user_id, client_id in user_ids_2_client_ids.items()
        }

        # Group interactions by user to build index lists per client
        grouped = train_interactions.groupby(self.user_col).groups
        client_to_train_indices: Dict[int, List[int]] = {}
        client_train_sizes: Dict[int, int] = {}

        for user_id, indices in grouped.items():
            client_id = user_ids_2_client_ids[int(user_id)]
            index_list = list(indices)
            client_to_train_indices[client_id] = index_list
            client_train_sizes[client_id] = len(index_list)

        # Update num_clients on the strategy instance
        self.num_clients = num_clients

        return ClientPartition(
            num_clients=num_clients,
            client_ids_2_user_ids=client_ids_2_user_ids,
            user_ids_2_client_ids=user_ids_2_client_ids,
            client_to_train_indices=client_to_train_indices,
            client_train_sizes=client_train_sizes,
            client_to_val_indices=None,
            client_to_test_indices=None,
            metadata=self.metadata,
        )


class RandomUserPartitionStrategy(BaseClientPartitionStrategy):
    """Assign whole users to a fixed number of clients by random shuffle.

    Users are sorted, shuffled with a private NumPy generator, then split
    into ``num_clients`` groups. With ``size_by="users"`` the groups have
    near-equal user counts. With ``size_by="interactions"`` each remaining
    user is given to the client with the largest interaction-count deficit
    relative to a uniform target, so client interaction totals are nearly
    equal.

    Cross-silo settings use ``1 ≤ num_clients < num_users``. Setting
    ``num_clients`` equal to the number of training users yields one user
    per client. Setting it to ``1`` places every user on a single client.

    Parameters
    ----------
    num_clients : int
        Number of clients. Must be an integer ``>= 1``. The value must
        also satisfy ``num_clients <=`` the number of distinct training
        users when :meth:`partition` runs.
    size_by : {"users", "interactions"}, optional
        Quantity to balance across clients. ``"users"`` (default) splits
        users into contiguous chunks whose sizes differ by at most one.
        ``"interactions"`` uses greedy fill against uniform interaction
        targets.
    seed : int or None, optional
        Seed for the strategy's ``numpy.random.Generator``. ``None``
        leaves the generator unseeded. Default is None.
    user_col : str, optional
        User-id column in the training frame. Default is ``"user_id"``.
    metadata : dict or None, optional
        Stored on the strategy and copied onto the returned partition.
        Default is None.

    Raises
    ------
    ValueError
        If ``num_clients`` is not an integer ``>= 1``, or ``size_by`` is
        not ``"users"`` or ``"interactions"``. :meth:`partition` also
        raises if ``user_col`` is missing or ``num_clients`` exceeds the
        number of distinct training users.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 0, 1, 2, 2, 3],
    ...     "item_id": [10, 11, 10, 12, 13, 10],
    ... })
    >>> strategy = RandomUserPartitionStrategy(2, seed=0)
    >>> part = strategy.partition(df)
    >>> part.num_clients
    2
    >>> sorted(part.user_ids_2_client_ids)
    [0, 1, 2, 3]
    """

    def __init__(
        self,
        num_clients: int,
        size_by: str = "users",
        seed: int | None = None,
        user_col: str = "user_id",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(num_clients, int)
            or isinstance(num_clients, bool)
            or num_clients < 1
        ):
            raise ValueError(
                f"num_clients must be an integer >= 1, got {num_clients!r}"
            )
        if size_by not in {"users", "interactions"}:
            raise ValueError(
                "size_by must be 'users' or 'interactions', "
                f"got {size_by!r}"
            )
        super().__init__(
            num_clients=num_clients, user_col=user_col, metadata=metadata
        )
        self.size_by = size_by
        self.seed = seed

    def partition(self, train_interactions: pd.DataFrame) -> ClientPartition:
        """Assign every training user to one client and build the partition.

        Parameters
        ----------
        train_interactions : pandas.DataFrame
            Training interactions. Must contain ``self.user_col``.

        Returns
        -------
        ClientPartition
            Partition over ``self.num_clients`` clients.

        Raises
        ------
        ValueError
            If ``user_col`` is missing, or there are fewer distinct
            training users than ``num_clients``.
        """
        if self.user_col not in train_interactions.columns:
            raise ValueError(
                f"train_interactions must contain a '{self.user_col}' "
                "column with user identifiers"
            )

        user_ids = np.sort(
            train_interactions[self.user_col].unique().astype(np.int64)
        )
        num_users = int(user_ids.size)
        if self.num_clients > num_users:
            raise ValueError(
                f"num_clients ({self.num_clients}) must be <= number of "
                f"distinct training users ({num_users})"
            )

        rng = np.random.default_rng(self.seed)
        shuffled = user_ids.copy()
        rng.shuffle(shuffled)

        user_interaction_counts = (
            train_interactions.groupby(self.user_col).size().astype(np.int64)
        )
        # Align counts to integer user ids after the unique/sort path.
        counts_by_user = {
            int(uid): int(user_interaction_counts.loc[uid])
            for uid in user_ids
        }

        if self.size_by == "users":
            user_to_client = self._assign_by_users(shuffled)
            target_proportion = None
        else:
            user_to_client = self._assign_by_interactions(
                shuffled, counts_by_user, n_train=len(train_interactions)
            )
            target_proportion = 1.0 / float(self.num_clients)

        partition = _build_client_partition(
            train_interactions,
            self.user_col,
            user_to_client,
            self.num_clients,
            metadata=self.metadata,
        )
        _log_partition_sizes(
            partition.client_ids_2_user_ids,
            partition.client_train_sizes,
            target_interaction_proportion=target_proportion,
        )
        return partition

    def _assign_by_users(self, shuffled_users: np.ndarray) -> Dict[int, int]:
        """Split shuffled users into contiguous near-equal chunks.

        Parameters
        ----------
        shuffled_users : numpy.ndarray
            Shuffled user ids of length ``num_users``.

        Returns
        -------
        dict of int to int
            Mapping from user id to client id.
        """
        num_users = int(shuffled_users.size)
        base = num_users // self.num_clients
        remainder = num_users % self.num_clients
        user_to_client: Dict[int, int] = {}
        offset = 0
        for client_id in range(self.num_clients):
            size = base + (1 if client_id < remainder else 0)
            chunk = shuffled_users[offset : offset + size]
            for user_id in chunk:
                user_to_client[int(user_id)] = client_id
            offset += size
        return user_to_client

    def _assign_by_interactions(
        self,
        shuffled_users: np.ndarray,
        counts_by_user: Mapping[int, int],
        n_train: int,
    ) -> Dict[int, int]:
        """Greedy-fill clients toward uniform interaction targets.

        Parameters
        ----------
        shuffled_users : numpy.ndarray
            Shuffled user ids.
        counts_by_user : mapping of int to int
            Training interaction count per user.
        n_train : int
            Total number of training interactions.

        Returns
        -------
        dict of int to int
            Mapping from user id to client id.
        """
        target = float(n_train) / float(self.num_clients)
        sizes = np.zeros(self.num_clients, dtype=np.float64)
        user_to_client: Dict[int, int] = {}
        for user_id in shuffled_users:
            uid = int(user_id)
            deficits = target - sizes
            client_id = int(np.argmax(deficits))
            user_to_client[uid] = client_id
            sizes[client_id] += float(counts_by_user[uid])
        return user_to_client


class DirichletUserPartitionStrategy(BaseClientPartitionStrategy):
    """Assign whole users to clients with Dirichlet quantity skew.

    Users are sorted, shuffled with a private NumPy generator, then
    assigned so that client sizes follow a symmetric Dirichlet draw.
    Small ``alpha`` yields a few large clients and many small ones;
    large ``alpha`` approaches equal sizes.

    With ``size_by="interactions"`` (default) each client is seeded
    with one user, then remaining users are greedy-filled toward
    interaction targets ``p_k * N_train``. With ``size_by="users"``
    extra user counts are drawn from a Multinomial, so final user
    counts are exactly ``1 + m_k``.

    Cross-silo settings use ``1 ≤ num_clients < num_users``. Setting
    ``num_clients`` equal to the number of training users yields one
    user per client; ``alpha`` and ``size_by`` then have no effect.
    Setting ``num_clients`` to ``1`` places every user on a single
    client.

    Parameters
    ----------
    num_clients : int
        Number of clients. Must be an integer ``>= 1``. The value must
        also satisfy ``num_clients <=`` the number of distinct training
        users when :meth:`partition` runs.
    alpha : float
        Positive concentration of the symmetric Dirichlet. Must be
        ``> 0``.
    size_by : {"interactions", "users"}, optional
        Quantity to skew across clients. ``"interactions"`` (default)
        greedy-fills against Dirichlet interaction targets.
        ``"users"`` draws Multinomial user counts.
    seed : int or None, optional
        Seed for the strategy's ``numpy.random.Generator``. ``None``
        leaves the generator unseeded. Default is None.
    user_col : str, optional
        User-id column in the training frame. Default is ``"user_id"``.
    metadata : dict or None, optional
        Stored on the strategy and copied onto the returned partition.
        Default is None.

    Raises
    ------
    ValueError
        If ``num_clients`` is not an integer ``>= 1``, ``alpha`` is not
        a positive real number, or ``size_by`` is not ``"interactions"``
        or ``"users"``. :meth:`partition` also raises if ``user_col`` is
        missing or ``num_clients`` exceeds the number of distinct
        training users.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 0, 1, 2, 2, 3],
    ...     "item_id": [10, 11, 10, 12, 13, 10],
    ... })
    >>> strategy = DirichletUserPartitionStrategy(2, alpha=0.5, seed=0)
    >>> part = strategy.partition(df)
    >>> part.num_clients
    2
    >>> sorted(part.user_ids_2_client_ids)
    [0, 1, 2, 3]
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float,
        size_by: str = "interactions",
        seed: int | None = None,
        user_col: str = "user_id",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(num_clients, int)
            or isinstance(num_clients, bool)
            or num_clients < 1
        ):
            raise ValueError(
                f"num_clients must be an integer >= 1, got {num_clients!r}"
            )
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(
                f"alpha must be a positive real number, got {alpha!r}"
            )
        if float(alpha) <= 0.0:
            raise ValueError(
                f"alpha must be a positive real number, got {alpha!r}"
            )
        if size_by not in {"interactions", "users"}:
            raise ValueError(
                "size_by must be 'interactions' or 'users', "
                f"got {size_by!r}"
            )
        super().__init__(
            num_clients=num_clients, user_col=user_col, metadata=metadata
        )
        self.alpha = float(alpha)
        self.size_by = size_by
        self.seed = seed

    def partition(self, train_interactions: pd.DataFrame) -> ClientPartition:
        """Assign every training user to one client and build the partition.

        Parameters
        ----------
        train_interactions : pandas.DataFrame
            Training interactions. Must contain ``self.user_col``.

        Returns
        -------
        ClientPartition
            Partition over ``self.num_clients`` clients.

        Raises
        ------
        ValueError
            If ``user_col`` is missing, or there are fewer distinct
            training users than ``num_clients``.
        """
        if self.user_col not in train_interactions.columns:
            raise ValueError(
                f"train_interactions must contain a '{self.user_col}' "
                "column with user identifiers"
            )

        user_ids = np.sort(
            train_interactions[self.user_col].unique().astype(np.int64)
        )
        num_users = int(user_ids.size)
        if self.num_clients > num_users:
            raise ValueError(
                f"num_clients ({self.num_clients}) must be <= number of "
                f"distinct training users ({num_users})"
            )

        if self.num_clients == num_users:
            logging.warning(
                "DirichletUserPartitionStrategy: num_clients equals the "
                "number of training users (%s); alpha and size_by have "
                "no effect",
                num_users,
            )

        rng = np.random.default_rng(self.seed)
        shuffled = user_ids.copy()
        rng.shuffle(shuffled)
        proportions = rng.dirichlet(
            np.full(self.num_clients, self.alpha, dtype=np.float64)
        )

        user_interaction_counts = (
            train_interactions.groupby(self.user_col).size().astype(np.int64)
        )
        counts_by_user = {
            int(uid): int(user_interaction_counts.loc[uid])
            for uid in user_ids
        }

        if self.size_by == "users":
            user_to_client = self._assign_by_users(
                shuffled, proportions, rng
            )
            target_proportion: float | Sequence[float] | None = None
        else:
            user_to_client = self._assign_by_interactions(
                shuffled,
                counts_by_user,
                proportions,
                n_train=len(train_interactions),
            )
            target_proportion = proportions

        partition = _build_client_partition(
            train_interactions,
            self.user_col,
            user_to_client,
            self.num_clients,
            metadata=self.metadata,
        )
        _log_partition_sizes(
            partition.client_ids_2_user_ids,
            partition.client_train_sizes,
            target_interaction_proportion=target_proportion,
        )
        return partition

    def _assign_by_users(
        self,
        shuffled_users: np.ndarray,
        proportions: np.ndarray,
        rng: np.random.Generator,
    ) -> Dict[int, int]:
        """Seed each client, then allocate Multinomial extra users.

        Parameters
        ----------
        shuffled_users : numpy.ndarray
            Shuffled user ids of length ``num_users``.
        proportions : numpy.ndarray
            Dirichlet proportions of length ``num_clients``.
        rng : numpy.random.Generator
            Generator used for the Multinomial draw.

        Returns
        -------
        dict of int to int
            Mapping from user id to client id. Final user counts are
            exactly ``1 + m_k`` for Multinomial extras ``m``.
        """
        user_to_client: Dict[int, int] = {}
        for client_id in range(self.num_clients):
            user_to_client[int(shuffled_users[client_id])] = client_id

        remaining = int(shuffled_users.size) - self.num_clients
        if remaining > 0:
            extras = rng.multinomial(remaining, proportions)
            offset = self.num_clients
            for client_id, extra_count in enumerate(extras):
                count = int(extra_count)
                for user_id in shuffled_users[offset : offset + count]:
                    user_to_client[int(user_id)] = client_id
                offset += count
        return user_to_client

    def _assign_by_interactions(
        self,
        shuffled_users: np.ndarray,
        counts_by_user: Mapping[int, int],
        proportions: np.ndarray,
        n_train: int,
    ) -> Dict[int, int]:
        """Seed each client, then greedy-fill toward Dirichlet targets.

        Parameters
        ----------
        shuffled_users : numpy.ndarray
            Shuffled user ids.
        counts_by_user : mapping of int to int
            Training interaction count per user.
        proportions : numpy.ndarray
            Dirichlet proportions of length ``num_clients``.
        n_train : int
            Total number of training interactions.

        Returns
        -------
        dict of int to int
            Mapping from user id to client id.
        """
        targets = proportions * float(n_train)
        sizes = np.zeros(self.num_clients, dtype=np.float64)
        user_to_client: Dict[int, int] = {}

        for client_id in range(self.num_clients):
            uid = int(shuffled_users[client_id])
            user_to_client[uid] = client_id
            sizes[client_id] += float(counts_by_user[uid])

        for user_id in shuffled_users[self.num_clients :]:
            uid = int(user_id)
            deficits = targets - sizes
            client_id = int(np.argmax(deficits))
            user_to_client[uid] = client_id
            sizes[client_id] += float(counts_by_user[uid])
        return user_to_client


# Previous name. Kept so existing imports continue to resolve to the same class.
CrossDeviceSplittingStrategy = CrossDevicePartitionStrategy
