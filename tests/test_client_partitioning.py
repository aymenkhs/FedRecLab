"""Tests for user partition strategies and shared partition helpers."""

import logging

import numpy as np
import pandas as pd
import pytest

from fedreclab.data.client_partitioning import (
    DirichletUserPartitionStrategy,
    RandomUserPartitionStrategy,
    _build_client_partition,
)


def _toy_frame():
    """Six users with unequal activity; user 0 is the most active."""
    rows = []
    # user 0: 10 interactions; users 1..5: 1 each
    for item in range(10):
        rows.append({"user_id": 0, "item_id": item})
    for user_id in range(1, 6):
        rows.append({"user_id": user_id, "item_id": 0})
    return pd.DataFrame(rows)


def test_constructor_rejects_invalid_num_clients():
    with pytest.raises(ValueError, match="num_clients"):
        RandomUserPartitionStrategy(0)
    with pytest.raises(ValueError, match="num_clients"):
        RandomUserPartitionStrategy(-1)
    with pytest.raises(ValueError, match="num_clients"):
        RandomUserPartitionStrategy(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_clients"):
        RandomUserPartitionStrategy(True)  # type: ignore[arg-type]


def test_constructor_rejects_invalid_size_by():
    with pytest.raises(ValueError, match="size_by"):
        RandomUserPartitionStrategy(2, size_by="rows")


def test_partition_rejects_too_many_clients():
    df = _toy_frame()
    strategy = RandomUserPartitionStrategy(7, seed=0)
    with pytest.raises(ValueError, match="num_clients"):
        strategy.partition(df)


def test_users_mode_assigns_every_user_once_and_balances_counts():
    df = _toy_frame()
    part = RandomUserPartitionStrategy(3, size_by="users", seed=0).partition(df)

    assert part.num_clients == 3
    assert sorted(part.user_ids_2_client_ids) == list(range(6))
    assert len(set(part.user_ids_2_client_ids)) == 6

    user_counts = [
        len(part.client_ids_2_user_ids[c]) for c in range(3)
    ]
    assert max(user_counts) - min(user_counts) <= 1
    assert sum(user_counts) == 6

    for client_id, users in part.client_ids_2_user_ids.items():
        assert users == sorted(users)
        for user_id in users:
            assert part.user_ids_2_client_ids[user_id] == client_id


def test_users_mode_same_seed_is_stable_and_row_order_independent():
    df = _toy_frame()
    strategy = RandomUserPartitionStrategy(3, size_by="users", seed=42)
    first = strategy.partition(df)
    second = strategy.partition(df)
    assert first.user_ids_2_client_ids == second.user_ids_2_client_ids

    reversed_df = df.iloc[::-1].copy()
    from_reversed = strategy.partition(reversed_df)
    assert from_reversed.user_ids_2_client_ids == first.user_ids_2_client_ids


def test_interactions_mode_matches_greedy_oracle_and_has_no_empty_client():
    df = _toy_frame()
    num_clients = 3
    seed = 7
    strategy = RandomUserPartitionStrategy(
        num_clients, size_by="interactions", seed=seed
    )
    part = strategy.partition(df)

    user_ids = np.sort(df["user_id"].unique().astype(np.int64))
    counts = df.groupby("user_id").size().astype(np.int64)
    counts_by_user = {int(uid): int(counts.loc[uid]) for uid in user_ids}
    rng = np.random.default_rng(seed)
    shuffled = user_ids.copy()
    rng.shuffle(shuffled)
    target = float(len(df)) / float(num_clients)
    sizes = np.zeros(num_clients, dtype=np.float64)
    expected = {}
    for user_id in shuffled:
        uid = int(user_id)
        client_id = int(np.argmax(target - sizes))
        expected[uid] = client_id
        sizes[client_id] += float(counts_by_user[uid])

    assert part.user_ids_2_client_ids == expected
    assert all(
        len(part.client_ids_2_user_ids[c]) > 0 for c in range(num_clients)
    )
    assert sorted(part.user_ids_2_client_ids) == list(range(6))


def test_interactions_mode_preserves_non_range_index_labels():
    df = _toy_frame()
    df.index = pd.Index(range(100, 100 + len(df)))
    part = RandomUserPartitionStrategy(
        2, size_by="interactions", seed=1
    ).partition(df)

    all_indices = []
    for client_id in range(2):
        all_indices.extend(part.client_to_train_indices[client_id])
    assert sorted(all_indices) == list(df.index)
    assert sum(part.client_train_sizes.values()) == len(df)


def test_num_clients_one_puts_every_user_on_client_zero():
    df = _toy_frame()
    part = RandomUserPartitionStrategy(1, seed=0).partition(df)
    assert part.num_clients == 1
    assert part.client_ids_2_user_ids[0] == list(range(6))
    assert set(part.user_ids_2_client_ids.values()) == {0}
    assert part.client_train_sizes[0] == len(df)


def test_num_clients_equals_num_users_one_user_per_client():
    df = _toy_frame()
    for size_by in ("users", "interactions"):
        part = RandomUserPartitionStrategy(
            6, size_by=size_by, seed=0
        ).partition(df)
        assert part.num_clients == 6
        for client_id in range(6):
            assert len(part.client_ids_2_user_ids[client_id]) == 1


def test_seed_none_stays_none_and_does_not_touch_global_numpy_state():
    df = _toy_frame()
    state_before = np.random.get_state()
    strategy = RandomUserPartitionStrategy(3, seed=None)
    assert strategy.seed is None
    strategy.partition(df)
    assert strategy.seed is None
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])


def test_metadata_is_copied_onto_partition():
    df = _toy_frame()
    meta = {"source": "unit-test"}
    part = RandomUserPartitionStrategy(
        2, seed=0, metadata=meta
    ).partition(df)
    assert part.metadata == meta


def test_build_client_partition_helper_sorts_users_and_sizes():
    df = pd.DataFrame(
        {
            "user_id": [1, 0, 0, 1],
            "item_id": [10, 11, 12, 13],
        },
        index=[5, 6, 7, 8],
    )
    part = _build_client_partition(
        df, "user_id", {0: 0, 1: 1}, num_clients=2, metadata=None
    )
    assert part.client_ids_2_user_ids[0] == [0]
    assert part.client_ids_2_user_ids[1] == [1]
    assert part.client_to_train_indices[0] == [6, 7]
    assert part.client_to_train_indices[1] == [5, 8]
    assert part.client_train_sizes == {0: 2, 1: 2}


def test_dirichlet_constructor_rejects_invalid_num_clients():
    with pytest.raises(ValueError, match="num_clients"):
        DirichletUserPartitionStrategy(0, alpha=0.5)
    with pytest.raises(ValueError, match="num_clients"):
        DirichletUserPartitionStrategy(-1, alpha=0.5)
    with pytest.raises(ValueError, match="num_clients"):
        DirichletUserPartitionStrategy(1.5, alpha=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_clients"):
        DirichletUserPartitionStrategy(True, alpha=0.5)  # type: ignore[arg-type]


def test_dirichlet_constructor_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        DirichletUserPartitionStrategy(2, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        DirichletUserPartitionStrategy(2, alpha=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        DirichletUserPartitionStrategy(2, alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        DirichletUserPartitionStrategy(2, alpha="0.5")  # type: ignore[arg-type]


def test_dirichlet_constructor_rejects_invalid_size_by_and_defaults():
    with pytest.raises(ValueError, match="size_by"):
        DirichletUserPartitionStrategy(2, alpha=0.5, size_by="rows")
    strategy = DirichletUserPartitionStrategy(2, alpha=0.5)
    assert strategy.size_by == "interactions"
    assert strategy.alpha == 0.5


def test_dirichlet_partition_rejects_too_many_clients():
    df = _toy_frame()
    strategy = DirichletUserPartitionStrategy(7, alpha=0.5, seed=0)
    with pytest.raises(ValueError, match="num_clients"):
        strategy.partition(df)


def test_dirichlet_users_mode_matches_multinomial_oracle():
    df = _toy_frame()
    num_clients = 3
    alpha = 0.5
    seed = 11
    part = DirichletUserPartitionStrategy(
        num_clients, alpha=alpha, size_by="users", seed=seed
    ).partition(df)

    user_ids = np.sort(df["user_id"].unique().astype(np.int64))
    rng = np.random.default_rng(seed)
    shuffled = user_ids.copy()
    rng.shuffle(shuffled)
    proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    extras = rng.multinomial(int(user_ids.size) - num_clients, proportions)

    expected: dict[int, int] = {}
    for client_id in range(num_clients):
        expected[int(shuffled[client_id])] = client_id
    offset = num_clients
    for client_id, extra_count in enumerate(extras):
        for user_id in shuffled[offset : offset + int(extra_count)]:
            expected[int(user_id)] = client_id
        offset += int(extra_count)

    assert part.user_ids_2_client_ids == expected
    assert sorted(part.user_ids_2_client_ids) == list(range(6))
    for client_id in range(num_clients):
        users = part.client_ids_2_user_ids[client_id]
        assert users == sorted(users)
        assert len(users) == 1 + int(extras[client_id])
        assert len(users) > 0


def test_dirichlet_interactions_mode_matches_greedy_oracle():
    df = _toy_frame()
    num_clients = 3
    alpha = 0.3
    seed = 7
    part = DirichletUserPartitionStrategy(
        num_clients, alpha=alpha, size_by="interactions", seed=seed
    ).partition(df)

    user_ids = np.sort(df["user_id"].unique().astype(np.int64))
    counts = df.groupby("user_id").size().astype(np.int64)
    counts_by_user = {int(uid): int(counts.loc[uid]) for uid in user_ids}
    rng = np.random.default_rng(seed)
    shuffled = user_ids.copy()
    rng.shuffle(shuffled)
    proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    targets = proportions * float(len(df))
    sizes = np.zeros(num_clients, dtype=np.float64)
    expected: dict[int, int] = {}
    for client_id in range(num_clients):
        uid = int(shuffled[client_id])
        expected[uid] = client_id
        sizes[client_id] += float(counts_by_user[uid])
    for user_id in shuffled[num_clients:]:
        uid = int(user_id)
        client_id = int(np.argmax(targets - sizes))
        expected[uid] = client_id
        sizes[client_id] += float(counts_by_user[uid])

    assert part.user_ids_2_client_ids == expected
    assert all(
        len(part.client_ids_2_user_ids[c]) > 0 for c in range(num_clients)
    )
    assert sorted(part.user_ids_2_client_ids) == list(range(6))


def test_dirichlet_same_seed_is_stable_and_row_order_independent():
    df = _toy_frame()
    strategy = DirichletUserPartitionStrategy(
        3, alpha=0.5, size_by="users", seed=42
    )
    first = strategy.partition(df)
    second = strategy.partition(df)
    assert first.user_ids_2_client_ids == second.user_ids_2_client_ids

    reversed_df = df.iloc[::-1].copy()
    from_reversed = strategy.partition(reversed_df)
    assert from_reversed.user_ids_2_client_ids == first.user_ids_2_client_ids


def test_dirichlet_seed_none_stays_none_and_does_not_touch_global_numpy_state():
    df = _toy_frame()
    state_before = np.random.get_state()
    strategy = DirichletUserPartitionStrategy(3, alpha=0.5, seed=None)
    assert strategy.seed is None
    strategy.partition(df)
    assert strategy.seed is None
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])


def test_dirichlet_num_clients_one_puts_every_user_on_client_zero():
    df = _toy_frame()
    part = DirichletUserPartitionStrategy(1, alpha=0.5, seed=0).partition(df)
    assert part.num_clients == 1
    assert part.client_ids_2_user_ids[0] == list(range(6))
    assert set(part.user_ids_2_client_ids.values()) == {0}
    assert part.client_train_sizes[0] == len(df)


def test_dirichlet_num_clients_equals_num_users_warns_and_one_each(caplog):
    df = _toy_frame()
    with caplog.at_level(logging.WARNING):
        part = DirichletUserPartitionStrategy(
            6, alpha=0.1, size_by="interactions", seed=0
        ).partition(df)
    assert part.num_clients == 6
    for client_id in range(6):
        assert len(part.client_ids_2_user_ids[client_id]) == 1
    assert any(
        "alpha and size_by have no effect" in record.message
        for record in caplog.records
    )


def test_dirichlet_metadata_is_copied_onto_partition():
    df = _toy_frame()
    meta = {"source": "dirichlet-unit-test"}
    part = DirichletUserPartitionStrategy(
        2, alpha=1.0, seed=0, metadata=meta
    ).partition(df)
    assert part.metadata == meta
