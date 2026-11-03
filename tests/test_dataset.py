"""Split indices are row positions; the frame index must not affect labelling."""

import numpy as np
import pandas as pd
import pytest

from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from fedreclab.data.dataset import InteractionDataset
from fedreclab.data.load_data import load_dataframe_dataset
from fedreclab.data.split import RatioSplitStrategy


def _rating_filter_frame():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "item_id": [10, 11, 12, 10, 11, 13, 10, 11, 12, 13],
            "rating": [5.0, 2.0, 4.0, 1.0, 5.0, 4.0, 5.0, 2.0, 4.0, 5.0],
        }
    )


def test_load_dataframe_k_core_zero_rating_filter_uses_positions():
    dataset = load_dataframe_dataset(
        df=_rating_filter_frame(),
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=False
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=0,
        rating_exclusion_threshold=3.0,
        time_col=None,
    )

    n = len(dataset.interaction_data)
    assert n == 7
    assert list(dataset.interaction_data.index) == list(range(n))
    assert list(dataset.train_idx) == list(range(n))
    assert (dataset.interaction_data["split"] == "train").all()
    assert len(dataset.train_interactions) == n


def test_load_dataframe_k_core_zero_rating_filter_ratio_split():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "user_id": rng.integers(0, 20, size=n),
            "item_id": rng.integers(0, 20, size=n),
            "rating": rng.choice([1.0, 2.0, 4.0, 5.0], size=n),
        }
    )
    dataset = load_dataframe_dataset(
        df=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=True,
            include_test=True,
            val_size=0.1,
            test_size=0.2,
            seed=42,
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        include_validation=True,
        include_test=True,
        k_core=0,
        rating_exclusion_threshold=3.0,
        time_col=None,
    )

    assert list(dataset.interaction_data.index) == list(
        range(len(dataset.interaction_data))
    )
    assert (dataset.train_interactions["split"] == "train").all()
    assert (dataset.val_interactions["split"] == "val").all()
    assert (dataset.test_interactions["split"] == "test").all()


def test_splits_included_reads_positions_from_noncontiguous_index():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 11, 10, 11, 10],
            "rating": [5.0] * 5,
            "split": ["train", "test", "train", "test", "train"],
        },
        index=[0, 2, 4, 6, 8],
    )
    dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=True
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=0,
        include_test=True,
        splits_included=True,
        time_col=None,
    )

    np.testing.assert_array_equal(dataset.train_idx, np.array([0, 2, 4]))
    np.testing.assert_array_equal(dataset.test_idx, np.array([1, 3]))
    assert (dataset.train_interactions["split"] == "train").all()
    assert (dataset.test_interactions["split"] == "test").all()


def test_split_data_labels_match_positional_indices_on_permuted_index():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [0, 1, 0, 1, 0, 1],
            "rating": [5.0] * 6,
        },
        index=[5, 3, 1, 0, 2, 4],
    )
    dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=True, test_size=0.3, seed=0
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=0,
        include_test=True,
        time_col=None,
    )
    dataset.split_data()

    assert list(dataset.interaction_data.index) == list(range(len(df)))
    assert (dataset.train_interactions["split"] == "train").all()
    assert (dataset.test_interactions["split"] == "test").all()
    labelled_train = set(np.flatnonzero(dataset.interaction_data["split"] == "train"))
    assert labelled_train == set(dataset.train_idx)


def _cascade_frame():
    """Users 0 and 1 interact with {10, 11}; user 2 interacts with {11, 12}."""
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [10, 11, 10, 11, 11, 12],
            "rating": [5.0] * 6,
        }
    )


def _interaction_dataset(df, **kwargs):
    kwargs.setdefault(
        "splitting_strategy",
        RatioSplitStrategy(include_validation=False, include_test=False),
    )
    kwargs.setdefault(
        "client_partitioning_strategy",
        CrossDevicePartitionStrategy(),
    )
    kwargs.setdefault("time_col", None)
    return InteractionDataset(interaction_data=df, **kwargs)


def _original_ids(dataset):
    users = {dataset.reverse_user_id_map[i] for i in range(dataset.num_users)}
    items = {dataset.reverse_item_id_map[i] for i in range(dataset.num_items)}
    return users, items


def test_min_user_interactions_keeps_rare_items():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
            "item_id": [10, 11, 12, 13, 99, 10, 11, 12, 13, 14, 10, 77],
            "rating": [5.0] * 12,
        }
    )
    dataset = _interaction_dataset(df, min_user_interactions=5)

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1}
    assert original_items == {10, 11, 12, 13, 14, 99}
    assert 77 not in original_items
    assert dataset.num_interactions == 10


def test_k_core_removes_users_left_below_threshold():
    dataset = _interaction_dataset(_cascade_frame(), k_core=2)

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1}
    assert original_items == {10, 11}
    assert dataset.num_interactions == 4


def test_k_core_removes_rare_items_of_active_users():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "item_id": [10, 11, 12, 99, 10, 11, 12, 10, 11, 12],
            "rating": [5.0] * 10,
        }
    )
    dataset = _interaction_dataset(df, k_core=3)

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1, 2}
    assert original_items == {10, 11, 12}
    assert 99 not in original_items
    assert dataset.num_interactions == 9


def test_min_interaction_filter_is_single_pass():
    dataset = _interaction_dataset(
        _cascade_frame(),
        min_user_interactions=2,
        min_item_interactions=2,
    )

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1, 2}
    assert original_items == {10, 11}
    assert dataset.num_interactions == 5


def test_min_item_interactions_only():
    dataset = _interaction_dataset(
        _cascade_frame(),
        min_user_interactions=0,
        min_item_interactions=2,
    )

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1, 2}
    assert original_items == {10, 11}
    assert dataset.num_interactions == 5


def test_min_filter_runs_before_k_core():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            "item_id": [10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 99, 11, 99],
            "rating": [5.0] * 13,
        }
    )
    dataset = _interaction_dataset(
        df,
        min_user_interactions=0,
        min_item_interactions=3,
        k_core=2,
    )

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1, 2}
    assert original_items == {10, 11, 12}
    assert 99 not in original_items
    assert dataset.num_interactions == 9


def test_zero_thresholds_leave_data_unchanged():
    df = _cascade_frame()
    dataset = _interaction_dataset(df)

    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1, 2}
    assert original_items == {10, 11, 12}
    assert dataset.num_interactions == len(df)
    assert dataset.k_core == 0
    assert dataset.min_user_interactions == 0
    assert dataset.min_item_interactions == 0


def test_negative_threshold_raises():
    df = _cascade_frame()
    with pytest.raises(ValueError):
        _interaction_dataset(df, k_core=-1)
    with pytest.raises(ValueError):
        _interaction_dataset(df, min_user_interactions=-1)
    with pytest.raises(ValueError):
        _interaction_dataset(df, min_item_interactions=-1)


def test_filter_removing_everything_raises():
    df = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "item_id": [10, 11, 12],
            "rating": [5.0, 5.0, 5.0],
        }
    )
    with pytest.raises(ValueError):
        _interaction_dataset(df, k_core=2)
    with pytest.raises(ValueError):
        _interaction_dataset(df, min_user_interactions=2)


def test_load_dataframe_forwards_filters():
    dataset = load_dataframe_dataset(
        df=_cascade_frame(),
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=False
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=2,
        min_user_interactions=2,
        min_item_interactions=2,
        time_col=None,
    )

    assert dataset.k_core == 2
    assert dataset.min_user_interactions == 2
    assert dataset.min_item_interactions == 2
    original_users, original_items = _original_ids(dataset)
    assert original_users == {0, 1}
    assert original_items == {10, 11}
    assert dataset.num_interactions == 4
