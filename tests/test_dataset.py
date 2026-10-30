"""Split indices are row positions; the frame index must not affect labelling."""

import numpy as np
import pandas as pd

from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.load_data import load_dataframe_dataset
from src.fedreclab.data.split import RatioSplitStrategy


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
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
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
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
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
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
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
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
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
