from src.fedreclab.data.split import LeaveNOutSplitStrategy, RatioSplitStrategy
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Generates synthetic interaction data."""
    return pd.DataFrame(
        {
            "user_id": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
            ],  # User A: 4, User B: 3, User C: 5
            "item_id": [
                101,
                102,
                103,
                104,
                101,
                105,
                103,
                101,
                102,
                103,
                104,
                105,
                106,
            ],
            "timestamp": [10, 20, 30, 40, 15, 25, 30, 5, 10, 15, 20, 25, 30],
            "rating": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2],
        }
    )


def test_chronological_split(sample_data):
    """Verify that latest interactions go to test/val when chrono=True."""
    splitter = LeaveNOutSplitStrategy(
        include_validation=True,
        include_test=True,
        n_val=1,
        n_test=1,
        chrono=True,
        rating_threshold=3,
    )
    train_idx, val_idx, test_idx = splitter.split(sample_data)

    # User A (4 interactions): rows 0,1 -> train, 2 -> val, 3 -> test
    assert 3 in test_idx
    assert 6 in test_idx
    assert 11 in test_idx
    assert 2 in val_idx
    assert 5 in val_idx
    assert 10 in val_idx
    assert (
        0 in train_idx
        and 1 in train_idx
        and 4 in train_idx
        and 7 in train_idx
        and 8 in train_idx
        and 9 in train_idx
    )


def test_random_split(sample_data):
    """Verify that latest interactions go to test/val when chrono=True."""
    splitter = LeaveNOutSplitStrategy(
        include_validation=True,
        include_test=True,
        n_val=1,
        n_test=1,
        chrono=True,
        rating_threshold=3,
    )
    train_idx, val_idx, test_idx = splitter.split(sample_data)

    # User A (4 interactions): rows 0,1 -> train, 2 -> val, 3 -> test
    assert 3 == len(test_idx)
    assert 3 == len(val_idx)
    # print(train_idx)
    assert 6 == len(train_idx)


def test_no_val_or_test_enabled(sample_data):
    """When both flags are False, all rows return in train_idx."""
    splitter = LeaveNOutSplitStrategy(include_validation=False, include_test=False)
    train_idx, val_idx, test_idx = splitter.split(sample_data)
    assert len(train_idx) == len(sample_data)
    assert val_idx is None
    assert test_idx is None
