"""Client-count vs fraction resolution for RandomClientSelection."""

import pytest

from src.fedreclab.federated.client_selection import RandomClientSelection


def test_integer_count_is_not_treated_as_a_fraction():
    chosen = RandomClientSelection(20, 5).select()
    assert len(chosen) == 5
    assert len(set(chosen)) == 5


def test_float_is_treated_as_a_fraction():
    chosen = RandomClientSelection(20, 0.25).select()
    assert len(chosen) == 5


def test_frac_greater_than_one_raises():
    with pytest.raises(ValueError, match="fraction in \\(0, 1]"):
        RandomClientSelection(20, 20, "frac")


def test_count_larger_than_population_raises():
    with pytest.raises(ValueError, match="without replacement"):
        RandomClientSelection(20, 25, "num")
