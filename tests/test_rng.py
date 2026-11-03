"""Selection and sampling follow the passed seed, not global RNGs."""

import random

import numpy as np

from fedreclab.federated.client_selection import RandomClientSelection
from fedreclab.methods.fedncf.fedncf_client import FedNCFClientAlgorithm


def test_unseeded_selection_returns_the_asked_number_of_clients():
    chosen = RandomClientSelection(20, 5, "num").select()
    assert len(chosen) == 5
    assert len(set(chosen)) == 5


def test_same_seed_same_selection():
    a = RandomClientSelection(20, 5, "num", seed=0)
    b = RandomClientSelection(20, 5, "num", seed=0)
    assert a.select(0) == b.select(0)
    assert a.select(0) != a.select(1)


def test_selection_ignores_the_random_module():
    random.seed(1)
    a = RandomClientSelection(20, 5, "num", seed=0).select()
    random.seed(99)
    b = RandomClientSelection(20, 5, "num", seed=0).select()
    assert a == b


def test_unseeded_selection_does_not_consume_global_numpy_rng():
    np.random.seed(123)
    RandomClientSelection(20, 5, "num").select()
    got = np.random.rand(4)
    np.random.seed(123)
    expected = np.random.rand(4)
    np.testing.assert_array_equal(got, expected)


def test_negative_sampling_uses_the_passed_seed():
    client = FedNCFClientAlgorithm(neg_samples=3, predictive_factor=4)
    client.num_items = 8
    args = (np.array([0, 0]), np.array([1, 2]), np.array([1.0, 1.0]), {0: {1, 2}})
    np.random.seed(1)
    x1, _ = client._build_epoch_samples(*args, np.random.default_rng([0, 0, 0]))
    np.random.seed(99)
    x2, _ = client._build_epoch_samples(*args, np.random.default_rng([0, 0, 0]))
    np.testing.assert_array_equal(x1, x2)
