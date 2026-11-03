"""Expansion sampling follows expansion_seed, else the server seed."""

import numpy as np

from fedreclab.methods.fedpergnn.fedpergnn import FedPerGNN


def _server(expansion_seed, seed):
    server = FedPerGNN.__new__(FedPerGNN)
    server.expansion_seed = expansion_seed
    server.seed = seed
    return server


def test_expansion_falls_back_to_server_seed():
    rng = _server(expansion_seed=None, seed=0)._expansion_rng()
    assert rng.integers(1000) == np.random.default_rng(0).integers(1000)


def test_explicit_expansion_seed_wins_over_server_seed():
    drawn = _server(expansion_seed=7, seed=0)._expansion_rng().integers(1000)
    assert drawn == np.random.default_rng(7).integers(1000)
    assert drawn != np.random.default_rng(0).integers(1000)


def test_unseeded_expansion_uses_numpy_global():
    assert _server(expansion_seed=None, seed=None)._expansion_rng() is np.random
