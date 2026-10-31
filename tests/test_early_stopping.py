"""Patience early stopping must name a ranking metric; it has no loss."""

import pytest

from src.fedreclab.federated.base_server import BaseFederatedServer


def _patience_server(early_stopping=True, parameters=None, custom=False):
    server = object.__new__(BaseFederatedServer)
    server.early_stopping = early_stopping
    server.early_stopping_custom = custom
    server.early_stopping_parameters = parameters
    server.early_stopping_callback = None
    return server


def test_patience_early_stopping_requires_explicit_metric():
    server = _patience_server(parameters={})
    with pytest.raises(ValueError, match="requires early_stopping_parameters\\['metric'\\]"):
        server._initialize_early_stopping()


def test_patience_early_stopping_requires_metric_when_parameters_are_none():
    server = _patience_server(parameters=None)
    with pytest.raises(ValueError, match="requires early_stopping_parameters\\['metric'\\]"):
        server._initialize_early_stopping()


def test_patience_early_stopping_rejects_loss_metric():
    server = _patience_server(parameters={"metric": "loss"})
    with pytest.raises(ValueError, match="metric 'loss' is not produced"):
        server._initialize_early_stopping()


def test_patience_early_stopping_defaults_direction_to_max():
    server = _patience_server(parameters={"metric": "ndcg@10"})
    server._initialize_early_stopping()
    assert server.early_stopping_parameters["metric"] == "ndcg@10"
    assert server.early_stopping_parameters["direction"] == "max"
    assert server.early_stopping_parameters["patience"] == 5
    assert server.early_stopping_best_result == float("-inf")


def test_disabled_early_stopping_does_not_require_a_metric():
    server = _patience_server(early_stopping=False, parameters=None)
    server._initialize_early_stopping()
    assert "metric" not in server.early_stopping_parameters
    assert server.early_stopping_parameters["direction"] == "max"


def test_custom_early_stopping_does_not_require_a_metric():
    server = _patience_server(parameters=None, custom=True)
    server._initialize_early_stopping()
    assert server.early_stopping_parameters is None
