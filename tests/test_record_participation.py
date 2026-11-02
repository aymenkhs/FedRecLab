"""Server participation counts increment per selected client and log coverage."""

import logging

import numpy as np

from src.fedreclab.federated.base_server import BaseFederatedServer


def _server(num_clients=4, current_round=0):
    server = object.__new__(BaseFederatedServer)
    server.participation_counts = np.zeros(num_clients, dtype=np.int64)
    server.current_round = current_round
    return server


def test_record_participation_increments_only_selected_clients():
    server = _server()
    server._record_participation([0, 2])
    np.testing.assert_array_equal(server.participation_counts, [1, 0, 1, 0])


def test_record_participation_accumulates_across_rounds():
    server = _server()
    server._record_participation([0, 1])
    server.current_round = 1
    server._record_participation([1, 3])
    np.testing.assert_array_equal(server.participation_counts, [1, 2, 0, 1])


def test_record_participation_logs_coverage_summary(caplog):
    server = _server()
    caplog.set_level(logging.INFO, logger="src.fedreclab.federated.base_server")
    server._record_participation([0, 2])
    assert (
        "Round 0 selected [0, 2]; participation min/max/mean=0/1/0.500; "
        "never selected=2"
    ) in caplog.text


def test_record_participation_empty_selection_leaves_counts_unchanged(caplog):
    server = _server()
    caplog.set_level(logging.INFO, logger="src.fedreclab.federated.base_server")
    server._record_participation([])
    np.testing.assert_array_equal(server.participation_counts, [0, 0, 0, 0])
    assert (
        "Round 0 selected []; participation min/max/mean=0/0/0.000; "
        "never selected=4"
    ) in caplog.text
