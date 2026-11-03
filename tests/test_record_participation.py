"""Server participation counts increment per selected client and log coverage."""

import logging

import numpy as np

from fedreclab.federated.base_server import BaseFederatedServer


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


def test_record_participation_logs_summary_at_info_and_ids_at_debug(caplog):
    server = _server()
    caplog.set_level(logging.DEBUG, logger="fedreclab.federated.base_server")
    server._record_participation([0, 2])
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert info == [
        "Round 0 selected 2 clients; participation min/max/mean=0/1/0.500; "
        "never selected=2"
    ]
    assert debug == ["Round 0 selected client ids: [0, 2]"]


def test_record_participation_empty_selection_leaves_counts_unchanged(caplog):
    server = _server()
    caplog.set_level(logging.INFO, logger="fedreclab.federated.base_server")
    server._record_participation([])
    np.testing.assert_array_equal(server.participation_counts, [0, 0, 0, 0])
    assert (
        "Round 0 selected 0 clients; participation min/max/mean=0/0/0.000; "
        "never selected=4"
    ) in caplog.text
    assert "selected client ids" not in caplog.text
