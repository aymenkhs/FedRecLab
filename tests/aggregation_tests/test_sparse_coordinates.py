"""How aggregators treat a coordinate that only one client sent.

This is a property of the aggregators, not of a method. MetaMF's
per-user rows of U and FedMF's sparse item uploads both produce
coordinates that are non-zero in only a few client updates.
"""

import numpy as np

from fedreclab.federated.aggregation import Bulyan, Krum, Median, TrimmedMean
from fedreclab.federated.utils import ParamIO, RoundStatistics


def _info(n):
    selected = list(range(n))
    return RoundStatistics(
        current_round=0,
        selected=selected,
        responded=selected,
        dropped=[],
        num_training_examples={cid: 1 for cid in selected},
        training_losses={cid: [] for cid in selected},
        training_durations={cid: 0.0 for cid in selected},
    )


def _pio():
    return ParamIO(residency="server", exchange="gradient", apply_to="server")


def _sparse_updates(n, unique_value=6.0):
    """Coordinate 0 is ``unique_value`` on client 0 and 0 elsewhere.

    Coordinate 1 is ``i + 1`` on client ``i``.
    """
    return [
        np.array([unique_value if i == 0 else 0.0, float(i + 1)], dtype=np.float64)
        for i in range(n)
    ]


def test_median_k3_zeros_a_coordinate_only_one_client_sent():
    out = Median(3).aggregate(_info(3), None, _sparse_updates(3, 6.0), _pio())
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[1], 2.0)


def test_median_k2_is_the_mean_of_the_two_values():
    out = Median(2).aggregate(_info(2), None, _sparse_updates(2, 6.0), _pio())
    np.testing.assert_allclose(out[0], 3.0)
    np.testing.assert_allclose(out[1], 1.5)


def test_trimmed_mean_with_nothing_trimmed_is_the_mean():
    out = TrimmedMean(3, beta=0.0).aggregate(
        _info(3), None, _sparse_updates(3, 6.0), _pio()
    )
    np.testing.assert_allclose(out[0], 2.0)
    np.testing.assert_allclose(out[1], 2.0)


def test_trimmed_mean_drops_a_lone_nonzero_when_trim_reaches_it():
    n = 10
    out = TrimmedMean(n, beta=0.1).aggregate(
        _info(n), None, _sparse_updates(n, 6.0), _pio()
    )
    np.testing.assert_allclose(out[0], 0.0)


def test_krum_returns_one_clients_row_not_a_blend():
    updates = _sparse_updates(3, 6.0)
    out = Krum(3, num_byzantine=0).aggregate(_info(3), None, updates, _pio())
    first_coords = [float(u[0]) for u in updates]
    assert float(out[0]) in first_coords
    np.testing.assert_allclose(out[0], 0.0)


def test_multi_krum_averages_selected_clients_only():
    updates = _sparse_updates(3, 6.0)
    out = Krum(3, num_byzantine=0, multi_krum=True, num_selected=2).aggregate(
        _info(3), None, updates, _pio()
    )
    pairwise = {
        float(np.mean([updates[i][0], updates[j][0]]))
        for i in range(3)
        for j in range(i + 1, 3)
    }
    assert any(np.isclose(float(out[0]), value) for value in pairwise)
    assert not np.isclose(float(out[0]), float(np.mean([u[0] for u in updates])))


def test_bulyan_on_selected_clients_is_a_coordinate_wise_trimmed_mean():
    n = 7
    f = 1
    updates = _sparse_updates(n, 6.0)
    out = Bulyan(n, num_byzantine=f).aggregate(_info(n), None, updates, _pio())
    stacked = np.stack(updates, axis=0)
    krum = Krum(n, num_byzantine=f, multi_krum=True, num_selected=n - 2 * f)
    X = stacked.reshape(n, -1)
    chosen = krum._multi_krum_select(X, n - 2 * f)
    candidates = stacked[chosen]
    beta = n - 4 * f
    coord_median = np.median(candidates, axis=0)
    deviations = np.abs(candidates - coord_median)
    idx = np.argpartition(deviations, kth=beta - 1, axis=0)[:beta]
    expected = np.take_along_axis(candidates, idx, axis=0).mean(axis=0)
    np.testing.assert_allclose(out, expected)
