"""ChangeMagnitudeAvg: L2 change weights for dense and row layouts."""

import numpy as np
import pytest
import torch

from src.fedreclab.federated.aggregation import ChangeMagnitudeAvg
from src.fedreclab.federated.utils import ParamIO, RoundStatistics


def _info(selected=(0, 1)):
    selected = list(selected)
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
    return ParamIO(residency="server", exchange="params", apply_to="server")


def test_dense_weights_by_l2_change():
    prev = np.array([0.0, 0.0])
    updates = [np.array([2.0, 0.0]), np.array([0.0, 1.0])]
    out = ChangeMagnitudeAvg(2).aggregate(_info(), prev, updates, _pio())
    np.testing.assert_allclose(out, [4.0 / 3.0, 1.0 / 3.0])


def test_change_eps_drops_unchanged_client():
    prev = np.array([1.0, 0.0])
    updates = [np.array([1.0, 0.0]), np.array([3.0, 0.0])]
    out = ChangeMagnitudeAvg(2, change_eps=0.0).aggregate(
        _info(), prev, updates, _pio()
    )
    np.testing.assert_allclose(out, [3.0, 0.0])


def test_raises_when_no_client_exceeds_eps():
    prev = np.array([1.0])
    updates = [np.array([1.0]), np.array([1.0])]
    with pytest.raises(ValueError, match="change_eps"):
        ChangeMagnitudeAvg(2).aggregate(_info(), prev, updates, _pio())


def test_dict_keys_weighted_independently():
    prev = {"a": np.array([0.0]), "b": np.array([0.0])}
    updates = [
        {"a": np.array([2.0]), "b": np.array([0.0])},
        {"a": np.array([0.0]), "b": np.array([1.0])},
    ]
    out = ChangeMagnitudeAvg(2).aggregate(_info(), prev, updates, _pio())
    np.testing.assert_allclose(out["a"], [2.0])
    np.testing.assert_allclose(out["b"], [1.0])


def test_row_layout_uses_per_column_change_weights():
    prev = np.array([[0.0], [0.0]])
    values = np.array([[[2.0], [0.0]], [[0.0], [1.0]]])
    mask = np.array([[True, False], [False, True]])
    out = ChangeMagnitudeAvg(2).aggregate_rows(
        _info(), prev, values, mask, _pio()
    )
    np.testing.assert_allclose(out[0], [2.0])
    np.testing.assert_allclose(out[1], [1.0])


def test_torch_output_preserves_dtype():
    prev = torch.tensor([0.0, 0.0], dtype=torch.float32)
    updates = [
        torch.tensor([2.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    ]
    out = ChangeMagnitudeAvg(2).aggregate(_info(), prev, updates, _pio())
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), [4.0 / 3.0, 1.0 / 3.0]
    )
