"""FedAtt scores each dict key independently (Ji et al. / official repo)."""

import math

import numpy as np
import torch

from fedreclab.federated.aggregation import FedAtt
from fedreclab.federated.utils import ParamIO, RoundStatistics


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


def test_dict_keys_get_independent_attention():
    """Complementary distances must not share one group-level softmax."""
    prev = {"w": np.array([0.0]), "b": np.array([0.0])}
    updates = [
        {"w": np.array([1.0]), "b": np.array([0.0])},
        {"w": np.array([0.0]), "b": np.array([1.0])},
    ]
    out = FedAtt(2).aggregate(_info(), prev, updates, _pio())

    # softmax([1, 0]) weights the far client by e / (e + 1).
    expected = math.e / (math.e + 1.0)
    np.testing.assert_allclose(out["w"], [expected])
    np.testing.assert_allclose(out["b"], [expected])


def test_single_tensor_matches_per_key_on_that_tensor():
    prev = np.array([0.0, 0.0])
    updates = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    out = FedAtt(2).aggregate(_info(), prev, updates, _pio())

    # Equal L2 distances -> equal attention -> mean of the two updates.
    np.testing.assert_allclose(out, [0.5, 0.5])


def test_row_layout_uses_attention_not_fedavg_mean():
    """FedAtt must not inherit FedAvg.aggregate_rows (that path is a mean)."""
    prev = np.array([[0.0]])
    values = np.array([[[1.0]], [[0.0]]])
    mask = np.array([[True], [True]])
    out = FedAtt(2).aggregate_rows(_info(), prev, values, mask, _pio())
    expected = math.e / (math.e + 1.0)
    np.testing.assert_allclose(out[0], [expected])


def test_row_dict_tables_do_not_share_attention():
    """aggregate_rows feeds each item as a dict; GMF/MLP must not share α."""
    prev = {
        "gmf": np.array([[0.0], [9.0]]),
        "mlp": np.array([[0.0], [9.0]]),
    }
    values = {
        "gmf": np.array([[[1.0], [9.0]], [[0.0], [9.0]]]),
        "mlp": np.array([[[0.0], [9.0]], [[1.0], [9.0]]]),
    }
    mask = np.array([[True, True], [True, True]])
    out = FedAtt(2).aggregate_rows(_info(), prev, values, mask, _pio())

    expected = math.e / (math.e + 1.0)
    np.testing.assert_allclose(out["gmf"][0], [expected])
    np.testing.assert_allclose(out["mlp"][0], [expected])
    # Row 1 is identical across clients and equal to prev, so it stays 9.
    np.testing.assert_allclose(out["gmf"][1], [9.0])
    np.testing.assert_allclose(out["mlp"][1], [9.0])


def test_torch_dict_preserves_dtype_and_device():
    prev = {"w": torch.tensor([0.0], dtype=torch.float32)}
    updates = [
        {"w": torch.tensor([1.0], dtype=torch.float32)},
        {"w": torch.tensor([0.0], dtype=torch.float32)},
    ]
    out = FedAtt(2).aggregate(_info(), prev, updates, _pio())
    assert isinstance(out["w"], torch.Tensor)
    assert out["w"].dtype == torch.float32
    expected = math.e / (math.e + 1.0)
    np.testing.assert_allclose(out["w"].detach().cpu().numpy(), [expected])
