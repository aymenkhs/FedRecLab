from typing import List, Dict, Any, Sequence

import numpy as np
import torch

from src.fedreclab.federated.utils import RoundStatistics, ParamIO


def _iter_client_updates(updates: Any) -> list[Any]:
    """
    Normalize server-provided updates into a python list of per-client updates.

    The server currently stacks non-dict updates (torch.stack / np.stack), so
    aggregators must accept either:
      - stacked torch.Tensor / np.ndarray with leading dim = num_clients
      - list[torch.Tensor] / list[np.ndarray]
      - list[dict[str, tensor/ndarray]] (dict-of-tensor groups are not stacked)
    """
    if isinstance(updates, (torch.Tensor, np.ndarray)):
        return [updates[i] for i in range(int(updates.shape[0]))]
    if isinstance(updates, list):
        return updates
    raise TypeError(f"Unsupported updates type: {type(updates)}")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten_update(update: Any) -> np.ndarray:
    """
    Flatten a single client's update into a 1D numpy vector for distance-based rules.
    Supports tensor/ndarray or dict-of-tensor/ndarray.
    """
    if isinstance(update, dict):
        parts = [_to_numpy(update[k]).reshape(-1) for k in sorted(update.keys())]
        if len(parts) == 0:
            return np.array([], dtype=np.float64)
        return np.concatenate(parts, axis=0)
    return _to_numpy(update).reshape(-1)


def _ensure_same_dict_keys(per_client: Sequence[dict]) -> list[str]:
    if len(per_client) == 0:
        raise ValueError("No client updates provided")
    keys0 = set(per_client[0].keys())
    for i, d in enumerate(per_client[1:], start=1):
        if set(d.keys()) != keys0:
            raise ValueError(f"Client update dict keys mismatch at index {i}")
    return sorted(keys0)


class BaseAggregation:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Dict[str, Any],
        updates: List[torch.Tensor] | List[np.ndarray],
        param_io: ParamIO,
    ):
        raise NotImplementedError("Subclasses must implement this method")

    def _to_torch_like(self, arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        """Preserve input dtype/device when returning torch outputs."""
        return torch.as_tensor(arr, dtype=like.dtype, device=like.device)

    def on_round_start(self, previous_round_statistics: RoundStatistics):
        pass

    def on_round_end(self, current_round_statistics: RoundStatistics):
        pass


class FedAvg(BaseAggregation):
    """
    Federated averaging (versatile for FedMF, FedNCF, FedPerGNN, etc.).

    Supports both single-tensor and dict-of-tensor (state-dict) parameter values.
    For dict values, each key's tensors are averaged independently.
    """

    def _average_dense_parameters(self, updates: List[np.ndarray] | np.ndarray, weights: np.ndarray):
        stacked = np.stack(updates, axis=0) if isinstance(updates, list) else np.asarray(updates)
        w = weights.reshape(-1, *([1] * (stacked.ndim - 1)))
        return np.sum(stacked * w, axis=0)

    def _aggregate_with_weights(self, updates, weights: np.ndarray):
        """Average updates using pre-computed weights. Handles both tensor and dict-of-tensor."""
        if isinstance(updates, list) and len(updates) > 0 and isinstance(updates[0], dict):
            result = {}
            keys = _ensure_same_dict_keys(updates)
            for key in keys:
                key_updates = [u[key] for u in updates]
                if isinstance(key_updates[0], torch.Tensor):
                    # preserve dtype/device of the first update to return a torch tensor with the same dtype/device
                    like = key_updates[0]
                    key_updates = [u.detach().cpu().numpy() for u in key_updates]
                    result[key] = self._to_torch_like(
                        self._average_dense_parameters(key_updates, weights),
                        like,
                    )
                else:
                    result[key] = self._average_dense_parameters(key_updates, weights)
            return result

        if isinstance(updates, torch.Tensor):
            like = updates[0]
            updates_np = updates.detach().cpu().numpy()
            return self._to_torch_like(self._average_dense_parameters(updates_np, weights), like)

        if isinstance(updates, np.ndarray):
            return self._average_dense_parameters(updates, weights)

        if isinstance(updates[0], torch.Tensor):
            # preserve dtype/device of the first update to return a torch tensor with the same dtype/device
            like = updates[0]
            updates = [u.detach().cpu().numpy() for u in updates]
            return self._to_torch_like(self._average_dense_parameters(updates, weights), like)
        return self._average_dense_parameters(updates, weights)

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: List[torch.Tensor] | List[np.ndarray],
        param_io: ParamIO,
    ):
        selected_clients = round_info.selected
        num_selected_clients = len(selected_clients)
        weights = np.array(
            [round_info.num_training_examples[c] for c in selected_clients],
            dtype=np.float64,
        )
        if weights.shape != (num_selected_clients,):
            raise ValueError(f"weights must have shape ({num_selected_clients},), got {weights.shape}")
        w_sum = weights.sum()
        if w_sum <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = weights / w_sum

        return self._aggregate_with_weights(updates, weights)


class SimpleAvg(FedAvg):
    def __init__(self, num_clients, **kwargs):
        super().__init__(num_clients, **kwargs)

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: List[torch.Tensor] | List[np.ndarray],
        param_io: ParamIO,
    ):
        per_client = _iter_client_updates(updates)
        n = len(per_client)
        weights = np.ones(n, dtype=np.float64) / n
        return self._aggregate_with_weights(updates, weights)


class FedMedian(BaseAggregation):
    def __init__(self, num_clients: int):
        super().__init__(num_clients)

    def aggregate(self):
        raise NotImplementedError("Not implemented yet")


class FedAtt(BaseAggregation):
    def __init__(self, num_clients: int, stepsize, metric):
        super().__init__(num_clients)
        self.stepsize = stepsize
        self.metric = metric

    def aggregate(self):
        raise NotImplementedError("Not implemented yet")
