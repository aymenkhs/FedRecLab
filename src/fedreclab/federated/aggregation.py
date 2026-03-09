from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any

import numpy as np
import torch

from src.fedreclab.federated.utils import RoundStatistics, ParamIO


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

    def _average_dense_parameters(self, updates: List[np.ndarray], weights: np.ndarray):
        stacked = np.stack(updates, axis=0)
        w = weights.reshape(-1, *([1] * (stacked.ndim - 1)))
        return np.sum(stacked * w, axis=0)

    def _aggregate_with_weights(self, updates, weights: np.ndarray):
        """Average updates using pre-computed weights. Handles both tensor and dict-of-tensor."""
        if isinstance(updates, list) and len(updates) > 0 and isinstance(updates[0], dict):
            result = {}
            for key in updates[0]:
                key_updates = [u[key] for u in updates]
                if isinstance(key_updates[0], torch.Tensor):
                    key_updates = [u.detach().cpu().numpy() for u in key_updates]
                result[key] = self._average_dense_parameters(key_updates, weights)
            return result

        if isinstance(updates[0], torch.Tensor):
            updates = [u.detach().cpu().numpy() for u in updates]
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
        weights = np.ones(len(updates), dtype=np.float64)
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
