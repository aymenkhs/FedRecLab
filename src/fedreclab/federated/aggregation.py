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


class Median(BaseAggregation):
    """Coordinate-wise median aggregation."""

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        if isinstance(per_client[0], dict):
            keys = _ensure_same_dict_keys(per_client)
            out: dict[str, Any] = {}
            for k in keys:
                vals = [_to_numpy(u[k]) for u in per_client]
                med = np.median(np.stack(vals, axis=0), axis=0)
                like = per_client[0][k]
                out[k] = self._to_torch_like(med, like) if isinstance(like, torch.Tensor) else med
            return out

        vals = [_to_numpy(u) for u in per_client]
        med = np.median(np.stack(vals, axis=0), axis=0)
        like0 = per_client[0]
        return self._to_torch_like(med, like0) if isinstance(like0, torch.Tensor) else med


class TrimmedMean(BaseAggregation):
    """Coordinate-wise trimmed mean aggregation."""

    def __init__(self, num_clients: int, beta: float = 0.1):
        super().__init__(num_clients)
        if not (0 <= beta < 0.5):
            raise ValueError(f"beta must satisfy 0 <= beta < 0.5, got {beta}")
        self.beta = float(beta)

    def _trimmed_mean_np(self, stacked: np.ndarray) -> np.ndarray:
        n = stacked.shape[0]
        trim_k = int(self.beta * n)
        if 2 * trim_k >= n:
            raise ValueError(
                f"Too much trimming: n={n}, beta={self.beta}, trim_k={trim_k}"
            )
        sorted_vals = np.sort(stacked, axis=0)
        kept = sorted_vals[trim_k : n - trim_k]
        return kept.mean(axis=0)

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        if isinstance(per_client[0], dict):
            keys = _ensure_same_dict_keys(per_client)
            out: dict[str, Any] = {}
            for k in keys:
                vals = np.stack([_to_numpy(u[k]) for u in per_client], axis=0)
                tm = self._trimmed_mean_np(vals)
                like = per_client[0][k]
                out[k] = self._to_torch_like(tm, like) if isinstance(like, torch.Tensor) else tm
            return out

        vals = np.stack([_to_numpy(u) for u in per_client], axis=0)
        tm = self._trimmed_mean_np(vals)
        like0 = per_client[0]
        return self._to_torch_like(tm, like0) if isinstance(like0, torch.Tensor) else tm


class FedAtt(FedAvg):
    """
    Attention-style aggregation.

    Computes a softmax over distances ||prev - update_i||_ord (ord=metric),
    then returns: prev - stepsize * sum_i att_i * (prev - update_i).
    """

    def __init__(self, num_clients: int, stepsize: float = 1.0, metric: int = 2):
        super().__init__(num_clients)
        self.stepsize = float(stepsize)
        self.metric = metric

    @staticmethod
    def _softmax_np(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64, copy=False)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        # Use a single attention over the whole parameter group (flattened),
        # then apply the resulting weights to each parameter tensor/value.
        prev_flat = _flatten_update(previous_parameters)
        upd_flat = np.stack([_flatten_update(u) for u in per_client], axis=0)
        if upd_flat.shape[1] != prev_flat.shape[0]:
            raise ValueError("FedAtt: previous_parameters and updates shapes are incompatible")

        diffs = prev_flat[None, :] - upd_flat
        dists = np.linalg.norm(diffs, ord=self.metric, axis=1)
        att = self._softmax_np(dists)

        if isinstance(per_client[0], dict):
            keys = _ensure_same_dict_keys(per_client)
            if not isinstance(previous_parameters, dict):
                raise TypeError("FedAtt: expected previous_parameters to be dict for dict updates")
            out: dict[str, Any] = {}
            for k in keys:
                prev_k = _to_numpy(previous_parameters[k])
                upd_k = np.stack([_to_numpy(u[k]) for u in per_client], axis=0)
                direction = np.sum((prev_k[None, :] - upd_k) * att.reshape(-1, *([1] * (upd_k.ndim - 1))), axis=0)
                next_k = prev_k - self.stepsize * direction
                like = previous_parameters[k]
                out[k] = self._to_torch_like(next_k, like) if isinstance(like, torch.Tensor) else next_k
            return out

        prev_np = _to_numpy(previous_parameters)
        upd_np = np.stack([_to_numpy(u) for u in per_client], axis=0)
        direction = np.sum((prev_np[None, :] - upd_np) * att.reshape(-1, *([1] * (upd_np.ndim - 1))), axis=0)
        next_np = prev_np - self.stepsize * direction
        like0 = previous_parameters
        return self._to_torch_like(next_np, like0) if isinstance(like0, torch.Tensor) else next_np


class Krum(BaseAggregation):
    """
    Krum / Multi-Krum robust aggregation.

    - Requires n >= 2f + 3 (raises otherwise).
    - Single-Krum selects 1 client update.
    - Multi-Krum selects m updates iteratively then averages them (SimpleAvg).
    """

    def __init__(
        self,
        num_clients: int,
        num_byzantine: int = 0,
        multi_krum: bool = False,
        num_selected: int | None = None,
    ):
        super().__init__(num_clients)
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be >= 0")
        self.num_byzantine = int(num_byzantine)
        self.multi_krum = bool(multi_krum)
        self.num_selected = num_selected

    @staticmethod
    def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
        # X: (n, p)
        norms = np.sum(X * X, axis=1, keepdims=True)
        return norms + norms.T - 2.0 * (X @ X.T)

    def _krum_scores(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        f = self.num_byzantine
        if n < 2 * f + 3:
            raise ValueError(f"Krum requires n >= 2f + 3, got n={n}, f={f}")
        D = self._pairwise_sq_dists(X)
        num_closest = n - f - 2
        scores = np.empty((n,), dtype=np.float64)
        for i in range(n):
            d = np.concatenate([D[i, :i], D[i, i + 1 :]], axis=0)
            closest = np.partition(d, num_closest - 1)[:num_closest]
            scores[i] = float(np.sum(closest))
        return scores

    def _multi_krum_select(self, X: np.ndarray, m: int) -> list[int]:
        n = X.shape[0]
        if m <= 0 or m > n:
            raise ValueError(f"num_selected must be in [1, n], got m={m}, n={n}")
        selected: list[int] = []
        available = list(range(n))
        for _ in range(m):
            X_cur = X[available]
            scores = self._krum_scores(X_cur)
            winner_local = int(np.argmin(scores))
            winner_global = available[winner_local]
            selected.append(winner_global)
            available.remove(winner_global)
        return selected

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        X = np.stack([_flatten_update(u).astype(np.float64, copy=False) for u in per_client], axis=0)
        n = X.shape[0]
        f = self.num_byzantine
        if n < 2 * f + 3:
            raise ValueError(f"Krum requires n >= 2f + 3, got n={n}, f={f}")

        if not self.multi_krum:
            scores = self._krum_scores(X)
            winner = int(np.argmin(scores))
            return per_client[winner]

        m = self.num_selected
        if m is None:
            m = n - f - 2
        chosen = self._multi_krum_select(X, int(m))
        chosen_updates = [per_client[i] for i in chosen]
        # average chosen candidates uniformly
        weights = np.ones(len(chosen_updates), dtype=np.float64) / len(chosen_updates)
        return FedAvg(self.num_clients)._aggregate_with_weights(chosen_updates, weights)


class Bulyan(BaseAggregation):
    """
    Bulyan robust aggregation (Krum-based candidate selection + coordinate-wise rule).

    Requires n >= 4f + 3 (raises otherwise).
    """

    def __init__(self, num_clients: int, num_byzantine: int = 0):
        super().__init__(num_clients)
        if num_byzantine < 0:
            raise ValueError("num_byzantine must be >= 0")
        self.num_byzantine = int(num_byzantine)

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        n = len(per_client)
        f = self.num_byzantine
        if n < 4 * f + 3:
            raise ValueError(f"Bulyan requires n >= 4f + 3, got n={n}, f={f}")

        # Candidate selection (theta = n - 2f) using Multi-Krum selection.
        theta = n - 2 * f
        beta = n - 4 * f

        X = np.stack([_flatten_update(u).astype(np.float64, copy=False) for u in per_client], axis=0)
        krum = Krum(num_clients=self.num_clients, num_byzantine=f, multi_krum=True, num_selected=theta)
        chosen = krum._multi_krum_select(X, theta)
        candidates = [per_client[i] for i in chosen]

        def bulyan_on_array(stacked: np.ndarray) -> np.ndarray:
            # stacked: (theta, ...)
            coord_median = np.median(stacked, axis=0)
            deviations = np.abs(stacked - coord_median)
            # keep beta closest to median per coordinate
            idx = np.argpartition(deviations, kth=beta - 1, axis=0)[:beta]
            gathered = np.take_along_axis(stacked, idx, axis=0)
            return gathered.mean(axis=0)

        if isinstance(candidates[0], dict):
            keys = _ensure_same_dict_keys(candidates)
            out: dict[str, Any] = {}
            for k in keys:
                stacked = np.stack([_to_numpy(u[k]) for u in candidates], axis=0)
                agg = bulyan_on_array(stacked)
                like = candidates[0][k]
                out[k] = self._to_torch_like(agg, like) if isinstance(like, torch.Tensor) else agg
            return out

        stacked = np.stack([_to_numpy(u) for u in candidates], axis=0)
        agg = bulyan_on_array(stacked)
        like0 = candidates[0]
        return self._to_torch_like(agg, like0) if isinstance(like0, torch.Tensor) else agg
