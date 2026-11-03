from typing import List, Dict, Any, Sequence
from dataclasses import replace

import numpy as np
import torch

from fedreclab.federated.utils import RoundStatistics, ParamIO


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


def _require_finite(arr: np.ndarray, where: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {where}")


def _masked_weighted_mean(
    values: np.ndarray | dict,
    mask: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | dict:
    """Weighted mean over clients that have a row, renormalised per row.

    ``values`` is ``(n, k, d)`` or a dict of those. ``mask`` is ``(n, k)``.
    ``weights`` is ``(n,)`` unnormalised. NaNs in ``values`` fail loudly;
    the mask is boolean, not encoded as NaN.
    """
    if isinstance(values, dict):
        return {
            key: _masked_weighted_mean(val, mask, weights)
            for key, val in values.items()
        }
    values = np.asarray(values)
    _require_finite(values, "row-aggregation values")
    n, k = mask.shape
    w = np.asarray(weights, dtype=np.float64).reshape(n, 1) * mask.astype(np.float64)
    w_sum = w.sum(axis=0)
    if np.any(w_sum <= 0):
        raise ValueError("A packed row has no contributing client with positive weight")
    w_norm = w / w_sum.reshape(1, k)
    return np.sum(values * w_norm.reshape(n, k, 1), axis=0)


def _masked_sum(values: np.ndarray | dict, mask: np.ndarray) -> np.ndarray | dict:
    """Sum over clients that have a row; non-senders add 0.

    ``values`` is ``(n, k, d)`` or a dict of those. ``mask`` is ``(n, k)``.
    A column with no True entry fails; that is a packing bug, not a
    legitimate empty sum.

    Parameters
    ----------
    values : ndarray or dict of ndarray
        Packed client rows, shape ``(n, k, d)``.
    mask : ndarray of bool, shape (n, k)
        True where that client uploaded the column's row id.

    Returns
    -------
    ndarray or dict of ndarray
        Sum over axis 0, shape ``(k, d)``.

    Raises
    ------
    ValueError
        If any value is non-finite, or a packed row has no contributor.

    Examples
    --------
    >>> _masked_sum(
    ...     np.array([[[1.0]], [[3.0]]]),
    ...     np.array([[True], [True]]),
    ... )
    array([[4.]])
    """
    if isinstance(values, dict):
        return {key: _masked_sum(val, mask) for key, val in values.items()}
    values = np.asarray(values)
    _require_finite(values, "row-aggregation values")
    n, k = mask.shape
    contrib = mask.astype(np.float64).sum(axis=0)
    if np.any(contrib <= 0):
        raise ValueError("A packed row has no contributing client")
    return np.sum(values * mask.astype(np.float64).reshape(n, k, 1), axis=0)


def _masked_nanmedian(values: np.ndarray | dict, mask: np.ndarray) -> np.ndarray | dict:
    """Coordinate-wise median over contributing clients.

    Masked-out entries are NaN only for ``nanmedian``; real NaNs in ``values``
    still fail before that.
    """
    if isinstance(values, dict):
        return {key: _masked_nanmedian(val, mask) for key, val in values.items()}
    values = np.asarray(values)
    _require_finite(values, "row-aggregation values")
    filled = np.where(mask[..., None], values, np.nan)
    return np.nanmedian(filled, axis=0)


def _slice_round_info_for_clients(
    round_info: RoundStatistics,
    client_ids: list[int],
) -> RoundStatistics:
    return replace(
        round_info,
        selected=list(client_ids),
        responded=list(client_ids),
        dropped=[],
        num_training_examples={
            cid: round_info.num_training_examples[cid] for cid in client_ids
        },
        training_losses={
            cid: round_info.training_losses[cid] for cid in client_ids
        },
        training_durations={
            cid: round_info.training_durations[cid] for cid in client_ids
        },
    )


class BaseAggregation:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.row_fallback_count = 0
        self._last_call_fell_back = False

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Dict[str, Any],
        updates: List[torch.Tensor] | List[np.ndarray],
        param_io: ParamIO,
    ):
        raise NotImplementedError("Subclasses must implement this method")

    def aggregate_rows(
        self,
        round_info: RoundStatistics,
        previous_rows: Any,
        values: np.ndarray | dict,
        mask: np.ndarray,
        param_io: ParamIO,
    ):
        """Aggregate packed row updates.

        Default: one ``aggregate()`` call per union row, over the clients
        that sent it. Mean-family aggregators and Median override this with
        a vectorised masked reduce.

        ``values`` / ``mask`` cover the union of rows uploaded this round
        (shape ``n × k_union × d``), not necessarily the full catalogue.

        Parameters
        ----------
        round_info : RoundStatistics
            Statistics aligned with axis 0 of ``values`` (``selected`` order).
        previous_rows : array or dict of arrays
            Previous table rows at the union indices, shape ``(k, d)``.
        values : array or dict of arrays
            Packed client rows, shape ``(n, k, d)``.
        mask : ndarray of bool, shape (n, k)
            True where that client uploaded the column's row id.
        param_io : ParamIO
            Group declaration (unused by the default loop).

        Returns
        -------
        ndarray or dict of ndarray
            Aggregated rows, shape ``(k, d)``.
        """
        self.row_fallback_count = 0
        n, k = mask.shape
        selected = list(round_info.selected)
        if len(selected) != n:
            raise ValueError(
                f"round_info.selected length {len(selected)} != packed n={n}"
            )
        is_dict = isinstance(values, dict)
        keys = list(values.keys()) if is_dict else None
        out_cols: list[Any] = []
        for j in range(k):
            contrib = np.flatnonzero(mask[:, j])
            if contrib.size == 0:
                raise ValueError("Packed union contains a row with no contributors")
            client_ids = [selected[int(i)] for i in contrib]
            row_info = _slice_round_info_for_clients(round_info, client_ids)
            if is_dict:
                updates = [
                    {key: np.asarray(values[key][int(i), j]) for key in keys}
                    for i in contrib
                ]
                prev = {key: np.asarray(previous_rows[key][j]) for key in keys}
            else:
                updates = [np.asarray(values[int(i), j]) for i in contrib]
                prev = np.asarray(previous_rows[j])
            self._last_call_fell_back = False
            aggregated = self.aggregate(row_info, prev, updates, param_io)
            if self._last_call_fell_back:
                self.row_fallback_count += 1
            if is_dict:
                aggregated = {key: _to_numpy(aggregated[key]) for key in keys}
            else:
                aggregated = _to_numpy(aggregated)
            out_cols.append(aggregated)
        if is_dict:
            return {
                key: np.stack([col[key] for col in out_cols], axis=0)
                for key in keys
            }
        return np.stack(out_cols, axis=0)

    def _to_torch_like(self, arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        """Preserve input dtype/device when returning torch outputs."""
        return torch.as_tensor(arr, dtype=like.dtype, device=like.device)

    def on_round_start(self, previous_round_statistics: RoundStatistics):
        pass

    def on_round_end(self, current_round_statistics: RoundStatistics):
        pass


class FedAvg(BaseAggregation):
    """
    Weighted average of client updates, weighted by num_training_examples.
    
    Works with any method: for row-layout groups (ParamIO.layout="rows") the 
    server applies it per row, over the clients that updated that row.
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

    def aggregate_rows(
        self,
        round_info: RoundStatistics,
        previous_rows: Any,
        values: np.ndarray | dict,
        mask: np.ndarray,
        param_io: ParamIO,
    ):
        self.row_fallback_count = 0
        selected = list(round_info.selected)
        n = mask.shape[0]
        if len(selected) != n:
            raise ValueError(
                f"round_info.selected length {len(selected)} != packed n={n}"
            )
        weights = np.array(
            [round_info.num_training_examples[c] for c in selected],
            dtype=np.float64,
        )
        return _masked_weighted_mean(values, mask, weights)


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

    def aggregate_rows(
        self,
        round_info: RoundStatistics,
        previous_rows: Any,
        values: np.ndarray | dict,
        mask: np.ndarray,
        param_io: ParamIO,
    ):
        self.row_fallback_count = 0
        n = mask.shape[0]
        weights = np.ones(n, dtype=np.float64)
        return _masked_weighted_mean(values, mask, weights)


class Sum(BaseAggregation):
    """Unweighted sum of client updates.

    Method-agnostic: sums tensors or dict-of-tensors as given, without
    inspecting ``ParamIO.exchange`` or the calling method. Pairing with
    ``exchange="gradient"`` or ``"delta"`` is the experiment's job.

    Needs only the sum of updates, not the individuals
    (secure-aggregation compatible). ``previous_parameters`` and
    ``num_training_examples`` are ignored.

    For ``ParamIO.layout="rows"``, a client that did not send a row
    contributes 0 to that row. Zero-valued dense uploads therefore match
    omitting those rows.

    Parameters
    ----------
    num_clients : int
        Population size stored on ``BaseAggregation``.

    Examples
    --------
    >>> info = RoundStatistics(
    ...     current_round=0, selected=[0, 1], responded=[0, 1], dropped=[],
    ...     num_training_examples={0: 1, 1: 1},
    ...     training_losses={0: [], 1: []}, training_durations={0: 0.0, 1: 0.0},
    ... )
    >>> pio = ParamIO(residency="server", exchange="gradient", apply_to="server")
    >>> Sum(2).aggregate(info, None, [np.array([1.0]), np.array([3.0])], pio)
    array([4.])
    """

    def _sum_updates(self, updates: Any):
        """Sum one group of client tensors or dict-of-tensors.

        Parameters
        ----------
        updates : list or stacked array
            Per-client values, already accepted by ``_iter_client_updates``.

        Returns
        -------
        ndarray, Tensor, or dict
            Elementwise sum, same layout and type as the first client.

        Raises
        ------
        ValueError
            If there are no updates, dict keys differ, or a value is
            non-finite.
        """
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        if isinstance(per_client[0], dict):
            keys = _ensure_same_dict_keys(per_client)
            out: dict[str, Any] = {}
            for k in keys:
                stacked = np.stack([_to_numpy(u[k]) for u in per_client], axis=0)
                _require_finite(stacked, "Sum client updates")
                summed = stacked.sum(axis=0)
                like = per_client[0][k]
                out[k] = (
                    self._to_torch_like(summed, like)
                    if isinstance(like, torch.Tensor)
                    else summed
                )
            return out

        stacked = np.stack([_to_numpy(u) for u in per_client], axis=0)
        _require_finite(stacked, "Sum client updates")
        summed = stacked.sum(axis=0)
        like0 = per_client[0]
        if isinstance(like0, torch.Tensor):
            return self._to_torch_like(summed, like0)
        return summed

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        """Sum dense client updates.

        Parameters
        ----------
        round_info : RoundStatistics
            Unused; present for the aggregator interface.
        previous_parameters : any
            Unused.
        updates : list or stacked array
            Per-client tensors or dict-of-tensors.
        param_io : ParamIO
            Group declaration (unused).

        Returns
        -------
        ndarray, Tensor, or dict
            Elementwise sum over the client axis.

        Raises
        ------
        ValueError
            If there are no updates or a value is non-finite.
        """
        return self._sum_updates(updates)

    def aggregate_rows(
        self,
        round_info: RoundStatistics,
        previous_rows: Any,
        values: np.ndarray | dict,
        mask: np.ndarray,
        param_io: ParamIO,
    ):
        """Masked sum over clients that uploaded each union row.

        Parameters
        ----------
        round_info : RoundStatistics
            Unused; present for the aggregator interface.
        previous_rows : array or dict of arrays
            Unused.
        values : ndarray or dict of ndarray
            Packed client rows, shape ``(n, k, d)``.
        mask : ndarray of bool, shape (n, k)
            True where that client uploaded the column's row id.
        param_io : ParamIO
            Group declaration (unused).

        Returns
        -------
        ndarray or dict of ndarray
            Summed rows, shape ``(k, d)``.

        Raises
        ------
        ValueError
            If a packed row has no contributor or a value is non-finite.
        """
        self.row_fallback_count = 0
        return _masked_sum(values, mask)


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

    def aggregate_rows(
        self,
        round_info: RoundStatistics,
        previous_rows: Any,
        values: np.ndarray | dict,
        mask: np.ndarray,
        param_io: ParamIO,
    ):
        self.row_fallback_count = 0
        return _masked_nanmedian(values, mask)


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


class FedAtt(BaseAggregation):
    """Layer-wise attentive aggregation (Ji et al., IJCNN 2019).

    For each tensor, compute a softmax over client distances
    ``||prev - update_i||_ord`` (``ord=metric``) and return

    ``prev - stepsize * sum_i att_i * (prev - update_i)``.

    Dict groups score **each key independently**. That matches the official
    ``shaoxiongji/fed-att`` loop over state-dict tensors. The paper calls
    this layer-wise; the code does not glue a module's weight and bias.

    Row-layout groups still call this once per union row, over that row's
    senders (the ``BaseAggregation`` loop, not FedAvg's masked mean). A
    row that is itself a dict (e.g. FedNCF GMF and MLP tables) therefore
    also gets one attention per table.

    Deviations from Ji et al.:
      The optional Gaussian randomisation (paper Eq. 6) is not applied.

    Parameters
    ----------
    num_clients : int
        Population size stored on ``BaseAggregation``.
    stepsize : float, optional
        ``ε`` in the server update. Default is 1.0.
    metric : int, optional
        Order of the vector norm used as the client–server distance.
        Default is 2 (Euclidean on the flattened tensor).

    Examples
    --------
    Two tensors with complementary distances get different attentions::

        FedAtt(2).aggregate(info, {"w": w0, "b": b0}, [u0, u1], pio)
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

    def _fedatt_next(self, prev: np.ndarray, updates: np.ndarray) -> np.ndarray:
        """One FedAtt step on a single tensor.

        Parameters
        ----------
        prev : ndarray
            Previous server tensor, shape ``d``.
        updates : ndarray
            Stacked client tensors, shape ``(n, *d)``.

        Returns
        -------
        ndarray
            Updated tensor, same shape as ``prev``.

        Raises
        ------
        ValueError
            If a client tensor does not match ``prev``, or if any value
            is non-finite.
        """
        prev = np.asarray(prev)
        updates = np.asarray(updates)
        if updates.shape[1:] != prev.shape:
            raise ValueError(
                f"FedAtt: update shape {updates.shape} is incompatible with "
                f"previous shape {prev.shape}"
            )
        _require_finite(prev, "FedAtt previous parameters")
        _require_finite(updates, "FedAtt client updates")
        n = int(updates.shape[0])
        diffs_flat = prev.reshape(1, -1) - updates.reshape(n, -1)
        dists = np.linalg.norm(diffs_flat, ord=self.metric, axis=1)
        att = self._softmax_np(dists)
        direction = np.sum(
            (prev[None, ...] - updates) * att.reshape(n, *([1] * prev.ndim)),
            axis=0,
        )
        return prev - self.stepsize * direction

    def _cast_fedatt(self, next_np: np.ndarray, like: Any):
        """Return ``next_np`` as a torch tensor when ``like`` is one.

        Parameters
        ----------
        next_np : ndarray
            FedAtt result in NumPy.
        like : array or Tensor
            Previous value, used only for type/dtype/device.

        Returns
        -------
        ndarray or Tensor
            ``next_np``, or a tensor matching ``like``.
        """
        if isinstance(like, torch.Tensor):
            return self._to_torch_like(next_np, like)
        return next_np

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        """Aggregate one group (or one row) with per-tensor attention.

        Parameters
        ----------
        round_info : RoundStatistics
            Unused; present for the aggregator interface.
        previous_parameters : array or dict of arrays
            Server tensors before this round.
        updates : list or stacked array
            Per-client tensors, same layout as ``previous_parameters``.
        param_io : ParamIO
            Group declaration (unused).

        Returns
        -------
        array or dict of arrays
            Updated tensors, same type/layout as ``previous_parameters``.

        Raises
        ------
        ValueError
            If there are no updates, a dict key is missing, or a tensor
            shape does not match the previous parameters.
        TypeError
            If updates are a dict but ``previous_parameters`` is not.
        """
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        if isinstance(per_client[0], dict):
            keys = _ensure_same_dict_keys(per_client)
            if not isinstance(previous_parameters, dict):
                raise TypeError("FedAtt: expected previous_parameters to be dict for dict updates")
            missing = [k for k in keys if k not in previous_parameters]
            if missing:
                raise ValueError(
                    f"FedAtt: previous_parameters missing keys {missing}"
                )
            out: dict[str, Any] = {}
            for k in keys:
                prev_k = _to_numpy(previous_parameters[k])
                upd_k = np.stack([_to_numpy(u[k]) for u in per_client], axis=0)
                next_k = self._fedatt_next(prev_k, upd_k)
                out[k] = self._cast_fedatt(next_k, previous_parameters[k])
            return out

        prev_np = _to_numpy(previous_parameters)
        upd_np = np.stack([_to_numpy(u) for u in per_client], axis=0)
        next_np = self._fedatt_next(prev_np, upd_np)
        return self._cast_fedatt(next_np, previous_parameters)


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
            # Too few candidates for the theoretical Krum guarantee.
            # Instead of raising, fall back to equal scores so callers
            # can gracefully reduce to simple averaging or pick any.
            return np.zeros((n,), dtype=np.float64)
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
        self._last_call_fell_back = False
        if n < 2 * f + 3:
            # Too few clients for Krum's theoretical guarantee; fall back to simple average.
            self._last_call_fell_back = True
            weights = np.ones(n, dtype=np.float64) / float(n)
            return FedAvg(self.num_clients)._aggregate_with_weights(per_client, weights)

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
        self._last_call_fell_back = False
        if n < 4 * f + 3:
            # Too few clients for Bulyan; fall back to simple average.
            self._last_call_fell_back = True
            weights = np.ones(n, dtype=np.float64) / float(n)
            return FedAvg(self.num_clients)._aggregate_with_weights(per_client, weights)

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


class ChangeMagnitudeAvg(BaseAggregation):
    """Weighted average by L2 change from the previous parameters.

    For each client update ``u_i`` and previous value ``prev``, the weight
    is ``q_i = ||u_i - prev||_2``.     Clients with ``q_i <= change_eps`` are
    dropped; the remaining weights are renormalised and used for a
    weighted mean of the absolute updates (not of the deltas). If every
    client is dropped, the previous value is kept: the weighted mean of
    an empty set is undefined, and every uploaded value is within
    ``change_eps`` of that previous value. On the row path each such
    packed row increments ``row_fallback_count``.

    This is the item-embedding rule from FedFast ActvAGG (Muhammad et al.,
    KDD 2020, Algorithm 3 lines 5–12), exposed as a method-agnostic
    aggregator. It is not FedAvg: weights come from change magnitude, not
    ``num_training_examples``. Dict-of-tensor groups are scored **per
    key** independently.

    Needs individual updates and ``previous_parameters`` (not compatible
    with sum-only secure aggregation).

    Parameters
    ----------
    num_clients : int
        Population size stored on ``BaseAggregation``.
    change_eps : float, optional
        Clients with L2 change at most this value are ignored. Default
        is ``0.0`` (paper: strictly positive change).

    Raises
    ------
    ValueError
        If ``change_eps`` is negative, a value is non-finite, or shapes
        do not match ``previous``.

    Examples
    --------
    >>> info = RoundStatistics(
    ...     current_round=0, selected=[0, 1], responded=[0, 1], dropped=[],
    ...     num_training_examples={0: 1, 1: 1},
    ...     training_losses={0: [], 1: []}, training_durations={0: 0.0, 1: 0.0},
    ... )
    >>> pio = ParamIO(residency="server", exchange="params", apply_to="server")
    >>> prev = np.array([0.0, 0.0])
    >>> updates = [np.array([2.0, 0.0]), np.array([0.0, 1.0])]
    >>> out = ChangeMagnitudeAvg(2).aggregate(info, prev, updates, pio)
    >>> np.allclose(out, [4.0 / 3.0, 1.0 / 3.0])
    True
    """

    def __init__(self, num_clients: int, change_eps: float = 0.0):
        super().__init__(num_clients)
        if change_eps < 0:
            raise ValueError(f"change_eps must be >= 0, got {change_eps}")
        self.change_eps = float(change_eps)
        self._weigher = FedAvg(num_clients)

    @staticmethod
    def _l2_change(update: np.ndarray, previous: np.ndarray) -> float:
        diff = np.asarray(update, dtype=np.float64) - np.asarray(
            previous, dtype=np.float64
        )
        _require_finite(diff, "ChangeMagnitudeAvg update-previous difference")
        return float(np.linalg.norm(diff.reshape(-1)))

    def _weights_for_updates(
        self,
        previous_parameters: Any,
        per_client: list[Any],
    ) -> np.ndarray | None:
        """Return renormalised change-magnitude weights for one tensor.

        Parameters
        ----------
        previous_parameters : array
            Value before this round.
        per_client : list of array
            One update per client. Must not be dicts.

        Returns
        -------
        ndarray or None
            Weights summing to 1 over clients with ``q_i > change_eps``.
            ``None`` when every client is at or below ``change_eps``; the
            caller keeps ``previous_parameters``.

        Raises
        ------
        TypeError
            If ``per_client`` holds dicts.
        ValueError
            If a difference is non-finite, or the kept weights do not sum
            to a positive value.
        """
        if isinstance(per_client[0], dict):
            raise TypeError(
                "ChangeMagnitudeAvg._weights_for_updates expects non-dict updates"
            )
        prev_np = _to_numpy(previous_parameters)
        qs = np.array(
            [self._l2_change(_to_numpy(u), prev_np) for u in per_client],
            dtype=np.float64,
        )
        keep = qs > self.change_eps
        if not np.any(keep):
            return None
        weights = np.where(keep, qs, 0.0)
        w_sum = float(weights.sum())
        if w_sum <= 0:
            raise ValueError("ChangeMagnitudeAvg weights must sum to a positive value")
        return weights / w_sum

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Any,
        updates: Any,
        param_io: ParamIO,
    ):
        """Average dense updates weighted by L2 change from ``previous``.

        Parameters
        ----------
        round_info : RoundStatistics
            Unused for weights; present for the aggregator interface.
        previous_parameters : array or dict of arrays
            Server (or prior) tensors before this round.
        updates : list or stacked array
            Per-client tensors, same layout as ``previous_parameters``.
        param_io : ParamIO
            Group declaration (unused).

        Returns
        -------
        array or dict of arrays
            Weighted mean of contributing updates. A tensor, or one dict
            key, whose clients are all at or below ``change_eps`` is
            returned as the corresponding previous value.

        Raises
        ------
        ValueError
            If there are no updates, or a value is non-finite.
        TypeError
            If updates are a dict but ``previous_parameters`` is not.
        """
        del round_info, param_io
        per_client = _iter_client_updates(updates)
        if len(per_client) == 0:
            raise ValueError("No client updates provided")

        if isinstance(per_client[0], dict):
            keys = _ensure_same_dict_keys(per_client)
            if not isinstance(previous_parameters, dict):
                raise TypeError(
                    "ChangeMagnitudeAvg: expected previous_parameters to be "
                    "dict for dict updates"
                )
            missing = [k for k in keys if k not in previous_parameters]
            if missing:
                raise ValueError(
                    f"ChangeMagnitudeAvg: previous_parameters missing keys {missing}"
                )
            out: dict[str, Any] = {}
            for key in keys:
                key_updates = [u[key] for u in per_client]
                weights = self._weights_for_updates(
                    previous_parameters[key], key_updates
                )
                if weights is None:
                    out[key] = previous_parameters[key]
                else:
                    out[key] = self._weigher._aggregate_with_weights(
                        key_updates, weights
                    )
            return out

        weights = self._weights_for_updates(previous_parameters, per_client)
        if weights is None:
            return previous_parameters
        return self._weigher._aggregate_with_weights(per_client, weights)

    def aggregate_rows(
        self,
        round_info: RoundStatistics,
        previous_rows: Any,
        values: np.ndarray | dict,
        mask: np.ndarray,
        param_io: ParamIO,
    ):
        """Per-row change-magnitude weighted mean over contributing clients.

        Parameters
        ----------
        round_info : RoundStatistics
            Unused for weights.
        previous_rows : array or dict of arrays
            Previous table rows at the union indices, shape ``(k, d)``.
        values : ndarray or dict of ndarray
            Packed client rows, shape ``(n, k, d)``.
        mask : ndarray of bool, shape (n, k)
            True where that client uploaded the column's row id.
        param_io : ParamIO
            Group declaration (unused).

        Returns
        -------
        ndarray or dict of ndarray
            Aggregated rows, shape ``(k, d)``. A packed row whose
            contributors are all at or below ``change_eps`` is copied from
            ``previous_rows`` and counted in ``row_fallback_count``.

        Raises
        ------
        ValueError
            If a packed row has no contributors, or a value is non-finite.
        """
        del round_info, param_io
        self.row_fallback_count = 0
        if isinstance(values, dict):
            keys = list(values.keys())
            return {
                key: self._aggregate_rows_array(
                    previous_rows[key], values[key], mask
                )
                for key in keys
            }
        return self._aggregate_rows_array(previous_rows, values, mask)

    def _aggregate_rows_array(
        self,
        previous_rows: Any,
        values: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        prev = np.asarray(previous_rows)
        vals = np.asarray(values)
        _require_finite(vals, "ChangeMagnitudeAvg row-aggregation values")
        _require_finite(prev, "ChangeMagnitudeAvg previous rows")
        n, k = mask.shape
        out = np.empty_like(prev, dtype=np.float64)
        for j in range(k):
            contrib = np.flatnonzero(mask[:, j])
            if contrib.size == 0:
                raise ValueError("Packed union contains a row with no contributors")
            qs = np.zeros(n, dtype=np.float64)
            for i in contrib:
                qs[i] = self._l2_change(vals[i, j], prev[j])
            keep = qs > self.change_eps
            if not np.any(keep):
                out[j] = prev[j]
                self.row_fallback_count += 1
                continue
            w = np.where(keep, qs, 0.0)
            w = w / w.sum()
            # Broadcast weights over contributing clients only via full n vector.
            out[j] = np.sum(vals * w.reshape(n, 1, 1), axis=0)[j]
        if isinstance(previous_rows, torch.Tensor):
            return self._to_torch_like(out, previous_rows)
        return out.astype(prev.dtype, copy=False)
