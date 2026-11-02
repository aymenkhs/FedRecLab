"""Shared FCIS server base and helpers for EFVAE-family methods.

Source: Zhang et al., CIKM 2024. Federated Collaborative Importance
Sampling (FCIS) is shared by the paper's MultVAE, MultDAE and RecVAE
variants. This module holds that shared protocol:

* ParamIO table (sparse encoder/decoder item rows + dense middle)
* FIMS sampler rebuild each round
* custom Kingma–Ba Adam, including row groups (the base row path would
  apply SGD-style exchange before ``apply_to``)
* full-model download simulation (authors' reference also ships the full
  ``state_dict`` and only accounts sparse bytes)
* sampled-softmax CE used by all three backbones

Concrete MultVAE lives in ``efvae.py`` / ``efvae_client.py``. Intended
later subclasses: MultDAE (CE only, no anneal) and RecVAE (different
model and optional enc/dec schedule). Override points:

* ``_initialize_server_parameters`` / model builder
* client ``compute_fcis_loss`` (objective after FCIS logits)
* anneal / extra ``extra_payload`` fields
* local train schedule (RecVAE)

Simulation shortcuts
--------------------
Full parameter tables are downloaded every round. Secret sharing / MPC
is omitted (plaintext gradients; training metrics match cancelled
shares). Communication-cost accounting is not implemented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.efvae.efvae_sampler import FIMSUniSampler

ROW_GROUPS = ("enc_embeddings", "dec_embeddings", "dec_bias")
DENSE_GROUPS = ("middle",)


def default_param_ios() -> dict[str, ParamIO]:
    """Return the FCIS ParamIO map shared by EFVAE-family methods.

    Returns
    -------
    dict of str to ParamIO
        Row-layout embedding tables and a dense ``middle`` group, all
        server-resident gradients applied by custom Adam.

    Examples
    --------
    >>> pio = default_param_ios()
    >>> pio["enc_embeddings"].layout, pio["middle"].layout
    ('rows', 'dense')
    """
    row = ParamIO(
        residency="server",
        exchange="gradient",
        apply_to="custom",
        layout="rows",
    )
    dense = ParamIO(
        residency="server",
        exchange="gradient",
        apply_to="custom",
        layout="dense",
    )
    return {
        "enc_embeddings": ParamIO(
            residency=row.residency,
            exchange=row.exchange,
            apply_to=row.apply_to,
            layout=row.layout,
        ),
        "dec_embeddings": ParamIO(
            residency=row.residency,
            exchange=row.exchange,
            apply_to=row.apply_to,
            layout=row.layout,
        ),
        "dec_bias": ParamIO(
            residency=row.residency,
            exchange=row.exchange,
            apply_to=row.apply_to,
            layout=row.layout,
        ),
        "middle": dense,
    }


def sampled_softmax_ce(
    neg_logit: torch.Tensor,
    neg_log_prob: torch.Tensor,
    pos_logit: torch.Tensor,
    pos_log_prob: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Importance-sampled softmax cross-entropy (paper / reference).

    Parameters
    ----------
    neg_logit, pos_logit : torch.Tensor
        Decoder scores for negatives / positives.
    neg_log_prob, pos_log_prob : torch.Tensor
        Proposal scores from the FCIS sampler (detached inside).
    pos_mask : torch.Tensor or None, optional
        Broadcastable mask over positive positions (1 = keep). ``None``
        keeps every positive column.

    Returns
    -------
    torch.Tensor
        Scalar mean loss over the batch.

    Raises
    ------
    ValueError
        If the loss is non-finite.

    Examples
    --------
    >>> pos = torch.zeros(2, 1)
    >>> neg = torch.zeros(2, 3)
    >>> float(sampled_softmax_ce(neg, neg, pos, pos))
    1.386...
    """
    new_pos = pos_logit - pos_log_prob.detach()
    new_neg = neg_logit - neg_log_prob.detach()
    neg_log_sum_exp = torch.logsumexp(new_neg, dim=-1, keepdim=True)
    final = torch.logaddexp(new_pos, neg_log_sum_exp)
    per_pos = -new_pos + final
    if pos_mask is not None:
        mask = pos_mask.to(dtype=per_pos.dtype)
        per_user = (per_pos * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    else:
        per_user = per_pos.sum(dim=-1)
    loss = per_user.mean()
    if not torch.isfinite(loss):
        raise ValueError("Non-finite sampled-softmax CE")
    return loss


def build_hybrid_items(
    train_items: np.ndarray,
    num_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return train items union ``|train|`` fake items (``rho=1``).

    Parameters
    ----------
    train_items : ndarray
        Positive item ids for one user.
    num_items : int
        Catalogue size.
    rng : numpy.random.Generator
        Seeded generator for fake draws.

    Returns
    -------
    ndarray
        Sorted unique hybrid item ids, dtype int64. When fewer than
        ``|train|`` non-train items exist, every remaining item is used.
    """
    train = np.unique(np.asarray(train_items, dtype=np.int64).reshape(-1))
    if train.size == 0:
        return train
    candidates = np.setdiff1d(
        np.arange(num_items, dtype=np.int64), train, assume_unique=False
    )
    # Paper uses rho=1. If the catalogue is too small to draw that many
    # non-train items (common in unit tests), take every remaining item.
    n_fake = min(int(train.size), int(candidates.size))
    if n_fake == 0:
        return train
    fakes = rng.choice(candidates, size=n_fake, replace=False)
    return np.union1d(train, fakes.astype(np.int64))


def sparse_row_upload(
    table_grad: torch.Tensor,
    row_ids: np.ndarray,
) -> dict[str, Any]:
    """Build a ``{rows, values}`` upload for one embedding table.

    Only rows in ``row_ids`` whose gradient is non-zero (any component)
    are included, matching the reference aggregation mask.

    Parameters
    ----------
    table_grad : torch.Tensor
        Full gradient table, shape ``(num_items, d)``.
    row_ids : ndarray
        Candidate row indices (hybrid ∪ sampled).

    Returns
    -------
    dict
        ``rows`` (int64) and ``values`` (float32) for ParamIO layout
        ``rows``.
    """
    rows = np.unique(np.asarray(row_ids, dtype=np.int64).reshape(-1))
    if rows.size == 0:
        return {
            "rows": np.array([], dtype=np.int64),
            "values": np.zeros((0, int(table_grad.shape[1])), dtype=np.float32),
        }
    idx = torch.as_tensor(rows, dtype=torch.long, device=table_grad.device)
    values = table_grad.index_select(0, idx)
    nonzero = values.detach().abs().sum(dim=-1) > 0
    if not bool(nonzero.any()):
        return {
            "rows": np.array([], dtype=np.int64),
            "values": np.zeros((0, int(table_grad.shape[1])), dtype=np.float32),
        }
    keep = nonzero.cpu().numpy().astype(bool)
    rows_keep = rows[keep]
    vals = values[nonzero].detach().cpu().numpy().astype(np.float32)
    return {"rows": rows_keep, "values": vals}


def sparse_row_upload_named(
    grad_table: torch.Tensor,
    param_key: str,
    row_ids: np.ndarray,
) -> dict[str, Any]:
    """Row upload whose ``values`` are a single-key dict for FedMF-style groups.

    Parameters
    ----------
    grad_table : torch.Tensor
        Gradient of shape ``(num_items, d)``.
    param_key : str
        Inner ParamIO key, e.g. ``\"enc_embeddings.weight\"``.
    row_ids : ndarray
        Candidate rows.

    Returns
    -------
    dict
        ``{\"rows\": ..., \"values\": {param_key: array}}``.
    """
    packed = sparse_row_upload(grad_table, row_ids)
    return {
        "rows": packed["rows"],
        "values": {param_key: packed["values"]},
    }


class EFVAEFCISServer(BaseFederatedServer):
    """FCIS federated server: sampler, ParamIO, custom Adam (incl. rows).

    Subclasses must implement ``_initialize_server_parameters`` and may
    extend ``_on_round_start_server`` for anneal or other extras.

    Parameters
    ----------
    lr : float, optional
        Adam step size. Default is ``1e-3``.
    betas : tuple of float, optional
        Adam ``(beta1, beta2)``. Default is ``(0.9, 0.999)``.
    eps : float, optional
        Adam ``eps``. Default is ``1e-8``.
    num_cluster : int, optional
        FIMS clusters per subspace. Default is 16.
    num_sample : int, optional
        Negatives per user query. Default is 800.
    **kwargs
        Forwarded to ``BaseFederatedServer``.

    Raises
    ------
    ValueError
        If Adam hyper-parameters are out of range.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        num_cluster: int = 16,
        num_sample: int = 800,
        **kwargs,
    ):
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas!r}")
        if float(lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if int(num_cluster) < 1:
            raise ValueError(f"num_cluster must be >= 1, got {num_cluster}")
        if int(num_sample) < 1:
            raise ValueError(f"num_sample must be >= 1, got {num_sample}")
        self.lr = float(lr)
        self.betas = (beta1, beta2)
        self.eps = float(eps)
        self.num_cluster = int(num_cluster)
        self.num_sample = int(num_sample)
        self.adam_m: dict[str, dict[str, torch.Tensor]] = {}
        self.adam_v: dict[str, dict[str, torch.Tensor]] = {}
        self.adam_step = 0
        self._adam_step_round: int | None = None
        self.sampler: FIMSUniSampler | None = None
        super().__init__(param_ios=default_param_ios(), **kwargs)

    def _bump_adam_step_once_per_round(self) -> tuple[float, float, float]:
        """Increment Adam ``t`` once per federated round; return ``(t, bc1, bc2)``.

        All ParamIO groups share one Adam step index in a round, matching
        a single ``optimizer.step()`` over the whole model.

        Returns
        -------
        t : float
            Current Adam step (1-based).
        bc1, bc2 : float
            Bias-correction denominators.
        """
        if self._adam_step_round != self.current_round:
            self.adam_step += 1
            self._adam_step_round = int(self.current_round)
        t = float(self.adam_step)
        beta1, beta2 = self.betas
        return t, 1.0 - beta1 ** t, 1.0 - beta2 ** t

    def _zero_adam_moments(self, groups: dict[str, dict[str, torch.Tensor]]) -> None:
        """Allocate Adam first/second moments matching ``groups``.

        Parameters
        ----------
        groups : dict
            Server state groups from the model.

        Returns
        -------
        None
        """
        self.adam_m = {
            gname: {name: torch.zeros_like(t) for name, t in group.items()}
            for gname, group in groups.items()
        }
        self.adam_v = {
            gname: {name: torch.zeros_like(t) for name, t in group.items()}
            for gname, group in groups.items()
        }
        self.adam_step = 0
        self._adam_step_round = None

    def _on_round_start_server(self) -> None:
        """Rebuild FIMS from ``dec_embeddings`` and publish it on ``extra_payload``.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the decoder table is missing.
        """
        dec = self.global_parameters.get("dec_embeddings")
        if not isinstance(dec, dict) or "dec_embeddings.weight" not in dec:
            raise RuntimeError(
                "FCIS requires global_parameters['dec_embeddings']['dec_embeddings.weight']"
            )
        table = dec["dec_embeddings.weight"]
        num_items = int(table.shape[0])
        generator = None
        if self.seed is not None:
            round_idx = 0 if self.current_round is None else int(self.current_round)
            generator = torch.Generator()
            generator.manual_seed(int(self.seed) + 17 * round_idx + 1)
        if self.sampler is None:
            self.sampler = FIMSUniSampler(
                num_items=num_items,
                num_cluster=self.num_cluster,
                num_neg=self.num_sample,
                generator=generator,
            )
        else:
            self.sampler.generator = generator
        self.sampler.update(torch.as_tensor(table), generator=generator)
        if self.extra_payload is None:
            self.extra_payload = {}
        self.extra_payload["fcis_sampler_state"] = self.sampler.state_dict()

    def _aggregate_group(
        self,
        param_name: str,
        param_io: ParamIO,
        updates: list[Any],
        participant_ids: list[int],
    ) -> None:
        """Aggregate one group; row groups feed averaged grads into Adam.

        Parameters
        ----------
        param_name : str
            ParamIO group name.
        param_io : ParamIO
            Declaration for ``param_name``.
        updates : list
            Per-participant uploads.
        participant_ids : list of int
            Clients aligned with ``updates``.

        Returns
        -------
        None
        """
        if param_io.layout != "rows":
            super()._aggregate_group(
                param_name, param_io, updates, participant_ids
            )
            return

        self._assert_updates_match_layout(param_name, param_io, updates)
        previous = self.global_parameters[param_name]
        packed = self._pack_row_updates(param_name, previous, updates)
        if packed is None:
            return
        union_rows, values, mask, previous_rows = packed
        aggregator = self.aggregation_strategies[param_name]
        agg_rows = aggregator.aggregate_rows(
            self._round_statistics_for_participants(participant_ids),
            previous_rows,
            values,
            mask,
            param_io,
        )
        fallbacks = int(getattr(aggregator, "row_fallback_count", 0) or 0)
        if fallbacks and self.current_round_statistics is not None:
            self.current_round_statistics.row_aggregator_fallbacks[param_name] = (
                fallbacks
            )
        self._adam_update_row_group(param_name, union_rows, agg_rows)

    def _apply_aggregated_result_custom(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: list[int],
    ) -> None:
        """Adam-update a dense group from the aggregated gradient.

        Parameters
        ----------
        param_name : str
            Must be a dense FCIS group (``middle``).
        aggregated_result : dict
            Mean gradient keyed like ``global_parameters[param_name]``.
        param_io : ParamIO
            Unused; present for the hook signature.
        selected_client_ids : list of int
            Unused.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If keys mismatch or a value is non-finite.
        """
        del param_io, selected_client_ids
        if param_name not in DENSE_GROUPS:
            raise ValueError(
                f"FCIS dense custom Adam is defined for {DENSE_GROUPS}, "
                f"got {param_name!r}"
            )
        self._adam_update_dense_group(param_name, aggregated_result)

    def _adam_update_dense_group(
        self,
        param_name: str,
        aggregated_result: dict[str, Any],
    ) -> None:
        """Apply one Adam step to every tensor in a dense group."""
        old = self.global_parameters[param_name]
        if set(old) != set(aggregated_result):
            raise ValueError(
                f"Aggregated keys {sorted(aggregated_result)} do not match "
                f"parameter keys {sorted(old)}"
            )
        _, bc1, bc2 = self._bump_adam_step_once_per_round()
        updated: dict[str, torch.Tensor] = {}
        for name, old_value in old.items():
            like = old_value if torch.is_tensor(old_value) else torch.as_tensor(old_value)
            g = torch.as_tensor(
                aggregated_result[name], dtype=like.dtype, device=like.device
            )
            if not torch.isfinite(g).all():
                raise ValueError(f"Non-finite aggregated gradient for {param_name}.{name}")
            new_value = self._adam_step_tensor(param_name, name, like, g, bc1, bc2)
            updated[name] = new_value
        self.global_parameters[param_name] = updated

    def _adam_update_row_group(
        self,
        param_name: str,
        union_rows: np.ndarray,
        agg_rows: Any,
    ) -> None:
        """Adam-update selected rows of a row-layout group."""
        old = self.global_parameters[param_name]
        _, bc1, bc2 = self._bump_adam_step_once_per_round()
        rows = np.asarray(union_rows, dtype=np.int64)
        if isinstance(old, dict):
            updated = {}
            for key, table in old.items():
                like = table if torch.is_tensor(table) else torch.as_tensor(table)
                grad_slice = torch.as_tensor(
                    agg_rows[key], dtype=like.dtype, device=like.device
                )
                if not torch.isfinite(grad_slice).all():
                    raise ValueError(
                        f"Non-finite aggregated gradient for {param_name}.{key}"
                    )
                out = like.clone()
                idx = torch.as_tensor(rows, dtype=torch.long, device=like.device)
                old_slice = like.index_select(0, idx)
                new_slice = self._adam_step_tensor(
                    param_name,
                    key,
                    old_slice,
                    grad_slice,
                    bc1,
                    bc2,
                    row_ids=rows,
                )
                out.index_copy_(0, idx, new_slice)
                updated[key] = out
            self.global_parameters[param_name] = updated
            return
        raise TypeError(
            f"FCIS row group {param_name!r} must be a dict table, got {type(old)}"
        )

    def _adam_step_tensor(
        self,
        param_name: str,
        tensor_name: str,
        like: torch.Tensor,
        g: torch.Tensor,
        bc1: float,
        bc2: float,
        row_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Bias-corrected Adam step for one tensor (full or row slice).

        Parameters
        ----------
        param_name, tensor_name : str
            Group and inner name for moment buffers.
        like : torch.Tensor
            Current parameter values (full tensor or row slice).
        g : torch.Tensor
            Gradient aligned with ``like``.
        bc1, bc2 : float
            Bias-correction denominators.
        row_ids : ndarray or None, optional
            When set, moments are read/written only at these rows of the
            full moment tables.

        Returns
        -------
        torch.Tensor
            Updated values, same shape as ``like``.
        """
        beta1, beta2 = self.betas
        m_full = self.adam_m[param_name][tensor_name].to(
            device=like.device, dtype=like.dtype
        )
        v_full = self.adam_v[param_name][tensor_name].to(
            device=like.device, dtype=like.dtype
        )
        if row_ids is None:
            m = beta1 * m_full + (1.0 - beta1) * g
            v = beta2 * v_full + (1.0 - beta2) * (g * g)
            self.adam_m[param_name][tensor_name] = m
            self.adam_v[param_name][tensor_name] = v
            m_hat = m / bc1
            v_hat = v / bc2
        else:
            idx = torch.as_tensor(row_ids, dtype=torch.long, device=like.device)
            m_slice = m_full.index_select(0, idx)
            v_slice = v_full.index_select(0, idx)
            m_slice = beta1 * m_slice + (1.0 - beta1) * g
            v_slice = beta2 * v_slice + (1.0 - beta2) * (g * g)
            m_full = m_full.clone()
            v_full = v_full.clone()
            m_full.index_copy_(0, idx, m_slice)
            v_full.index_copy_(0, idx, v_slice)
            self.adam_m[param_name][tensor_name] = m_full
            self.adam_v[param_name][tensor_name] = v_full
            m_hat = m_slice / bc1
            v_hat = v_slice / bc2
        new_value = like - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        if not torch.isfinite(new_value).all():
            raise ValueError(f"Non-finite Adam update for {param_name}.{tensor_name}")
        return new_value
