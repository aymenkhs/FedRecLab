"""Bias / fairness metrics for recommender evaluation.
 
Families implemented
--------------------
1. Category-aware metrics (Kheya et al., WWW'25): CC, RCR, CMAP, CDCG, CMRR, CRP
   and the Gender Balance Score GBS(M) = sum_c |M(c, U_a) - M(c, U_b)|.https://dl.acm.org/doi/10.1145/3696410.3714528
   
2. Beyond Parity (Yao & Huang NIPS, 2017): value / absolute / under / over / non-parity
   unfairness. Needs predicted *ratings* and true ratings on held-out pairs. https://dl.acm.org/doi/10.5555/3294996.3295052
   
3. UGF (Li et al., WWW'21): |mean(metric | advantaged) - mean(metric | disadvantaged)|,
   with advantaged = most active users (top ``activity_top_frac`` by train interactions). https://dl.acm.org/doi/10.1145/3442381.3449866
   
4. Calibration (Steck, RecSys'18): KL(p(c|u) || (1-alpha) q(c|u) + alpha p(c|u)). https://dl.acm.org/doi/10.1145/3240323.3240372

5. Popularity bias (Abdollahpouri et al., 2019): GAP / delta-GAP per user group,
   head/mid/tail exposure, average recommended popularity, item coverage. https://arxiv.org/abs/1907.13286
 
"""
from __future__ import annotations
 
import math
import warnings
from typing import Any, Dict, Hashable, Mapping, Sequence
 
import numpy as np
import pandas as pd

def _safe_div(num, den):
    """Elementwise num / den with 0 where den <= 0."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den > 0)
    return out

class CategoryMatrix:
    """Item x category matrix W with W[j, c] = 1 / |C_j| if item j has category c.
    Items with no category (or not in the catalogue) get an all-zero row. Refer to Kheya et al., WWW'25
    """
 
    def __init__(self, item_categories: Mapping[Hashable, Sequence[Hashable]], catalog_items):
        self.catalog = pd.Index(catalog_items)
        self.categories = list(dict.fromkeys(c for cs in item_categories.values() for c in cs))
        col = {c: i for i, c in enumerate(self.categories)}
        W = np.zeros((len(self.catalog), len(self.categories)))
        items = list(item_categories.keys())
        rows = self.catalog.get_indexer(items)
        for r, item in zip(rows, items):
            if r < 0:
                continue
            cs = list(dict.fromkeys(item_categories[item]))
            for c in cs:
                W[r, col[c]] = 1.0 / len(cs)
        self.W = W
        self.total_mass = W.sum(axis=0)  # sum_{v in V} C_{v,c} / |C_v|
 
    def weights(self, item_ids) -> np.ndarray:
        idx = self.catalog.get_indexer(item_ids)
        out = np.zeros((len(idx), self.W.shape[1]))
        ok = idx >= 0
        out[ok] = self.W[idx[ok]]
        return out
def category_user_metrics(
    cm: CategoryMatrix,
    ranked: np.ndarray,
    topk: Sequence[int],
    max_len: int,
    log_base: float = 2.0,
    compute_crp: bool = True,
) -> Dict[str, np.ndarray]:
    """Per-user category vectors (length = #categories) for one ranked list."""
    full = cm.weights(ranked[:max_len])
    out: Dict[str, np.ndarray] = {}
    if len(full) == 0:
        return out
    for k in topk:
        top = full[:k]
        L = len(top)
        j = np.arange(1, L + 1)[:, None]
        out[f"cc@{k}"] = top.sum(axis=0) / L
        out[f"rcr@{k}"] = _safe_div(top.sum(axis=0), cm.total_mass)
        P = np.cumsum(top, axis=0) / j
        out[f"cmap@{k}"] = _safe_div((P * top).sum(axis=0), cm.total_mass)
        disc = np.log(j + 1) / np.log(log_base)
        out[f"cdcg@{k}"] = (top / disc).sum(axis=0) / L
        out[f"cmrr@{k}"] = (top / j).sum(axis=0) / L
    if compute_crp:
        R = np.floor(cm.total_mass).astype(int)
        n = np.minimum(R, len(full))
        cum = np.cumsum(full, axis=0)
        num = np.where(n > 0, cum[np.maximum(n - 1, 0), np.arange(full.shape[1])], 0.0)
        out["crp"] = _safe_div(num, R)
    return out


def beyond_parity_metrics(
    pairs: pd.DataFrame,
    user_groups: Mapping[Hashable, Hashable],
    group_a: Hashable,
    group_b: Hashable,
) -> Dict[str, float]:
    df = pairs.copy()
    df["grp"] = df["user_id"].map(user_groups)
    df = df[df["grp"].isin([group_a, group_b])]
    pred_a = df.loc[df["grp"] == group_a, "pred"]
    pred_b = df.loc[df["grp"] == group_b, "pred"]
    if pred_a.empty or pred_b.empty:
        return {}
    # Non-parity: |E_g[y_hat] - E_notg[y_hat]| over ALL predictions of each group
    # (overall means, not per item).
    out = {"bp_upar": float(abs(pred_a.mean() - pred_b.mean()))}
 
    # Per-item metrics: only items rated by users of both groups contribute.
    g = df.groupby(["item_id", "grp"])[["pred", "rating"]].mean().unstack("grp")
    g = g[g[("pred", group_a)].notna() & g[("pred", group_b)].notna()]
    if g.empty:
        return out
    ea = g[("pred", group_a)] - g[("rating", group_a)]  # E_g[y_hat]_j - E_g[r]_j
    eb = g[("pred", group_b)] - g[("rating", group_b)]
    out.update(
        {
            "bp_uval": float((ea - eb).abs().mean()),
            "bp_uabs": float((ea.abs() - eb.abs()).abs().mean()),
            "bp_uunder": float((np.maximum(0, -ea) - np.maximum(0, -eb)).abs().mean()),
            "bp_uover": float((np.maximum(0, ea) - np.maximum(0, eb)).abs().mean()),
        }
    )
    return out