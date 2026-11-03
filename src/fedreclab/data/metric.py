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


class BiasMetrics:
    """Computes bias metrics from per-user ranked lists.
 
    Parameters
    ----------
    train_interactions : DataFrame
        Train interactions (``user_col``, ``item_col``). Used for item popularity,
        user profiles (calibration, GAP) and user activity (UGF).
    catalog_items : array-like
        Every item id that can be ranked (e.g. ``evaluator.full_item_list``).
    item_categories : mapping item_id -> sequence of categories, optional
        Enables category-aware metrics and calibration.
    user_groups : mapping user_id -> group label, optional
        Protected attribute (e.g. gender). Enables GBS, Beyond Parity and group gaps.
    protected_groups : (a, b), optional
        The two labels to compare. Inferred when ``user_groups`` has exactly two labels.
    topk : int or sequence of int
        Cut-offs for the top-k metrics.
    compute_crp : bool
        CRP needs ranked lists as long as floor(R_c) (can be large). Disable to keep
        memory at ``max(topk)`` per user.
    log_base : float
        Log base in CDCG's discount (paper writes ``log(j+1)``; base 2 by default).
    calibration_alpha : float
        Smoothing in Steck's calibration (0.01 in the paper).
    calibration_rank_discount : bool
        Weight recommended items by 1/log2(rank+1) when building q(c|u).
    activity_top_frac : float
        Fraction of most active users forming the advantaged group for UGF (0.05).
    head_share, tail_share : float
        Head / tail items = most / least popular items covering that share of train interactions.
    popularity_user_fracs : (blockbuster, diverse, niche)
        Sizes of the user groups ordered by mean popularity of their train profile.
    """
 
    def __init__(
        self,
        train_interactions: pd.DataFrame,
        catalog_items,
        *,
        item_categories: Mapping[Hashable, Sequence[Hashable]] | None = None,
        user_groups: Mapping[Hashable, Hashable] | None = None,
        protected_groups: tuple | None = None,
        topk: int | Sequence[int] = (10,),
        user_col: str = "user_id",
        item_col: str = "item_id",
        compute_crp: bool = True,
        log_base: float = 2.0,
        calibration_alpha: float = 0.01,
        calibration_rank_discount: bool = False,
        activity_top_frac: float = 0.05,
        head_share: float = 0.2,
        tail_share: float = 0.2,
        popularity_user_fracs: tuple = (0.2, 0.6, 0.2),
    ):
        self.topk = [topk] if isinstance(topk, int) else list(topk)
        self.catalog = pd.Index(catalog_items)
        self.log_base = log_base
        self.compute_crp = compute_crp
        self.alpha = calibration_alpha
        self.rank_discount = calibration_rank_discount
        self.user_groups = dict(user_groups) if user_groups else None
        self.group_pair = None
        if self.user_groups:
            labels = sorted(set(self.user_groups.values()), key=str)
            if protected_groups is None:
                if len(labels) != 2:
                    raise ValueError(
                        f"user_groups has {len(labels)} labels; pass protected_groups=(a, b)."
                    )
                protected_groups = tuple(labels)
            self.group_pair = tuple(protected_groups)
 
        self.cm = CategoryMatrix(item_categories, self.catalog) if item_categories else None
 
        train = train_interactions[[user_col, item_col]].drop_duplicates()
        self.profiles = {u: g.to_numpy() for u, g in train.groupby(user_col)[item_col]}
 
        # --- activity groups (UGF) ---
        counts = train.groupby(user_col).size().sort_values(ascending=False, kind="mergesort")
        n_adv = max(1, math.ceil(activity_top_frac * len(counts)))
        self.advantaged = set(counts.index[:n_adv])
 
        # --- popularity ---
        if head_share + tail_share >= 1.0:
            raise ValueError("head_share + tail_share must be < 1")
        item_counts = train.groupby(item_col).size().reindex(self.catalog, fill_value=0)
        n_users = max(train[user_col].nunique(), 1)
        self._phi = (item_counts / n_users).to_numpy(dtype=float)
        total = float(item_counts.sum())
        desc = item_counts.sort_values(ascending=False, kind="mergesort")
        head = desc.index[(desc.cumsum() - desc) < head_share * total]
        asc = desc[::-1]
        tail = asc.index[(asc.cumsum() - asc) < tail_share * total].difference(head)
        bucket = pd.Series(1, index=self.catalog)  # 0 head, 1 mid, 2 tail
        bucket.loc[tail] = 2
        bucket.loc[head] = 0
        self._bucket = bucket.to_numpy()
 
        prof_pop = {
            u: float(self._phi[self.catalog.get_indexer(items)[self.catalog.get_indexer(items) >= 0]].mean())
            for u, items in self.profiles.items()
            if (self.catalog.get_indexer(items) >= 0).any()
        }
        self.profile_pop = prof_pop
        ordered = sorted(prof_pop, key=prof_pop.get, reverse=True)
        n = len(ordered)
        n_b = math.ceil(popularity_user_fracs[0] * n)
        n_n = math.ceil(popularity_user_fracs[2] * n)
        self.pop_group: Dict[Hashable, str] = {}
        for i, u in enumerate(ordered):
            self.pop_group[u] = (
                "blockbuster" if i < n_b else "niche" if i >= n - n_n else "diverse"
            )
 
    # ------------------------------------------------------------------ utils
    @property
    def max_rank_len(self) -> int:
        """How many ranked items per user should be passed to ``compute``."""
        base = max(self.topk)
        if self.cm is not None and self.compute_crp:
            base = max(base, int(np.floor(self.cm.total_mass.max())))
        return base
 
    def _masks(self, users):
        if not self.group_pair:
            return None
        a, b = self.group_pair
        arr = np.array([self.user_groups.get(u) for u in users], dtype=object)
        ma, mb = arr == a, arr == b
        if not ma.any() or not mb.any():
            warnings.warn(f"Group {a!r} or {b!r} has no evaluated users; group gaps skipped.")
            return None
        return ma, mb
 
    # ------------------------------------------------------------ main entry
    def compute(
        self,
        rankings: Mapping[Hashable, np.ndarray],
        per_user_metrics: Mapping[Hashable, Mapping[str, float]] | None = None,
        eval_pairs: pd.DataFrame | None = None,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        """Return ``(scalars, details)``.
 
        rankings : user_id -> ranked item ids (best first, length >= ``max_rank_len``
            when possible).
        per_user_metrics : user_id -> {"ndcg@10": ...}; enables UGF / group gaps.
        eval_pairs : DataFrame(user_id, item_id, pred, rating); enables Beyond Parity.
        """
        scalars: Dict[str, float] = {}
        details: Dict[str, Any] = {}
        if self.cm is not None:
            self._category_block(rankings, scalars, details)
            self._calibration_block(rankings, scalars)
        if per_user_metrics:
            self._ugf_block(per_user_metrics, scalars)
        self._popularity_block(rankings, scalars, details)
        if eval_pairs is not None and self.group_pair and len(eval_pairs):
            scalars.update(beyond_parity_metrics(eval_pairs, self.user_groups, *self.group_pair))
        return scalars, details
 
    # -------------------------------------------------------- category-aware
    def _category_block(self, rankings, scalars, details):
        kept, rows = [], {}
        for u, ranked in rankings.items():
            m = category_user_metrics(
                self.cm, np.asarray(ranked), self.topk, self.max_rank_len,
                self.log_base, self.compute_crp,
            )
            if not m:
                continue
            kept.append(u)
            for name, vec in m.items():
                rows.setdefault(name, []).append(vec)
        if not kept:
            return
        masks = self._masks(kept)
        for name, vecs in rows.items():
            M = np.vstack(vecs)
            tbl = pd.DataFrame({"all": M.mean(axis=0)}, index=self.cm.categories)
            if masks:
                ma, mb = masks
                a, b = self.group_pair
                tbl[str(a)] = M[ma].mean(axis=0)
                tbl[str(b)] = M[mb].mean(axis=0)
                tbl["abs_diff"] = (tbl[str(a)] - tbl[str(b)]).abs()
                scalars[f"gbs_{name}"] = float(tbl["abs_diff"].sum())
            details[f"category/{name}"] = tbl
 
    # ----------------------------------------------------------- calibration
    def _calibration_block(self, rankings, scalars):
        vals = {k: {} for k in self.topk}
        for u, ranked in rankings.items():
            prof = self.profiles.get(u)
            if prof is None:
                continue
            p = self.cm.weights(prof).sum(axis=0)
            if p.sum() <= 0:
                continue
            p = p / p.sum()
            for k in self.topk:
                top = np.asarray(ranked)[:k]
                if len(top) == 0:
                    continue
                w = (1.0 / np.log2(np.arange(2, len(top) + 2))) if self.rank_discount else np.ones(len(top))
                q = (w[:, None] * self.cm.weights(top)).sum(axis=0)
                if q.sum() > 0:
                    q = q / q.sum()
                q_t = (1 - self.alpha) * q + self.alpha * p
                m = p > 0
                vals[k][u] = float(np.sum(p[m] * np.log(p[m] / q_t[m])))
        for k, d in vals.items():
            if not d:
                continue
            scalars[f"calib_kl@{k}"] = float(np.mean(list(d.values())))
            users = list(d)
            masks = self._masks(users)
            if masks:
                v = np.array([d[u] for u in users])
                scalars[f"calib_kl_gap@{k}"] = float(abs(v[masks[0]].mean() - v[masks[1]].mean()))
 
    # ------------------------------------------------------------------ UGF
    def _ugf_block(self, per_user_metrics, scalars):
        users = list(per_user_metrics)
        adv = np.array([u in self.advantaged for u in users])
        masks = self._masks(users)
        names = per_user_metrics[users[0]].keys()
        for name in names:
            v = np.array([per_user_metrics[u][name] for u in users], dtype=float)
            if adv.any() and (~adv).any():
                scalars[f"ugf_{name}"] = float(abs(v[adv].mean() - v[~adv].mean()))
            if masks:
                scalars[f"group_gap_{name}"] = float(abs(v[masks[0]].mean() - v[masks[1]].mean()))
 
   