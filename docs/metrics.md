# Metrics

Two families:

1. **Ranking metrics** accepted by [`FullRankingEvaluator`](evaluator.md), implemented by [Cornac](https://cornac.preferred.ai/) classes in [`evaluator.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py).
2. **Bias / fairness metrics** in [`metric.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/metric.py). Original FedRecLab code; not wired into the evaluator. Call `BiasMetrics.compute` yourself.

Cornac framework citation: Salah, Truong, Lauw. Cornac: A Comparative Framework for Multimodal Recommender Systems. *JMLR* 21(95):1–5, 2020. [jmlr.org/papers/v21/20-137.html](https://jmlr.org/papers/v21/20-137.html).

There is no RMSE or MAE in either module.

## Overview — ranking (Cornac, via the evaluator)

| Key | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| `ndcg` (`NDCG`) | 2002 | [evaluator import](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Järvelin, Kekäläinen. Cumulated gain-based evaluation of IR techniques. *ACM TOIS* 20(4):422–446 | [10.1145/582415.582418](https://doi.org/10.1145/582415.582418) |
| `precision` (`Precision`) | — | [evaluator import](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Standard top-K precision; Cornac implementation (Salah et al., JMLR 2020) | [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |
| `recall` (`Recall`) | — | [evaluator import](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Standard top-K recall; Cornac implementation | [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |
| `hr` (`HitRatio`) | — | [evaluator import](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Hit ratio / Hit Rate at `k`; Cornac implementation | [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |
| `fmeasure` (`FMeasure`) | — | [evaluator import](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Harmonic mean of precision and recall at the same `k`; Cornac implementation | [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |
| `ncrr` (`NCRR`) | — | [evaluator import](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Cornac Normalized Cumulative Reciprocal Rank. The installed Cornac class docstring does not name a paper DOI | Cornac docs / [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |

## Overview — bias / fairness (`metric.py`)

| Family | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| Category-aware metrics and GBS | 2025 | [metric.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/metric.py) | Kheya, Bouadjenek, Aryal. Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness. WWW 2025 | [10.1145/3696410.3714528](https://doi.org/10.1145/3696410.3714528) |
| Beyond Parity | 2017 | [metric.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/metric.py) | Yao, Huang. Beyond Parity: Fairness Objectives for Collaborative Filtering. NeurIPS 2017 | [ACM DL](https://dl.acm.org/doi/10.5555/3294996.3295052) |
| UGF | 2021 | [metric.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/metric.py) | Li, Chen, Fu, Ge, Zhang. User-oriented Fairness in Recommendation. WWW 2021 | [10.1145/3442381.3449866](https://doi.org/10.1145/3442381.3449866) |
| Calibration | 2018 | [metric.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/metric.py) | Steck. Calibrated Recommendations. RecSys 2018 | [10.1145/3240323.3240372](https://doi.org/10.1145/3240323.3240372) |

---

## Ranking metrics (through the evaluator)

Pass string keys to `FullRankingEvaluator(metrics=[...], topk=...)`. The evaluator instantiates `TOPK_METRIC_CLASSES[metric](k=k)` and stores results as `"{metric}@{k}"`.

```python
from fedreclab.data.evaluator import FullRankingEvaluator

evaluator = FullRankingEvaluator(
    metrics=["ndcg", "hr", "precision", "recall"],
    topk=[5, 10],
    interaction_dataset=dataset,
    client_algorithm=client,
)
```

**Defaults.** Each Cornac class is constructed with the evaluation `k` only; there are no extra FedRecLab knobs on these metrics.

**Examples.** Quickstart and the other notebooks use `ndcg` and `hr` at 10.

---

## BiasMetrics

Original FedRecLab code in [metric.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/metric.py). Run after you have rankings (and optionally per-user ranking metrics or predicted-rating pairs).

### Constructor defaults

| Parameter | Default |
| --- | --- |
| `item_categories` | `None` |
| `user_groups` | `None` |
| `protected_groups` | `None` |
| `topk` | `(10,)` |
| `user_col` | `"user_id"` |
| `item_col` | `"item_id"` |
| `compute_crp` | `True` |
| `log_base` | `2.0` |
| `calibration_alpha` | `0.01` |
| `calibration_rank_discount` | `False` |
| `activity_top_frac` | `0.05` |
| `head_share` | `0.2` |
| `tail_share` | `0.2` |
| `popularity_user_fracs` | `(0.2, 0.6, 0.2)` |

`compute(rankings, per_user_metrics=None, eval_pairs=None)` returns `(scalars, details)`.

| Input | Enables |
| --- | --- |
| `rankings` | user → ranked item ids (best first) |
| `item_categories` + catalogue | Category metrics, GBS, calibration |
| `per_user_metrics` | UGF / group gaps |
| `eval_pairs` (`user_id`, `item_id`, `pred`, `rating`) | Beyond Parity |
| `user_groups` / `protected_groups` | GBS and group gaps between two groups |

### Category-aware metrics and GBS (Kheya et al., WWW 2025)

Needs `CategoryMatrix(item_categories, catalog_items)`. Per user and each `k` in `topk`: CC, RCR, CMAP, CDCG, CMRR, and CRP when `compute_crp=True`. Scalars include `gbs_{name}@k` and `gbs_crp` when two protected groups are set.

### Beyond Parity (Yao & Huang, NeurIPS 2017)

Scalars: `bp_upar`, `bp_uval`, `bp_uabs`, `bp_uunder`, `bp_uover`. Needs predicted ratings and true ratings on held-out pairs.

### UGF (Li et al., WWW 2021)

Absolute gap of a per-user metric between the most active users (top `activity_top_frac` by train interactions) and the rest: `ugf_{name}`. With two protected groups, also `group_gap_{name}`.

### Calibration (Steck, RecSys 2018)

KL between the user's category profile and a mixture of the top-K list: `calib_kl@k`, and `calib_kl_gap@k` when groups are set. Mixture weight is `calibration_alpha`.

```python
from fedreclab.data.metric import BiasMetrics, CategoryMatrix

# After you have rankings from an evaluator path you control:
# bias = BiasMetrics(item_categories=..., user_groups=..., topk=(10,))
# scalars, details = bias.compute(rankings, per_user_metrics=per_user)
```

---

## Writing a custom metric

- **New top-K ranking name.** Add a Cornac-style metric class and register it in `Evaluator.TOPK_METRIC_CLASSES`. The string list is closed until you edit that dict.
- **New fairness score.** Add a function in the style of `beyond_parity_metrics` or another block on `BiasMetrics`, and call it with the rankings (and optional per-user metrics) you already produced. No notebook demonstrates a custom metric.
