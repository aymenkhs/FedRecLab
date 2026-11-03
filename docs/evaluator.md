# Evaluator

Evaluators live in [`src/fedreclab/data/evaluator.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py). Evaluation is client-level: for each client the evaluator builds candidate item arrays and calls `client_algorithm.compute_scores(...)`. Every candidate of a user is scored by the same function. There is no `"loss"` key in the result (scoring is unlabelled).

Evaluators are single-use: build a new one for every server. Ranking metric names accepted by `metrics` are documented in full in [`metrics.md`](metrics.md). There is no RMSE or MAE in this module, which matters for FedMF, MetaMF and rating-mode FedPerGNN.

[`BiasMetrics`](metrics.md#biasmetrics) is not called from here; run it yourself on the rankings you collect.

## Overview

| Class | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| Evaluator | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Abstract base (library protocol) | — |
| FullRankingEvaluator | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/evaluator.py) | Concrete full-catalogue ranking protocol (library) | — |

Accepted ranking metric strings (Cornac implementations; see [`metrics.md`](metrics.md)): `ndcg`, `precision`, `recall`, `hr`, `ncrr`, `fmeasure`. Unknown keys raise `ValueError`.

---

## Evaluator

Abstract base. Subclasses must implement `evaluate`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `metrics` | required | Sequence of ranking metric names from the table above |
| `topk` | required | Int or sequence of ints; an int is stored as a one-element list |
| `interaction_dataset` | required | Dataset used to build candidates and targets |
| `client_algorithm` | required | Provides `compute_scores` |
| `compute_confidence_intervals` | `False` | If True, aggregate with mean / std / normal CI |
| `confidence_interval_alpha` | `0.05` | Two-sided normal CI level |
| `logging_callback` | `None` | Optional `callback(current_round, metric_results, split, extra_payload)` |
| `verbose` | `True` | Progress logging |

Per-user scores become keys `"{metric}@{k}"` via `compute_metrics_per_user`. Confidence intervals use `z = Φ⁻¹(1 − α/2)` and return `mean`, `std`, `ei`, `ci`.

---

## FullRankingEvaluator

The only concrete evaluator in the library. Candidates are every catalogue item in `interaction_dataset.interaction_data` minus the user's items from a configured exclusion set:

| Split | Default exclusion |
| --- | --- |
| validation (`split="val"`) | `exclude_on_val=("train",)` |
| test (`split="test"`) | `exclude_on_test=("train", "val")` |

Validation items are therefore not treated as test negatives. Pass `exclude_on_test=("train",)` to restore the older “exclude train only” convention. The eval split itself must not appear in its exclusion set, or the positives would be dropped from the ranking. Candidate arrays are built during `evaluate()` for users who have interactions in that split.

| Extra parameter | Default |
| --- | --- |
| `threshold_rating` | `0.0` |
| `exclude_on_val` | `("train",)` |
| `exclude_on_test` | `("train", "val")` |

**Protocol.** For each client in `client_states`, find users with eval interactions (optionally `rating >= threshold_rating`), build candidates, call `compute_scores`, sort by score descending, run the configured Cornac metrics, then average over users (or return CI dicts).

```python
from fedreclab.data.evaluator import FullRankingEvaluator

evaluator = FullRankingEvaluator(
    metrics=["ndcg", "hr"],
    topk=10,
    interaction_dataset=dataset,
    client_algorithm=client,
)

results = evaluator.evaluate(
    current_round=server.early_stopping_best_round,
    global_parameters=server.best_global_parameters,
    client_states=server.best_client_states,
    extra_payload=server.best_extra_payload,
    split="test",
)
```

---

## Writing a custom evaluator

Subclass `Evaluator` and implement `evaluate(...)`. Reuse `compute_metrics_per_user` for Cornac top-K metrics, or call [`BiasMetrics`](metrics.md#biasmetrics) yourself on the rankings. No notebook subclasses `Evaluator`.
