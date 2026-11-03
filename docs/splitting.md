# User / interaction splitting

Split strategies live in [`src/fedreclab/data/split.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/split.py). A strategy's `split` returns train, validation and test **row indices** into the interaction table. Column names default to `user_id`, `item_col="item_id"`, `rating_col="rating"`, `time_col="timestamp"`.

Ratio and stratified splits delegate the draw to [Cornac](https://cornac.preferred.ai/) eval methods. Cite Salah, Truong, Lauw. Cornac: A Comparative Framework for Multimodal Recommender Systems. *JMLR* 21(95):1–5, 2020. [jmlr.org/papers/v21/20-137.html](https://jmlr.org/papers/v21/20-137.html).

Training structures (graphs, histories, vocabularies) must be built from training data only. Never use validation / test interactions outside evaluation.

## Overview

| Strategy | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| RatioSplitStrategy | 2020 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/split.py) | Wraps Cornac `RatioSplit` (Salah, Truong, Lauw, JMLR 2020) | [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |
| StratifiedSplitStrategy | 2020 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/split.py) | Wraps Cornac `StratifiedSplit` (same Cornac citation) | [JMLR 21(95)](https://jmlr.org/papers/v21/20-137.html) |
| LeaveNOutSplitStrategy | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/split.py) | Library leave-N-out per user (not a Cornac wrapper; no single originating paper claimed) | — |

Shared train / val / test flags on the ratio and stratified bases:

| Parameter | Default |
| --- | --- |
| `include_validation` | `False` |
| `include_test` | `False` |
| `val_size` | `0.0` |
| `test_size` | `0.0` |
| `seed` | `42` |
| `rating_threshold` | `0.0` |

If both validation and test are disabled, all interactions are returned as train.

---

## RatioSplitStrategy

Random ratio-based split of all interactions into train / val / test.

```python
from fedreclab.data.split import RatioSplitStrategy

split = RatioSplitStrategy(
    include_validation=True, include_test=True,
    val_size=0.1, test_size=0.2, seed=42)
```

No dedicated notebook. Prefer stratified splits when you need a fixed fraction per user.

---

## StratifiedSplitStrategy

Stratified split per user or per item, with optional chronological ordering. If `chrono=True`, interactions within each group are sorted by timestamp before splitting and Cornac is asked for `UIRT`.

| Extra parameter | Default |
| --- | --- |
| `by` | `"user"` (`"user"` or `"item"`) |
| `chrono` | `False` |

```python
from fedreclab.data.split import StratifiedSplitStrategy

split = StratifiedSplitStrategy(
    include_validation=True, include_test=True,
    val_size=0.1, test_size=0.2, chrono=True, by="user", seed=42)
```

This is the split in [`examples/quickstart.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/quickstart.ipynb) and the other current notebooks.

---

## LeaveNOutSplitStrategy

For every user, `n_val` interactions go to validation and `n_test` to test; the rest are train. When `chrono=True`, the chronological order of the sets is respected. Users without enough eligible interactions raise `ValueError`.

| Extra parameter | Default |
| --- | --- |
| `n_val` | `1` |
| `n_test` | `1` |
| `chrono` | `False` |
| `seed` | `42` |
| `rating_threshold` | `0.0` |

```python
from fedreclab.data.split import LeaveNOutSplitStrategy

split = LeaveNOutSplitStrategy(
    include_validation=True, include_test=True,
    n_val=1, n_test=1, chrono=False, seed=42)
```

No notebook demonstrates this strategy.

---

## Writing a custom split strategy

Subclass `BaseSplitStrategy` and implement `split(self, interaction_data)` returning `(train_idx, val_idx, test_idx)`. For train / val / test helpers that talk to Cornac, subclass `__BaseTrainValTestSplit` and implement `_build_cornac_splitter`. No notebook defines a new strategy.
