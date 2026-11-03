# Supported datasets

Datasets enter the library as a single interaction table. [`load_cornac_dataset`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/load_data.py) downloads a [Cornac](https://cornac.preferred.ai/) dump, builds a pandas frame, optionally filters ratings, then builds an [`InteractionDataset`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/dataset.py), splits it and partitions clients. IDs are remapped to contiguous 0-based integers inside `InteractionDataset`; **0 is a real id** after remapping.

Cornac framework citation: Salah, Truong, Lauw. Cornac: A Comparative Framework for Multimodal Recommender Systems. *JMLR* 21(95):1–5, 2020. [jmlr.org/papers/v21/20-137.html](https://jmlr.org/papers/v21/20-137.html).

Dataset papers below are the standard citations for each dump; they are not declared inside the library source. Timestamps follow `CORNAC_INCLUDE_TIMESTAMPS` in [`load_data.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/load_data.py).

## Overview — Cornac keys

| Key | Year | Loader code | Paper | DOI / link | Timestamps |
| --- | --- | --- | --- | --- | --- |
| `movielens_100k` | 2016 | [`load_data.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/load_data.py) | Harper, Konstan. The MovieLens Datasets: History and Context. *ACM TiiS* 5(4) | [10.1145/2827872](https://doi.org/10.1145/2827872) | yes (`100K`) |
| `movielens_1m` | 2016 | same | same | [10.1145/2827872](https://doi.org/10.1145/2827872) | yes (`1M`) |
| `movielens_10m` | 2016 | same | same | [10.1145/2827872](https://doi.org/10.1145/2827872) | yes (`10M`) |
| `movielens_20m` | 2016 | same | same | [10.1145/2827872](https://doi.org/10.1145/2827872) | yes (`20M`) |
| `filmtrust` | 2013 | same | Guo, Zhang, Yorke-Smith. A Novel Bayesian Similarity Measure for Recommender Systems. IJCAI 2013. Cornac's loader points at [librec.net/datasets](https://www.librec.net/datasets.html) and does not name a paper in its docstring | [IJCAI 2013 proceedings](https://www.ijcai.org/Proceedings/13/) | no |
| `amazon_clothing` | 2016 | same | He, McAuley. Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering. WWW 2016. Cornac builds on McAuley's Amazon dumps | [10.1145/2872427.2883037](https://doi.org/10.1145/2872427.2883037) | no |
| `amazon_digital_music` | 2016 | same | same Amazon dump family | [10.1145/2872427.2883037](https://doi.org/10.1145/2872427.2883037) | no |
| `amazon_office` | 2016 | same | same Amazon dump family | [10.1145/2872427.2883037](https://doi.org/10.1145/2872427.2883037) | no |
| `amazon_toy` | 2016 | same | same Amazon dump family | [10.1145/2872427.2883037](https://doi.org/10.1145/2872427.2883037) | no |
| `netflix_small` | 2007 | same | Bennett, Lanning. The Netflix Prize. KDD Cup and Workshop 2007 | [KDD Cup 2007](https://www.kdd.org/kdd2007/KDDCup.html) (no Crossref DOI) | yes (`small`) |
| `netflix_original` | 2007 | same | same | [KDD Cup 2007](https://www.kdd.org/kdd2007/KDDCup.html) | yes (`original`) |
| `tradesy` | 2016 | same | He, McAuley. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. AAAI 2016. Cornac notes this is the VBPR dump | [10.1609/aaai.v30i1.10340](https://doi.org/10.1609/aaai.v30i1.10340) | no |
| `epinions` | 2007 | same | Massa, Avesani. Trust-aware Recommender Systems. RecSys 2007 | [10.1145/1297231.1297235](https://doi.org/10.1145/1297231.1297235) | no |
| `citeulike` | 2011 | same | Wang, Blei. Collaborative Topic Modeling for Recommending Scientific Articles. KDD 2011 (Cornac also notes the later `citeulike-a` collection of Wang, Chen, Li, IJCAI 2013) | [10.1145/2020408.2020480](https://doi.org/10.1145/2020408.2020480) | no |

## Loader parameters

`load_cornac_dataset(dataset_name, splitting_strategy, client_partitioning_strategy, ...)`:

| Parameter | Default |
| --- | --- |
| `user_col` / `item_col` / `rating_col` | `"user_id"` / `"item_id"` / `"rating"` |
| `time_col` | `"timestamp"` (ignored when the Cornac source has no timestamps) |
| `include_validation` / `include_test` | `False` |
| `k_core` | `0` |
| `min_user_interactions` / `min_item_interactions` | `0` |
| `seed` | `42` |
| `rating_exclusion_threshold` | `0.0` (keep rows with rating `>=` this value when `> 0`) |

Unknown `dataset_name` raises `ValueError` listing the keys above. See [`splitting.md`](splitting.md) and [`partitioning.md`](partitioning.md) for the strategy objects.

### Instantiation (Cornac)

```python
from fedreclab.data.load_data import load_cornac_dataset
from fedreclab.data.split import StratifiedSplitStrategy
from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy

dataset = load_cornac_dataset(
    dataset_name="movielens_100k",
    splitting_strategy=StratifiedSplitStrategy(
        include_validation=True, include_test=True,
        val_size=0.1, test_size=0.2, chrono=True, by="user"),
    client_partitioning_strategy=CrossDevicePartitionStrategy(),
    include_validation=True, include_test=True, seed=42,
)
```

**Example.** [`examples/quickstart.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/quickstart.ipynb) loads `movielens_100k` this way. For FedVAE's MovieLens protocol, pass `rating_exclusion_threshold=4`.

---

## Custom datasets

Any interaction table can be loaded with `load_dataframe_dataset(df, splitting_strategy, client_partitioning_strategy, ...)`. The function runs the same filter → `InteractionDataset` → split → partition pipeline. A missing rating or timestamp column warns and is dropped. Direct `InteractionDataset(...)` construction is the lower-level path; set `splits_included=True` to read an existing `split` column instead of calling a split strategy.

`load_dataframe_dataset` defaults match the Cornac loader (`seed=42`, `k_core=0`, interaction minima `0`, `rating_exclusion_threshold=0.0`).

```python
import pandas as pd
from fedreclab.data.load_data import load_dataframe_dataset
from fedreclab.data.split import StratifiedSplitStrategy
from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy

df = pd.DataFrame({
    "user_id": [...],
    "item_id": [...],
    "rating": [...],
    "timestamp": [...],  # optional
})

dataset = load_dataframe_dataset(
    df,
    splitting_strategy=StratifiedSplitStrategy(
        include_validation=True, include_test=True,
        val_size=0.1, test_size=0.2, by="user"),
    client_partitioning_strategy=CrossDevicePartitionStrategy(),
    include_validation=True, include_test=True, seed=42,
)
```

**Worked example.** [`examples/custom_data.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/custom_data.ipynb) builds a synthetic interaction table, then loads Last.fm 2K as a pandas frame. Last.fm 2K is **not** a `load_cornac_dataset` key. [`examples/implement_your_fed_rec.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/implement_your_fed_rec.ipynb) also builds a small frame for the popularity baseline.

---

## Writing a custom dataset path

Prefer `load_dataframe_dataset` for any new dump. If you need pre-assigned train / val / test labels, put a `split` column on the frame and construct `InteractionDataset(..., splits_included=True)` before calling `partition_clients()`. Do not re-map IDs yourself; `InteractionDataset` owns remapping.
