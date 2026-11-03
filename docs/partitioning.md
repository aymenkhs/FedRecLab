# Client partitioning

Partition strategies live in [`src/fedreclab/data/client_partitioning.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/client_partitioning.py). They assign every **training** user to exactly one client and return a `ClientPartition`. Validation and test interactions are never used to decide membership; optional val / test index lists on the partition are derived after the split for evaluation convenience.

No paper is cited in this module. Strategies are experiment configuration passed into `InteractionDataset` / the loaders in [`datasets.md`](datasets.md).

## Overview

| Strategy | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| CrossDevicePartitionStrategy | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/client_partitioning.py) | Library primitive (one user per client) | — |
| RandomUserPartitionStrategy | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/client_partitioning.py) | Library primitive (shuffle users into `K` clients) | — |
| DirichletUserPartitionStrategy | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/data/client_partitioning.py) | Library non-IID control in the spirit of Hsu, Qi, Brown. Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. arXiv:1909.06335 (2019). The class does not implement that paper | [arXiv:1909.06335](https://arxiv.org/abs/1909.06335) |

`ClientPartition` fields: `num_clients`, `client_ids_2_user_ids`, `user_ids_2_client_ids`, `client_to_train_indices`, `client_train_sizes`, optional `client_to_val_indices` / `client_to_test_indices`, `metadata`.

Alias: `CrossDeviceSplittingStrategy = CrossDevicePartitionStrategy` (kept for old imports).

---

## CrossDevicePartitionStrategy

Each unique training user is mapped to a distinct client. The number of clients is the number of distinct user ids in the training interactions; the constructor does not take `num_clients`.

| Parameter | Default |
| --- | --- |
| `user_col` | `"user_id"` |
| `metadata` | `None` |

```python
from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy

partition = CrossDevicePartitionStrategy()
```

This is the partition in [`examples/quickstart.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/quickstart.ipynb) and the other current notebooks.

---

## RandomUserPartitionStrategy

Shuffle users into `num_clients` clients.

| Parameter | Default |
| --- | --- |
| `num_clients` | required |
| `size_by` | `"users"` (`"users"` or `"interactions"`) |
| `seed` | `None` |
| `user_col` | `"user_id"` |
| `metadata` | `None` |

```python
from fedreclab.data.client_partitioning import RandomUserPartitionStrategy

partition = RandomUserPartitionStrategy(num_clients=100, size_by="users", seed=42)
```

No notebook demonstrates this strategy.

---

## DirichletUserPartitionStrategy

Symmetric Dirichlet skew over clients. `alpha` is required and must be positive. Smaller `alpha` yields stronger concentration.

| Parameter | Default |
| --- | --- |
| `num_clients` | required |
| `alpha` | required (`> 0`) |
| `size_by` | `"interactions"` (`"interactions"` or `"users"`) |
| `seed` | `None` |
| `user_col` | `"user_id"` |
| `metadata` | `None` |

```python
from fedreclab.data.client_partitioning import DirichletUserPartitionStrategy

partition = DirichletUserPartitionStrategy(
    num_clients=100, alpha=0.5, size_by="interactions", seed=42)
```

No notebook demonstrates this strategy.

---

## Writing a custom partition strategy

Subclass `BaseClientPartitionStrategy` (`num_clients`, `user_col="user_id"`, `metadata=None`) and implement `partition(self, train_interactions: pd.DataFrame) -> ClientPartition`. Assign every training user to exactly one client. No notebook defines a new strategy.
