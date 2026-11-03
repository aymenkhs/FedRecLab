# Client selection

Selectors live in [`src/fedreclab/federated/client_selection.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/client_selection.py). A selector returns unique, valid client ids. Length is `m` except the short last cohort of round-robin. Seeded `RandomClientSelection` is a pure function of `(seed, round)`. Round-robin and Power-of-Choice keep a cursor, permutation or stored losses, so the next cohort depends on call history. `replace=True` (with-replacement draws) is not implemented.

Selectors are single-use: build a new one for every server. When `seed` is still `None`, the server writes `selector.seed = server.seed`.

`FIMSUniSampler` in EFVAE is **item** proposal sampling for FCIS, not a client selector. See [`methods.md`](methods.md#efvae).

## Overview

| Selector | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| FullClientSelection | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/client_selection.py) | Library primitive (every client trains every round) | — |
| RandomClientSelection | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/client_selection.py) | Library primitive (uniform or size-weighted sampling without replacement). Uniform sampling is the usual context of McMahan et al., AISTATS 2017; this class is not that paper's algorithm | see [aggregation.md](aggregation.md#fedavg) |
| RoundRobinClientSelection | 2023 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/client_selection.py) | RR-CLI: Malinovsky, Horváth, Burlachenko, Richtárik. Federated Learning with Regularized Client Participation. CyCP: Cho, Sharma, Joshi, Xu, Kale, Zhang. On the Convergence of Federated Averaging with Cyclic Client Participation. ICML 2023 | [arXiv:2302.03662](https://arxiv.org/abs/2302.03662) · [PMLR v202](https://proceedings.mlr.press/v202/cho23b.html) |
| PowerOfChoiceClientSelection | 2022 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/client_selection.py) | Cho, Wang, Joshi. Towards Understanding Biased Client Selection in Federated Learning. AISTATS 2022 (communication-free rpow-d) | [PMLR v151](https://proceedings.mlr.press/v151/jee-cho22a.html) · [arXiv:2110.04127](https://arxiv.org/abs/2110.04127) |
| ActvSampClientSelection | 2020 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/client_selection.py) | Muhammad et al. FedFast, KDD 2020, Algorithm 2 | [10.1145/3394486.3403176](https://doi.org/10.1145/3394486.3403176) |

## Shared constructor fields

From `BaseClientSelection`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_clients` | required | Population size `K` |
| `number_selected_clients` | required | Count `m`, or a fraction of `K` |
| `selection_value_type` | `"auto"` | `"num"`, `"frac"`, or `"auto"` (ints as counts, floats as fractions) |
| `seed` | `None` | Selector seed; the server copies its seed when this stays `None` |

Resolved `m` must lie in `[1, K]`. A fraction must lie in `(0, 1]`.

---

## FullClientSelection

Every client trains every round. `m` is set equal to `num_clients`.

```python
from fedreclab.federated.client_selection import FullClientSelection

selector = FullClientSelection(dataset.num_clients)
```

Used by the faithful GPFedRec and SemiDFEGL setups, and by [`examples/implement_your_fed_rec.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/implement_your_fed_rec.ipynb).

---

## RandomClientSelection

Uniform sampling without replacement by default. With `weights` (typically `ClientPartition.client_train_sizes`), each draw picks client `k` with probability proportional to that weight, without replacement. For `m > 1` the inclusion probabilities are then not exactly those weights.

**Extra parameters.** `weights=None`, `replace=False` (`True` raises `NotImplementedError`).

**Pairing.** Uniform draws with [`FedAvg`](aggregation.md); weighted draws with [`SimpleAvg`](aggregation.md) so data size is not applied again at aggregation.

```python
from fedreclab.federated.client_selection import RandomClientSelection

selector = RandomClientSelection(dataset.num_clients, 0.02, "frac", seed=42)
# or size-weighted:
# RandomClientSelection(
#     dataset.num_clients, 10, "num", seed=42,
#     weights=dataset.client_partition.client_train_sizes)
```

**Examples.** [`examples/quickstart.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/quickstart.ipynb), [`examples/custom_data.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/custom_data.ipynb), [`examples/custom_aggregator.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/custom_aggregator.ipynb), [`examples/extend_fedncf.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/extend_fedncf.ipynb).

---

## RoundRobinClientSelection

Without `groups` this is RR-CLI / CyCP with the whole cohort training: a permutation of `[0, K)`, take the next `m` ids, advance the cursor. When `K` is not divisible by `m`:

- `remainder="short_last"` (default) keeps exactly-once coverage with a smaller final cohort.
- `remainder="wrap"` always returns `m` ids and breaks exactly-once across rounds.

With `groups`, the given disjoint lists are CyCP groups visited in order. `clients_per_group=None` trains the whole group; otherwise `N` clients are sampled uniformly without replacement from the current group.

**Extra parameters.** `reshuffle_each_cycle=False`, `groups=None`, `clients_per_group=None`, `remainder="short_last"`.

**Limitations.** Dropout re-queue is not implemented: a dropped client still loses that turn. Pair with [`SimpleAvg`](aggregation.md).

```python
from fedreclab.federated.client_selection import RoundRobinClientSelection

selector = RoundRobinClientSelection(dataset.num_clients, 8, "num", seed=42)
```

No notebook demonstrates this selector.

---

## PowerOfChoiceClientSelection

Communication-free rpow-d: sample `d` candidates by `p_k`, keep the `m` with the highest stored losses. Selected clients' mean of `training_losses` is stored in `on_round_end`. Unseen clients stay at `+∞`, so they are preferred until they have trained once. `d_schedule` implements adapow-d by shrinking `d` toward `m`. pow-d and cpow-d (which need a pre-train eval phase) are not implemented.

**Extra parameters.** `d` (required, `m <= d <= K`), `weights=None`, `d_schedule=None`.

**Attributes.** `requires_individual_losses = True`. Under a future aggregate-only visibility mode those scalars would travel outside secure aggregation.

**Deviation.** The stored loss is the mean of per-epoch batch means, not the paper's per-step mean. Pair with [`SimpleAvg`](aggregation.md).

```python
from fedreclab.federated.client_selection import PowerOfChoiceClientSelection

selector = PowerOfChoiceClientSelection(
    dataset.num_clients, 10, d=30, seed=42)
```

No notebook demonstrates this selector.

---

## ActvSampClientSelection

FedFast ActvSAMP (Algorithm 2): clients are partitioned into groups `G`. Each round draws `m // p` clients uniformly without replacement from every cluster (`p = |G|`). `m` must be divisible by `p`. Groups must be installed with `set_groups` before the first `select`. FedFast refreshes groups after each ActvAGG recluster.

**Deviation.** When a cluster has fewer than `m // p` members, that cluster contributes all of its members and any shortfall is filled from clients not yet chosen. The paper's prose also mentions round-robin one-per-cluster sampling; that variant is not implemented.

```python
from fedreclab.federated.client_selection import ActvSampClientSelection

selector = ActvSampClientSelection(dataset.num_clients, 40, "num", seed=42)
selector.set_groups([[0, 1], [2, 3], [4, 5]])  # FedFast calls this for you
```

No standalone notebook. Wired by [`FedFast`](methods.md#fedfast).

---

## Writing a custom selector

Subclass `BaseClientSelection` and implement `select(self, current_round=0) -> list[int]`. Override `on_round_start` / `on_round_end` only when the selector needs round statistics (as Power-of-Choice does). There is no custom-selector notebook; the pattern appears in tests (for example a scripted selector in `tests/gpfedrec/`).
