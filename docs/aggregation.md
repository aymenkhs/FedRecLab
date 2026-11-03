# Aggregation

Aggregators live in [`src/fedreclab/federated/aggregation.py`](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py). Construct every aggregator as `Name(num_clients, ...)`. Pass one instance for all groups, or a `{group_name: aggregator}` dict to the server.

**Update format.** Aggregators accept a stacked `torch.Tensor` / `np.ndarray` with leading dimension equal to the number of clients, a list of tensors / arrays, or a list of dicts of tensors. They compute in NumPy and return the input's type, dtype and device.

**Row layout.** For `ParamIO.layout="rows"`, the server packs `{rows, values}` uploads into `values` of shape `(n, k, d)` and a bool `mask` of shape `(n, k)`. Mean-family aggregators and Median override `aggregate_rows` with a vectorised masked reduce; others fall back to one `aggregate()` call per union row.

Aggregators are single-use: build new ones for every server. Methods must not special-case aggregator classes.

## Overview

| Aggregator | Year | Code | Paper | DOI / link |
| --- | --- | --- | --- | --- |
| FedAvg | 2017 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | McMahan, Moore, Ramage, Hampson, Agüera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017, PMLR 54:1273–1282 | [PMLR v54](https://proceedings.mlr.press/v54/mcmahan17a.html) · [arXiv:1602.05629](https://arxiv.org/abs/1602.05629) |
| SimpleAvg | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Library primitive (uniform mean) | — |
| Sum | — | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Library primitive (unweighted sum) | — |
| Median | 2018 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Yin, Chen, Ramchandran, Bartlett. Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML 2018, PMLR 80:5650–5659 | [PMLR v80](https://proceedings.mlr.press/v80/yin18a.html) · [arXiv:1803.01498](https://arxiv.org/abs/1803.01498) |
| TrimmedMean | 2018 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Same Yin et al. ICML 2018 paper | [PMLR v80](https://proceedings.mlr.press/v80/yin18a.html) · [arXiv:1803.01498](https://arxiv.org/abs/1803.01498) |
| FedAtt | 2019 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Ji, Pan, Long, Li, Jiang, Huang. Learning Private Neural Language Modeling with Attentive Aggregation. IJCNN 2019 | [10.1109/IJCNN.2019.8852464](https://doi.org/10.1109/IJCNN.2019.8852464) |
| Krum | 2017 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Blanchard, El Mhamdi, Guerraoui, Stainer. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS 2017 | [NeurIPS proceedings](https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent) |
| Bulyan | 2018 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | El Mhamdi, Guerraoui, Rouault. The Hidden Vulnerability of Distributed Learning in Byzantium. ICML 2018, PMLR 80:3521–3530 | [PMLR v80](https://proceedings.mlr.press/v80/mhamdi18a.html) · [arXiv:1802.07927](https://arxiv.org/abs/1802.07927) |
| ChangeMagnitudeAvg | 2020 | [class](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/aggregation.py) | Muhammad et al. FedFast, KDD 2020, Algorithm 3 lines 5–12 | [10.1145/3394486.3403176](https://doi.org/10.1145/3394486.3403176) |

Secure aggregation: `Sum` needs only the sum of updates. The others need the individual updates (and FedAtt / ChangeMagnitudeAvg also need `previous_parameters`).

---

## FedAvg

Weighted average of client updates by `num_training_examples` for the selected clients. Works with single tensors and dict-of-tensor state dicts. For row-layout groups the server applies it per row over the clients that updated that row.

```python
from fedreclab.federated.aggregation import FedAvg

agg = FedAvg(num_clients=dataset.num_clients)
```

Used on FedNCF's dense groups in [`examples/quickstart.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/quickstart.ipynb).

---

## SimpleAvg

Uniform mean over the clients that contributed. No paper.

```python
from fedreclab.federated.aggregation import SimpleAvg

agg = SimpleAvg(num_clients=dataset.num_clients)
```

Used on FedNCF item rows in the quickstart. Prefer this when the selector already weights by data size.

---

## Sum

Unweighted sum of client updates. Ignores `previous_parameters` and `num_training_examples`. For row layout, a client that did not send a row contributes 0. Pairing with `exchange="gradient"` or `"delta"` is the experiment's job. This is the faithful FedMF aggregator; see [`methods.md`](methods.md#fedmf).

```python
from fedreclab.federated.aggregation import Sum

agg = Sum(num_clients=dataset.num_clients)
```

Also used by the popularity baseline in [`examples/implement_your_fed_rec.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/implement_your_fed_rec.ipynb).

---

## Median

Coordinate-wise median. No extra constructor arguments.

```python
from fedreclab.federated.aggregation import Median

agg = Median(num_clients=dataset.num_clients)
```

No notebook demonstrates this aggregator.

---

## TrimmedMean

Coordinate-wise trimmed mean. Drops `int(beta * n)` values from each end of each coordinate.

| Parameter | Default |
| --- | --- |
| `beta` | `0.1` (`0 <= beta < 0.5`) |

```python
from fedreclab.federated.aggregation import TrimmedMean

agg = TrimmedMean(num_clients=dataset.num_clients, beta=0.1)
```

No notebook demonstrates this aggregator.

---

## FedAtt

Layer-wise attentive aggregation. For each tensor, compute a softmax over client distances `||prev − update_i||_ord` (`ord=metric`) and return `prev − stepsize * sum_i att_i * (prev − update_i)`. Dict groups score each key independently (matching the official `shaoxiongji/fed-att` loop). Needs `previous_parameters`.

| Parameter | Default |
| --- | --- |
| `stepsize` | `1.0` |
| `metric` | `2` (Euclidean) |

**Deviation.** The optional Gaussian randomisation (paper Eq. 6) is not applied.

```python
from fedreclab.federated.aggregation import FedAtt

agg = FedAtt(num_clients=dataset.num_clients, stepsize=1.0, metric=2)
```

No notebook demonstrates this aggregator.

---

## Krum

Krum / Multi-Krum. Flattens each client update (tensor or sorted dict keys) to 1D for distance. Single-Krum returns one client update. Multi-Krum selects `m` updates iteratively then averages them with a uniform mean.

| Parameter | Default |
| --- | --- |
| `num_byzantine` | `0` |
| `multi_krum` | `False` |
| `num_selected` | `None` (used when `multi_krum=True`) |

**Implementation note.** The class docstring states that `n < 2f + 3` raises. The code instead falls back to a uniform average and sets `_last_call_fell_back = True`. FedVAE can use Multi-Krum as the paper's optional SecureFilter.

```python
from fedreclab.federated.aggregation import Krum

agg = Krum(num_clients=dataset.num_clients, num_byzantine=1, multi_krum=True)
```

No notebook demonstrates this aggregator.

---

## Bulyan

Krum-based candidate selection followed by a coordinate-wise trimmed mean on the candidates.

| Parameter | Default |
| --- | --- |
| `num_byzantine` | `0` |

When `n < 4f + 3`, the code falls back to a simple average (same pattern as Krum).

```python
from fedreclab.federated.aggregation import Bulyan

agg = Bulyan(num_clients=dataset.num_clients, num_byzantine=1)
```

No notebook demonstrates this aggregator.

---

## ChangeMagnitudeAvg

Weighted average by L2 change from the previous parameters. For each client update `u_i` and previous value `prev`, the weight is `q_i = ||u_i − prev||_2`. Clients with `q_i <= change_eps` are dropped; the remaining weights are renormalised and used for a weighted mean of the absolute updates. If every client is dropped, the previous value is kept. Not weighted by `num_training_examples`. Dict-of-tensor groups are scored per key independently. Needs individual updates and `previous_parameters`.

| Parameter | Default |
| --- | --- |
| `change_eps` | `0.0` |

FedFast installs this on item embeddings when `actv_agg=True`. Reclustering and subordinate user writes stay in the [FedFast](methods.md#fedfast) method helper, not in this class.

```python
from fedreclab.federated.aggregation import ChangeMagnitudeAvg

agg = ChangeMagnitudeAvg(num_clients=dataset.num_clients, change_eps=0.0)
```

No standalone notebook. Wired by FedFast.

---

## Writing a custom aggregator

Subclass `BaseAggregation`, implement `aggregate(self, round_info, previous_parameters, updates, param_io)`, and optionally override `aggregate_rows` for a vectorised row path. Pass one instance per ParamIO group that is exchanged.

**Worked example.** [`examples/custom_aggregator.ipynb`](https://github.com/aymenkhs/FedRecLab/blob/master/examples/custom_aggregator.ipynb) subclasses `BaseAggregation` as `RandomCoordinate`, checks a dense group and a row-layout group, and trains FedNCF with a per-group aggregator map.
