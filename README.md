# FedRecLab

A PyTorch library for **simulating federated recommendation (FedRec) methods** under one data pipeline, one training loop and one evaluation protocol.

> **Status: pre-alpha.** APIs will change.

## Why

FedRec papers ship code in different frameworks, with different splits, client partitions and metrics, which makes baselines hard to reproduce and compare. FedRecLab re-implements published methods on a shared skeleton so that comparisons are fair, and lets you build a new method by subclassing two classes.

## What's inside

- **Data:** one interaction table as the single source of truth; ID remapping; bipartite k-core and minimum user/item interaction filters; ratio and stratified (optionally chronological) splits; cross-device client partitioning. Datasets via [Cornac](https://cornac.preferred.ai/) or any pandas DataFrame.
- **Federated core:** a round loop with hooks; per-parameter-group exchange rules (`ParamIO`); pluggable client selection and aggregation.
- **Aggregators:** FedAvg, SimpleAvg, Median, TrimmedMean, FedAtt, Krum / Multi-Krum, Bulyan. Row-layout groups (`ParamIO.layout="rows"`) apply the same aggregator per embedding row.
- **Evaluation:** full-ranking top-K metrics (HR, NDCG, Precision, Recall, …) computed per client.
- **Methods:** FedNCF, FedVAE, FedMF, GPFedRec, FedPerGNN, FedFast, EFVAE, MetaMF, SemiDFEGL.

## Install

FedRecLab is not on PyPI. Install it from GitHub. Python 3.10 or newer. The command installs the library and its pinned dependencies (NumPy, PyTorch, pandas, tqdm, Cornac). PyTorch is a large download.

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install "git+https://github.com/aymenkhs/FedRecLab.git"
```

Once that environment is active, `import fedreclab` works from any directory.

To work on this repository, clone it and install the package in editable mode, including pytest:

```bash
git clone https://github.com/aymenkhs/FedRecLab.git && cd FedRecLab
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start: FedNCF on MovieLens-100K

The same run, with a short explanation of each step, is in [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

```python
from fedreclab.data.load_data import load_cornac_dataset
from fedreclab.data.split import StratifiedSplitStrategy
from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from fedreclab.data.evaluator import FullRankingEvaluator
from fedreclab.federated.client_selection import RandomClientSelection
from fedreclab.federated.aggregation import FedAvg, SimpleAvg
from fedreclab.methods.fedncf.fedncf import FedNCF
from fedreclab.methods.fedncf.fedncf_client import FedNCFClientAlgorithm

dataset = load_cornac_dataset(
    dataset_name="movielens_100k",
    splitting_strategy=StratifiedSplitStrategy(
        include_validation=True, include_test=True,
        val_size=0.1, test_size=0.2, chrono=True, by="user"),
    client_partitioning_strategy=CrossDevicePartitionStrategy(),
    include_validation=True, include_test=True, seed=42,
)

client = FedNCFClientAlgorithm(predictive_factor=32, local_epochs=5, lr=5e-4,
                               batch_size=128, neg_samples=4)
evaluator = FullRankingEvaluator(metrics=["ndcg", "hr"], topk=10,
                                 interaction_dataset=dataset, client_algorithm=client)

server = FedNCF(
    predictive_factor=32,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=100,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.02, "frac"),
    aggregation_strategies={
        "item_embeddings": SimpleAvg(num_clients=dataset.num_clients),
        "shared_layers": FedAvg(num_clients=dataset.num_clients),
        "output_weights": FedAvg(num_clients=dataset.num_clients),
    },
    evaluator=evaluator,
    early_stopping=True,
    early_stopping_parameters={"metric": "ndcg@10", "direction": "max", "patience": 10},
)
server.run()

print(evaluator.evaluate(
    current_round=server.early_stopping_best_round,
    global_parameters=server.best_global_parameters,
    client_states=server.best_client_states,
    extra_payload=server.best_extra_payload,
    split="test",
))
```

Client algorithms, selectors, aggregators and evaluators are single-use: build new ones for every server. The dataset may be shared.

## Writing your own method

Subclass `BaseFederatedServer` and `BaseClientAlgorithm`, declare how each parameter group is exchanged with `ParamIO`, and put method-specific logic in the server hooks. Changing the loss of a method that is already in the library, or keeping some of its layers on the client, is shown for FedNCF in [`examples/extend_fedncf.ipynb`](examples/extend_fedncf.ipynb).

## Examples

- [`examples/quickstart.ipynb`](examples/quickstart.ipynb): FedNCF on MovieLens-100K (load, train, evaluate).
- [`examples/custom_data.ipynb`](examples/custom_data.ipynb): load a pandas DataFrame or Last.fm 2K as a custom dataset.
- [`examples/custom_aggregator.ipynb`](examples/custom_aggregator.ipynb): subclass `BaseAggregation` and plug it into FedNCF.
- [`examples/extend_fedncf.ipynb`](examples/extend_fedncf.ipynb): change FedNCF's loss, or keep layers on the client.
- [`examples/implement_your_fed_rec.ipynb`](examples/implement_your_fed_rec.ipynb): implement a new method from scratch (popularity baseline).

## Documentation

Component reference (tables, parameters, limitations, custom extension):

- [`docs/methods.md`](docs/methods.md): federated recommendation methods under `src/fedreclab/methods/`.
- [`docs/client_selection.md`](docs/client_selection.md): client sampling strategies.
- [`docs/aggregation.md`](docs/aggregation.md): aggregation strategies.
- [`docs/evaluator.md`](docs/evaluator.md): evaluation protocol and full-ranking evaluator.
- [`docs/partitioning.md`](docs/partitioning.md): client partition strategies.
- [`docs/splitting.md`](docs/splitting.md): train / validation / test split strategies.
- [`docs/metrics.md`](docs/metrics.md): Cornac ranking metrics and bias / fairness metrics.
- [`docs/datasets.md`](docs/datasets.md): Cornac datasets and custom DataFrame loading.

Also:

- `AGENTS.md`: conventions for contributors and coding agents.

## Roadmap

- **Server-side optimisers (FedOpt family).** Groups exchanged as
  `"gradient"` or `"delta"` are currently applied statelessly by the base
  server (`w - lr * g` or `w + delta`). We plan a generic server optimiser
  in `BaseFederatedServer` for those groups: server momentum (FedAvgM),
  FedAdagrad, FedAdam and FedYogi (Reddi et al., ICLR 2021), with
  configurable bias correction, per-row state for row-layout groups, and
  learning-rate schedules.

  Until then, any method can run its own server optimiser through the
  apply hook. Declare the group with
  `ParamIO(residency="server", exchange="gradient", apply_to="custom")`,
  keep the optimiser state (moments, step counter) on the server, and
  override `_apply_aggregated_result_custom` to turn the aggregated update
  into the new `self.global_parameters[param_name]`. The aggregator is
  unchanged and still swappable: it combines the client updates, and the
  hook decides how that combined update moves the model. See
  `src/fedreclab/methods/fedvae/fedvae.py` (server-side Adam on averaged
  client gradients) for a complete example.

- **Memory for full-model uploads (FedVAE).** Each selected client uploads
  a dense gradient of the whole model, and the round loop holds every
  upload before aggregating, so memory grows with
  `clients_per_round * num_items`. Planned fix: accumulate dense
  FedAvg/SimpleAvg/Sum without stacking, then stream mean-family
  aggregation as uploads arrive. Krum and Bulyan still need the
  individual updates.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).