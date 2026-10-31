# FedRecLab

A PyTorch library for **simulating federated recommendation (FedRec) methods** under one data pipeline, one training loop and one evaluation protocol.

> **Status: pre-alpha.** APIs will change. See `docs/issues/` for known problems.

## Why

FedRec papers ship code in different frameworks, with different splits, client partitions and metrics, which makes baselines hard to reproduce and compare. FedRecLab re-implements published methods on a shared skeleton so that comparisons are fair, and lets you build a new method by subclassing two classes.

## What's inside

- **Data:** one interaction table as the single source of truth; ID remapping; k-core filtering; ratio and stratified (optionally chronological) splits; cross-device client partitioning. Datasets via [Cornac](https://cornac.preferred.ai/) or any pandas DataFrame.
- **Federated core:** a round loop with hooks; per-parameter-group exchange rules (`ParamIO`); pluggable client selection and aggregation.
- **Aggregators:** FedAvg, SimpleAvg, Median, TrimmedMean, FedAtt, Krum / Multi-Krum, Bulyan.
- **Evaluation:** full-ranking top-K metrics (HR, NDCG, Precision, Recall, …) computed per client.
- **Methods:** FedNCF. FedPerGNN is being repaired; CdFed, FedMF, MetaMF, FedVAE, GPFedRec and SemiDFEGL are planned.

## Install

```bash
git clone <repo-url> FedRecLab && cd FedRecLab
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run code from the repository root (imports use `src.fedreclab`).

## Quick start: FedNCF on MovieLens-100K

```python
from src.fedreclab.data.load_data import load_cornac_dataset
from src.fedreclab.data.split import StratifiedSplitStrategy
from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.evaluator import FullRankingEvaluator
from src.fedreclab.federated.client_selection import RandomClientSelection
from src.fedreclab.federated.aggregation import SimpleAvg
from src.fedreclab.methods.fedncf.fedncf import FedNCF
from src.fedreclab.methods.fedncf.fedncf_client import FedNCFClientAlgorithm

dataset = load_cornac_dataset(
    dataset_name="movielens_100k",
    splitting_strategy=StratifiedSplitStrategy(
        include_validation=True, include_test=True,
        val_size=0.1, test_size=0.2, chrono=True, by="user"),
    client_partitioning_strategy=CrossDeviceSplittingStrategy(),
    include_validation=True, include_test=True, seed=42,
)

client = FedNCFClientAlgorithm(predictive_factor=32, local_epochs=5, lr=5e-4,
                               batch_size=128, neg_samples=3)
evaluator = FullRankingEvaluator(metrics=["ndcg", "hr"], topk=10,
                                 interaction_dataset=dataset, client_algorithm=client)

server = FedNCF(
    predictive_factor=32,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=100,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.02, "frac"),
    aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
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

## Writing your own method

Subclass `BaseFederatedServer` and `BaseClientAlgorithm`, declare how each parameter group is exchanged with `ParamIO`, and put method-specific logic in the server hooks. See `docs/PROJECT_CONTEXT.md` for the architecture and `AGENT.md` for conventions.

## Documentation

- `docs/PROJECT_CONTEXT.md`: architecture, design decisions, roadmap.
- `docs/issues/`: known issues (core, FedPerGNN, CdFed).
- `AGENT.md`: conventions for contributors and coding agents.

## Licence

TBD.