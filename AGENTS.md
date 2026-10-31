# AGENT.md — Working on FedRecLab

Instructions for coding agents (Cursor, Claude Code, Codex, …) working in this repository.

## 1. What this repository is
A PyTorch simulation library for federated recommendation research. The deliverable is the library itself: protocol-faithful baselines under one data pipeline, one round loop and one evaluation protocol. Correctness and clarity matter more than speed of feature delivery.

## 2. How to work with the maintainer
- **Discuss before building.** For any non-trivial change, first propose a plan: what changes, where, and why. Wait for approval before implementing. Long design discussions are expected and welcome.
- **Explain every decision.** When you implement, state each design decision and how it maps to what was agreed. If you deviate from an agreed plan, say so explicitly.
- **Prefer hooks over overrides.** Use the server/client hooks first. If you must override a pipeline method or change a base class, justify it in writing before doing it.
- **Do not add abstraction for its own sake.** No new base class unless a concrete second use exists. The maintainer has repeatedly rejected over-abstracted designs.
- **Do not run long experiments or test suites unless asked.** Make the change and let the maintainer run it. Quick import or syntax checks are fine.
- **Never edit plan files** (`*.plan.md`) unless asked; implement them as written.
- **Comments:** concise, explaining *why*, not restating code. Avoid clutter.
- **When the maintainer corrects a concept, restate it back** before continuing, to avoid repeating the misunderstanding (e.g. the FedPerGNN flags: `neighbor_embeddings_trainable` concerns only neighbour rows; the anchor user's embedding is always trainable).

## 3. Repository layout
```
src/fedreclab/
  data/        dataset.py, split.py, client_partitioning.py, load_data.py, evaluator.py
  federated/   base_server.py, base_client.py, aggregation.py, client_selection.py, utils.py
  methods/     one sub-package per method: <method>.py (server), <method>_client.py, <method>_model.py
docs/          PROJECT_CONTEXT.md, issues/
tests/         one folder per method (`tests/fedncf/`, `tests/fedpergnn/`, …);
               shared data/federated tests stay at `tests/` root.
```

Imports currently use the `src.fedreclab` root and must be run from the repository root. Do not add imports from scripts at the repository root into library code. A move to an installable `fedreclab` package is planned; do not start it without approval.

## 4. Architecture rules (non-negotiable unless revisited with the maintainer)
1. `InteractionDataset` is the single source of truth for interactions. IDs are remapped to contiguous 0-based integers there. Methods never re-map IDs or load data themselves.
2. One `BaseClientAlgorithm` object defines client behaviour; per-client data lives in `ClientStateBase`. Do not create per-client Python objects.
3. Client state holds **only client-resident data** (e.g. user embeddings, local heads, their optimiser state). Server-resident parameters arrive in `ClientTrainingRequest.global_payload` every round. Existing violations are tracked as debt; do not add new ones.
4. Every exchanged parameter group is declared with `ParamIO(residency, exchange, apply_to, layout="dense"|"rows")` by the method. Aggregators are experiment configuration, passed as one object or a `{group: aggregator}` dict. Row-layout groups upload `{rows, values}`; the server applies the configured aggregator per row.
5. **Methods must not special-case aggregator classes** (no `isinstance(agg, SomeAggregator)` in method code). Define the update format; any compatible aggregator must work.
6. Aggregators accept stacked arrays/tensors or lists of dict-of-tensors, compute in NumPy and return the input's type, dtype and device.
7. Evaluation is client-level: `compute_scores(client_id, candidate_items_per_user, global_parameters, client_state, extra_payload)`. Every candidate of a user is scored by the same function.
8. Training structures (graphs, histories, vocabularies) are built from **training data only**. Never use validation/test interactions outside evaluation.
9. Simulation shortcuts (e.g. a server using global knowledge in place of a privacy protocol) are allowed only if declared in the class docstring under "Simulation shortcuts".
10. Never silence numerical problems (`nan_to_num`, skipping non-finite losses). Fail loudly with a clear message.
11. Reproducibility: take randomness from seeded generators passed in from configuration, not from global `random` / `np.random` / unseeded torch.
12. Any Function or class added should have a full docstring with expected parameters and returns, expetions raised and examples. Follow numpy syle docstring. A reader should know what each function does when reading it.
13. Do not use 0 as a padding ID for items or users: 0 is a real ID after remapping. Shift indices or use explicit masks.

## 5. The round loop and hooks
See `docs/PROJECT_CONTEXT.md` §3.3 for the exact order. Hooks available to subclasses:

- **Server:** `_initialize_server_parameters` (required), `_before_training`, `_on_round_start_server`, `_post_selection_callback`, `_build_server_payload(client_id)`, `_update_statistics`, `_collect_updates`, `_aggregate_group`, `_apply_aggregated_result_custom`, `_on_round_end_server`.
- **Components:** client selection and aggregators have `on_round_start` / `on_round_end`.
- `self.current_round` and `self.selected_client_ids` are available inside all hooks.

## 6. Adding a method
1. Read the paper and any reference code. Write a short fidelity note: task (rating prediction or ranking), loss, what is exchanged, aggregation, privacy mechanisms, expansion, and defaults with their source.
2. Propose the `ParamIO` table and the hooks you will use. Get approval.
3. Implement `methods/<name>/<name>_model.py`, `<name>_client.py` (subclass `BaseClientAlgorithm`), `<name>.py` (subclass `BaseFederatedServer`).
4. Declare simulation shortcuts and deviations in docstrings.

## 7. Adding an aggregator, selector, split or partition strategy
Subclass the corresponding base class (`BaseAggregation`, `BaseClientSelection`, `BaseSplitStrategy`, `BaseClientPartitionStrategy`). Keep the constructor minimal. State the expected update format and what the aggregator needs to see (individual updates or only their sum; this matters for secure-aggregation compatibility).

## 8. Before you finish a task
- No new imports from repository-root scripts; no `isinstance` checks on aggregator classes in method code.
- No validation/test data used outside evaluation.
- Docstrings updated (including simulation shortcuts and deviations from the paper).
- The relevant `docs/issues/*.md` entry updated if you fixed or found an issue.
- Tests for a method go in `tests/<method>/`. Names describe the behaviour; do not use issue IDs.
- A summary of what changed, why, and what the maintainer should run to verify it.
