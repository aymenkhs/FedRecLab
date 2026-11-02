from typing import List, Callable, Dict, Any, Tuple
from dataclasses import replace
import copy
import logging

import torch
import numpy as np

from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.evaluator import Evaluator
from src.fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from src.fedreclab.federated.client_selection import BaseClientSelection
from src.fedreclab.federated.aggregation import BaseAggregation
from src.fedreclab.federated._compatibility_warnings import iter_selector_aggregator_warnings
from src.fedreclab.federated.utils import RoundStatistics, ParamIO, is_row_update

logger = logging.getLogger(__name__)

_ARITHMETIC_EXCHANGES = frozenset(
    {"delta", "gradient", "encrypted_delta", "encrypted_gradient"}
)


class BaseFederatedServer:
    def __init__(
        self,
        interaction_dataset: InteractionDataset,
        client_algorithm: BaseClientAlgorithm,
        num_rounds: int,
        param_ios: Dict[str, ParamIO],
        client_selection_strategy:BaseClientSelection,
        aggregation_strategies:BaseAggregation | Dict[str, BaseAggregation],
        client_state_class: type[ClientStateBase] = ClientStateBase,
        evaluator:Evaluator | None = None,
        validation_rounds_interval: int = 1,
        early_stopping: bool = False,
        early_stopping_custom: bool = False,
        early_stopping_parameters: dict | None = None,
        early_stopping_callback: Callable | None = None,
        training_framework: str = "torch",
        device: str | None = None,
        param_io_overrides: Dict[str, dict] | None = None,
        seed: int | None = None,
    ):
        self.interaction_dataset = interaction_dataset

        self.num_clients = interaction_dataset.num_clients
        self.num_rounds = num_rounds
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

        self.client_algorithm = client_algorithm

        # Resolve before init hooks so subclasses and the client can use them.
        self.training_framework = training_framework
        self.device = device
        if self.training_framework == "torch" and self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client_algorithm.device = self.device

        self.param_ios = param_ios
        if param_io_overrides:
            self._apply_param_io_overrides(param_io_overrides)

        self.early_stopping = early_stopping
        self.early_stopping_custom = early_stopping_custom
        self.early_stopping_parameters = early_stopping_parameters
        self.early_stopping_callback = early_stopping_callback
        self._initialize_early_stopping()

        # Scoring checkpoint for the best validation round. Live attributes
        # keep training; test eval after run() must use these copies.
        self.best_global_parameters: Dict[str, Any] | None = None
        self.best_client_states: Dict[int, ClientStateBase] | None = None
        self.best_extra_payload: Dict[str, Any] | None = None

        self.evaluator = evaluator
        self.validation_rounds_interval = validation_rounds_interval
        if self.early_stopping and not self.early_stopping_custom and self.evaluator is None:
            raise ValueError(
                "early_stopping=True requires an evaluator so patience can "
                "read a validation metric."
            )

        # Parse Client selection strategy
        if isinstance(client_selection_strategy, BaseClientSelection):
            self.client_selection_module = client_selection_strategy
            if self.client_selection_module.seed is None:
                self.client_selection_module.seed = self.seed
        else:
            raise ValueError(f"Invalid client selection strategy: {client_selection_strategy}")

        # Parse Aggregation strategy
        self.aggregation_strategies = {}
        if aggregation_strategies is None:
            raise ValueError("Aggregation strategy must be specified")
        elif isinstance(aggregation_strategies, BaseAggregation):
            for param_io_name, param_io in self.param_ios.items():
                if param_io.exchange is not None and param_io.exchange != "none":
                    self.aggregation_strategies[param_io_name] = aggregation_strategies
        elif isinstance(aggregation_strategies, dict):
            for paramters_name in aggregation_strategies:
                if paramters_name not in self.param_ios:
                    raise ValueError(f"Invalid parameter name: {paramters_name}")
                elif self.param_ios[paramters_name].exchange is None or self.param_ios[paramters_name].exchange == "none":
                    raise ValueError(f"Parameter {paramters_name} Cannot be Aggregated or exchanged with the server")
                else:
                    if isinstance(aggregation_strategies[paramters_name], BaseAggregation):
                        self.aggregation_strategies[paramters_name] = aggregation_strategies[paramters_name]
                    else:
                        raise ValueError(f"Invalid aggregation strategy: {aggregation_strategies[paramters_name]}")
        else:
            raise ValueError(f"Invalid aggregation strategies: {aggregation_strategies}")

        for message in iter_selector_aggregator_warnings(
            self.client_selection_module, self.aggregation_strategies.values()
        ):
            logger.warning(message)

        self.participation_counts = np.zeros(self.num_clients, dtype=np.int64)

        self.current_round_statistics = None
        self.previous_round_statistics = None

        # Current round index (set at the start of run_round).
        self.current_round: int | None = None

        # Selected clients for the current round (set in run_round after selection).
        # Exposed as an attribute so hooks can access it without changing signatures.
        self.selected_client_ids: List[int] | None = None

        self.client_state_class = client_state_class
        self.client_states = {}
        self._initialize_client_states()

        self.extra_payload = {}

        self.global_parameters = {}
        self.global_parameters = self._initialize_server_parameters()

    def _initialize_client_states(self):
        for client_id in range(self.num_clients):
            client_partition = self.interaction_dataset.client_partition
            users_in_client = client_partition.client_ids_2_user_ids[client_id]
            client_train_data_view = self.interaction_dataset.get_client_view(
                client_id, include_validation=False, include_test=False)

            result = self.client_algorithm.initialize_local_parameters(
                client_id=client_id,
                total_num_users=self.interaction_dataset.num_users,
                num_users=len(users_in_client),
                num_items=self.interaction_dataset.num_items,
                client_train_data_view=client_train_data_view,
                users_in_client=users_in_client,
            )

            # initialize_local_parameters may return:
            #   dict               -> local_parameters only
            #   (dict, state|None) -> local_parameters + initial optimizer state
            if isinstance(result, tuple):
                local_parameters, optimizer_state = result
            else:
                local_parameters = result
                optimizer_state = None

            self.client_states[client_id] = self.client_state_class(
                client_id=client_id,
                local_parameters=local_parameters,
                optimizer_state=optimizer_state,
            )
        return self.client_states

    def _initialize_server_parameters(self):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self):
        """
        Train the federated model for a given number of rounds.
        If validation is enabled, run validation at the specified interval.
        If early stopping is enabled, stop training when the early stopping
        criteria is met, and keep a scoring snapshot at the best validation
        round (``best_global_parameters``, ``best_client_states``,
        ``best_extra_payload``, ``early_stopping_best_round``). Test
        evaluation is the caller's job and should use that snapshot.
        """

        # Run before training callbacks
        self._before_training()

        for current_round in range(self.num_rounds):
            self.run_round(current_round=current_round)

            if self.evaluator is not None and current_round % self.validation_rounds_interval == 0:
                self.current_metrics = self.evaluator.evaluate(
                    current_round=current_round,
                    global_parameters=self.global_parameters,
                    client_states=self.client_states,
                    extra_payload=self.extra_payload,
                    split="val",
                )
                # Patience counts validation checks, not unmeasured rounds.
                if self.early_stopping and self._early_stopping(current_round):
                    logger.info(f"Early stopping triggered at round {current_round}")
                    break

            self.previous_round_statistics = self.current_round_statistics

    def _before_training(self):
        pass

    def _post_selection_callback(self):
        pass

    def _record_participation(self, selected_client_ids: List[int]) -> None:
        """Increment per-client selection counts and log a coverage summary.

        The INFO line reports how many clients were selected, the
        min/max/mean of ``participation_counts``, and how many clients
        have never been selected. The selected id list is logged at
        DEBUG so cross-device runs do not dump thousands of ids each
        round.

        Parameters
        ----------
        selected_client_ids : list of int
            Unique ids returned by this round's ``select``.

        Returns
        -------
        None
        """
        for cid in selected_client_ids:
            self.participation_counts[cid] += 1
        counts = self.participation_counts
        never = int(np.sum(counts == 0))
        logger.info(
            "Round %s selected %s clients; participation min/max/mean=%s/%s/%.3f; "
            "never selected=%s",
            self.current_round,
            len(selected_client_ids),
            int(counts.min()) if counts.size else 0,
            int(counts.max()) if counts.size else 0,
            float(counts.mean()) if counts.size else 0.0,
            never,
        )
        logger.debug(
            "Round %s selected client ids: %s",
            self.current_round,
            list(selected_client_ids),
        )

    def run_round(self, current_round: int):

        # Run start of round callbacks
        self.current_round = current_round
        self._on_round_start_server()
        self.client_selection_module.on_round_start(self.previous_round_statistics)
        self.__aggregation_round_callback(on_round_start=True)

        # Select clients
        selected_client_ids = self.client_selection_module.select(self.current_round)
        self.selected_client_ids = selected_client_ids
        self._record_participation(selected_client_ids)
        self._post_selection_callback()

        training_responses = []
        for client_id in selected_client_ids:
            training_response = self._train_client(client_id, current_round)
            training_responses.append(training_response)

        self.current_round_statistics = self._update_statistics(current_round, selected_client_ids, training_responses)

        # Aggregate paramters
        self.aggregate_parameters(training_responses)

        # Run end of round callbacks
        self.client_selection_module.on_round_end(self.current_round_statistics)
        self.__aggregation_round_callback(on_round_start=False)
        self._on_round_end_server()

    def _train_client(self, client_id: int, current_round: int):
        client_state = self.client_states[client_id]
        client_partition = self.interaction_dataset.client_partition
        training_data = self.interaction_dataset.get_client_view(client_id, include_validation=False, include_test=False)
        validation_data = None
        if getattr(self.interaction_dataset, "include_validation", False):
            validation_data = self.interaction_dataset.get_client_view(
                client_id,
                include_train=False,
                include_validation=True,
                include_test=False,
            )
        training_request = ClientTrainingRequest(
            client_id=client_id,
            current_round=current_round,
            users_in_client=client_partition.client_ids_2_user_ids[client_id],
            client_state=client_state,
            training_data=training_data,
            validation_data=validation_data,
            global_payload=self._build_server_payload(client_id=client_id),
            extra_payload=self.extra_payload,
            param_ios=self.param_ios,
            seed=self.seed,
        )
        training_response = self.client_algorithm.local_train(training_request)
        client_state = training_response.updated_client_state
        self.client_states[client_id] = client_state
        return training_response

    def _build_server_payload(self, client_id: int | None = None) -> Dict[str, Any]:
        """
        Build the payload sent from server to client according to self.param_ios.
        Only parameters whose residency is 'server' are included by default.
        If you later want client-specific filtering, use client_id.
        """
        server_payload = {}

        for param_name, param_io in self.param_ios.items():
            if param_io.residency == "server":
                if param_name not in self.global_parameters:
                    raise KeyError(f"Server parameter '{param_name}' missing from self.global_parameters")
                server_payload[param_name] = self.global_parameters[param_name]

        return server_payload

    def _update_statistics(self, current_round: int, selected_client_ids: List[int], training_responses: List[ClientTrainingResponse]):
        num_training_examples = {}
        training_losses = {}
        training_durations = {}
        for training_response in training_responses:
            client_id = training_response.client_id
            num_training_examples[client_id] = training_response.num_training_examples
            training_losses[client_id] = training_response.training_losses
            training_durations[client_id] = training_response.training_duration

        current_round_statistics = RoundStatistics(
            current_round=current_round,
            selected=selected_client_ids,
            responded=selected_client_ids,
            dropped=[],
            num_training_examples=num_training_examples,
            training_losses=training_losses,
            training_durations=training_durations,
        )

        return current_round_statistics

    def aggregate_parameters(
        self,
        training_responses: List[ClientTrainingResponse],
    ):
        """Aggregate each exchanged parameter group from the round's responses.

        Groups that no selected client sent are skipped. Unknown package names
        are rejected in ``_collect_updates``.

        Parameters
        ----------
        training_responses : list of ClientTrainingResponse
            Responses from the clients trained this round, in selection order.

        Returns
        -------
        None

        Examples
        --------
        Called from ``run_round`` after local training::

            self.aggregate_parameters(training_responses)
        """
        for param_name, param_io in self.param_ios.items():
            if param_io.exchange is None or param_io.exchange == "none":
                continue
            updates, participant_ids = self._collect_updates(
                param_name, param_io, training_responses
            )
            if not participant_ids:
                continue
            self._aggregate_group(param_name, param_io, updates, participant_ids)

    def _collect_updates(
        self,
        param_name: str,
        param_io: ParamIO,
        training_responses: List[ClientTrainingResponse],
    ) -> Tuple[List[Any], List[int]]:
        """Collect updates for one group, omitting clients that did not send it.

        Every package name on every response must be an exchanged ParamIO
        group. A misspelled name must not be ignored while the real group
        is silently skipped.

        Parameters
        ----------
        param_name : str
            ParamIO group to collect.
        param_io : ParamIO
            Declaration for ``param_name``. Unused by the default
            implementation; provided so overrides can branch on residency
            or exchange without looking the group up again.
        training_responses : list of ClientTrainingResponse
            Responses from the clients trained this round.

        Returns
        -------
        updates : list
            ``param_value`` of each package named ``param_name``, in response
            order.
        participant_ids : list of int
            Client ids aligned with ``updates``.

        Raises
        ------
        KeyError
            If a response contains a package whose name is not an exchanged
            ParamIO group.

        Examples
        --------
        Client 0 sends ``w``, client 1 does not::

            updates, ids = server._collect_updates("w", param_io, responses)
            # ids == [0]
        """
        exchanged = {
            name
            for name, pio in self.param_ios.items()
            if pio.exchange is not None and pio.exchange != "none"
        }
        updates: List[Any] = []
        participant_ids: List[int] = []
        for response in training_responses:
            unknown = sorted(
                name for name in response.updates_packages if name not in exchanged
            )
            if unknown:
                raise KeyError(
                    f"Client {response.client_id} sent update package(s) "
                    f"{unknown} that are not exchanged ParamIO groups "
                    f"{sorted(exchanged)}"
                )
            package = response.updates_packages.get(param_name)
            if package is None:
                continue
            updates.append(package.param_value)
            participant_ids.append(response.client_id)
        return updates, participant_ids

    def _round_statistics_for_participants(
        self,
        participant_ids: List[int],
    ) -> RoundStatistics:
        """Copy round statistics restricted to one group's participants.

        Uses ``dataclasses.replace`` so fields added to ``RoundStatistics``
        later are kept. Known client-keyed dicts are sliced by direct
        indexing.

        Parameters
        ----------
        participant_ids : list of int
            Clients that sent the group being aggregated.

        Returns
        -------
        RoundStatistics
            Same object as ``current_round_statistics`` except ``selected``
            and ``responded`` are ``participant_ids``, ``dropped`` is empty,
            and example/loss/duration maps contain only those clients.

        Raises
        ------
        KeyError
            If a participant is missing from a client-keyed statistics dict.
        RuntimeError
            If ``current_round_statistics`` has not been set.

        Examples
        --------
        >>> stats = server._round_statistics_for_participants([2, 5])
        >>> stats.selected
        [2, 5]
        """
        stats = self.current_round_statistics
        if stats is None:
            raise RuntimeError(
                "Per-group round statistics require current_round_statistics"
            )
        return replace(
            stats,
            selected=list(participant_ids),
            responded=list(participant_ids),
            dropped=[],
            num_training_examples={
                cid: stats.num_training_examples[cid] for cid in participant_ids
            },
            training_losses={
                cid: stats.training_losses[cid] for cid in participant_ids
            },
            training_durations={
                cid: stats.training_durations[cid] for cid in participant_ids
            },
        )

    def _aggregate_group(
        self,
        param_name: str,
        param_io: ParamIO,
        updates: List[Any],
        participant_ids: List[int],
    ) -> None:
        """Aggregate one exchanged group and apply the result.

        Override to special-case a single group without copying the loop
        over all ParamIO entries.

        Parameters
        ----------
        param_name : str
            ParamIO group being aggregated.
        param_io : ParamIO
            Declaration for ``param_name``.
        updates : list
            Per-participant values from ``_collect_updates``.
        participant_ids : list of int
            Clients aligned with ``updates``. Must be non-empty.

        Returns
        -------
        None

        Examples
        --------
        Default path used by ``aggregate_parameters``::

            self._aggregate_group("item_embeddings", param_io, updates, [0, 1])
        """
        self._assert_updates_match_layout(param_name, param_io, updates)

        if param_io.layout == "rows":
            if param_io.residency != "server":
                raise ValueError(
                    f"ParamIO.layout='rows' requires residency='server' "
                    f"(got {param_io.residency!r} for '{param_name}')"
                )
            previous_parameters = self.global_parameters[param_name]
            updated_table = self._aggregate_row_group(
                param_name, param_io, previous_parameters, updates, participant_ids
            )
            # Exchange is already applied per row; assign the table as params.
            apply_io = ParamIO(
                residency=param_io.residency,
                exchange="params",
                apply_to=param_io.apply_to,
                layout=param_io.layout,
            )
            self._apply_aggregated_result(
                param_name=param_name,
                aggregated_result=updated_table,
                param_io=apply_io,
                selected_client_ids=participant_ids,
            )
            return

        if param_io.residency == "server":
            previous_parameters = self.global_parameters[param_name]
        elif param_io.residency == "client":
            previous_parameters = [
                self.client_states[client_id].local_parameters[param_name]
                for client_id in participant_ids
            ]
            previous_parameters = self._convert_to_framework_compatible(
                previous_parameters
            )
        else:
            raise ValueError(
                f"Invalid ParamIO.residency={param_io.residency!r} for "
                f"parameter '{param_name}'"
            )

        updates = self._convert_to_framework_compatible(updates)
        aggregated_results = self.aggregation_strategies[param_name].aggregate(
            self._round_statistics_for_participants(participant_ids),
            previous_parameters,
            updates,
            param_io,
        )
        self._apply_aggregated_result(
            param_name=param_name,
            aggregated_result=aggregated_results,
            param_io=param_io,
            selected_client_ids=participant_ids,
        )

    def _apply_param_io_overrides(self, overrides: Dict[str, dict]) -> None:
        """Apply experiment-level field updates onto method ParamIO defaults."""
        allowed = {"residency", "exchange", "apply_to", "layout"}
        for name, fields in overrides.items():
            if name not in self.param_ios:
                raise ValueError(
                    f"param_io_overrides refers to unknown group {name!r}; "
                    f"known groups are {sorted(self.param_ios)}"
                )
            if not isinstance(fields, dict):
                raise TypeError(
                    f"param_io_overrides[{name!r}] must be a dict of ParamIO "
                    f"fields, got {type(fields).__name__}"
                )
            unknown = set(fields) - allowed
            if unknown:
                raise ValueError(
                    f"param_io_overrides[{name!r}] has unknown field(s) "
                    f"{sorted(unknown)}; allowed: {sorted(allowed)}"
                )
            pio = self.param_ios[name]
            for key, value in fields.items():
                setattr(pio, key, value)
            if pio.layout not in ("dense", "rows"):
                raise ValueError(
                    f"ParamIO.layout for '{name}' must be 'dense' or 'rows', "
                    f"got {pio.layout!r}"
                )

    def _assert_updates_match_layout(
        self,
        param_name: str,
        param_io: ParamIO,
        updates: List[Any],
    ) -> None:
        """Reject uploads that do not match ``param_io.layout``."""
        for update in updates:
            row_shaped = is_row_update(update)
            if param_io.layout == "rows" and not row_shaped:
                raise ValueError(
                    f"Group '{param_name}' has layout='rows' but an upload is "
                    f"not {{'rows', 'values'}} (got {type(update).__name__})"
                )
            if param_io.layout == "dense" and row_shaped:
                raise ValueError(
                    f"Group '{param_name}' has layout='dense' but an upload "
                    "uses the sparse {{'rows', 'values'}} format"
                )

    def _aggregate_row_group(
        self,
        param_name: str,
        param_io: ParamIO,
        previous_parameters: Any,
        updates: List[Any],
        participant_ids: List[int],
    ) -> Any:
        """Pack sparse uploads, aggregate union rows, apply exchange to those rows."""
        packed = self._pack_row_updates(param_name, previous_parameters, updates)
        if packed is None:
            return previous_parameters
        union_rows, values, mask, previous_rows = packed
        aggregator = self.aggregation_strategies[param_name]
        agg_rows = aggregator.aggregate_rows(
            self._round_statistics_for_participants(participant_ids),
            previous_rows,
            values,
            mask,
            param_io,
        )
        fallbacks = int(getattr(aggregator, "row_fallback_count", 0) or 0)
        if fallbacks:
            stats = self.current_round_statistics
            if stats is not None:
                stats.row_aggregator_fallbacks[param_name] = fallbacks
        return self._scatter_exchanged_rows(
            param_name, param_io, previous_parameters, union_rows, agg_rows
        )

    def _pack_row_updates(
        self,
        param_name: str,
        previous_parameters: Any,
        updates: List[Any],
    ):
        """Build ``(n, k_union, d)`` values and mask from sparse uploads.

        Returns None when every upload is empty (nothing to write).
        """
        parsed: List[tuple[np.ndarray, Any]] = []
        for i, update in enumerate(updates):
            rows = np.asarray(update["rows"], dtype=np.int64).reshape(-1)
            values = update["values"]
            if rows.size != len(np.unique(rows)):
                raise ValueError(
                    f"Group '{param_name}' upload {i} has duplicate row ids"
                )
            parsed.append((rows, values))

        n_table, table_keys, prev_np = self._row_table_numpy(previous_parameters)
        for i, (rows, _) in enumerate(parsed):
            if rows.size and (np.any(rows < 0) or np.any(rows >= n_table)):
                raise IndexError(
                    f"Group '{param_name}' upload {i} has row ids outside "
                    f"[0, {n_table})"
                )

        union = np.unique(np.concatenate([r for r, _ in parsed] + [np.array([], dtype=np.int64)]))
        if union.size == 0:
            return None
        col_of = {int(r): j for j, r in enumerate(union)}
        n_clients = len(parsed)
        k = int(union.size)
        mask = np.zeros((n_clients, k), dtype=bool)

        first_values = parsed[0][1]
        values_are_dict = isinstance(first_values, dict)
        if values_are_dict:
            if table_keys is None:
                raise TypeError(
                    f"Group '{param_name}' previous table is a tensor but "
                    "uploads use dict values"
                )
            if set(first_values) != set(table_keys):
                raise ValueError(
                    f"Group '{param_name}' upload value keys {sorted(first_values)} "
                    f"do not match table keys {sorted(table_keys)}"
                )
            stacked = {}
            for key in table_keys:
                d = int(prev_np[key].shape[1]) if prev_np[key].ndim > 1 else 1
                stacked[key] = np.zeros((n_clients, k, d), dtype=np.float64)
            for i, (rows, values) in enumerate(parsed):
                if not isinstance(values, dict) or set(values) != set(table_keys):
                    raise ValueError(
                        f"Group '{param_name}' upload {i} value keys do not "
                        f"match {sorted(table_keys)}"
                    )
                if rows.size == 0:
                    continue
                cols = np.array([col_of[int(r)] for r in rows], dtype=np.int64)
                mask[i, cols] = True
                for key in table_keys:
                    arr = np.asarray(values[key], dtype=np.float64)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, stacked[key].shape[-1])
                    if arr.shape[0] != rows.size:
                        raise ValueError(
                            f"Group '{param_name}' upload {i} key {key!r} has "
                            f"{arr.shape[0]} rows, expected {rows.size}"
                        )
                    stacked[key][i, cols] = arr
            prev_at = {key: prev_np[key][union] for key in table_keys}
            return union, stacked, mask, prev_at

        if table_keys is not None:
            raise TypeError(
                f"Group '{param_name}' previous table is a dict but uploads "
                "use a single tensor for values"
            )
        d = int(prev_np.shape[1]) if prev_np.ndim > 1 else 1
        stacked_arr = np.zeros((n_clients, k, d), dtype=np.float64)
        for i, (rows, values) in enumerate(parsed):
            if isinstance(values, dict):
                raise TypeError(
                    f"Group '{param_name}' upload {i} uses dict values; "
                    "expected a single tensor"
                )
            if rows.size == 0:
                continue
            cols = np.array([col_of[int(r)] for r in rows], dtype=np.int64)
            mask[i, cols] = True
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, d)
            if arr.shape[0] != rows.size:
                raise ValueError(
                    f"Group '{param_name}' upload {i} has {arr.shape[0]} "
                    f"value rows, expected {rows.size}"
                )
            stacked_arr[i, cols] = arr
        return union, stacked_arr, mask, prev_np[union]

    def _row_table_numpy(self, table: Any):
        """Return (n_rows, dict_keys_or_None, numpy table)."""
        if isinstance(table, dict):
            keys = sorted(table.keys())
            converted = {k: np.asarray(table[k]) for k in keys}
            n_rows = int(converted[keys[0]].shape[0])
            for k in keys[1:]:
                if int(converted[k].shape[0]) != n_rows:
                    raise ValueError(
                        "Dict table tensors have mismatched leading dimensions"
                    )
            return n_rows, keys, converted
        arr = np.asarray(table)
        return int(arr.shape[0]), None, arr

    def _scatter_exchanged_rows(
        self,
        param_name: str,
        param_io: ParamIO,
        previous_parameters: Any,
        union_rows: np.ndarray,
        agg_rows: Any,
    ) -> Any:
        """``table[rows] = apply_exchange(table[rows], agg_rows)``; other rows untouched."""
        if isinstance(previous_parameters, dict):
            out = {}
            for key, old in previous_parameters.items():
                out[key] = self._scatter_exchanged_rows(
                    param_name, param_io, old, union_rows, agg_rows[key]
                )
            return out
        old_slice = self._take_rows(previous_parameters, union_rows)
        new_slice = self._apply_exchange_to_value(
            old_value=old_slice,
            aggregated_result=self._coerce_like(agg_rows, old_slice),
            param_io=param_io,
            param_name=param_name,
        )
        return self._put_rows(previous_parameters, union_rows, new_slice)

    @staticmethod
    def _take_rows(table: Any, rows: np.ndarray):
        if isinstance(table, torch.Tensor):
            idx = torch.as_tensor(rows, dtype=torch.long, device=table.device)
            return table[idx]
        return np.asarray(table)[np.asarray(rows, dtype=np.int64)]

    @staticmethod
    def _put_rows(table: Any, rows: np.ndarray, new_rows: Any):
        if isinstance(table, torch.Tensor):
            out = table.clone()
            idx = torch.as_tensor(rows, dtype=torch.long, device=table.device)
            out[idx] = torch.as_tensor(new_rows, dtype=table.dtype, device=table.device)
            return out
        out = np.array(table, copy=True)
        out[np.asarray(rows, dtype=np.int64)] = np.asarray(new_rows)
        return out

    @staticmethod
    def _coerce_like(value: Any, like: Any):
        if isinstance(like, torch.Tensor):
            return torch.as_tensor(value, dtype=like.dtype, device=like.device)
        return np.asarray(value)

    def _apply_aggregated_result(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: List[int],
    ):
        """
        Apply aggregated results according to param_io.apply_to.

        Suggested conventions for apply_to:
        - "server": update self.global_parameters[param_name]
        - "clients": update selected clients' local_parameters[param_name]
        - "both": update both server and selected clients
        - "custom": delegate to _apply_aggregated_result_custom(...)
        """
        apply_to = getattr(param_io, "apply_to", None)

        if apply_to is None:
            raise ValueError(f"ParamIO.apply_to is missing for parameter '{param_name}'")

        if apply_to == "server":
            self._update_server_parameter(
                param_name=param_name,
                aggregated_result=aggregated_result,
                param_io=param_io,
            )
        elif apply_to == "clients":
            self._update_client_states_parameter(
                param_name=param_name,
                aggregated_result=aggregated_result,
                param_io=param_io,
                selected_client_ids=selected_client_ids,
            )
        elif apply_to == "both":
            self._update_server_parameter(
                param_name=param_name,
                aggregated_result=aggregated_result,
                param_io=param_io,
            )
            self._update_client_states_parameter(
                param_name=param_name,
                aggregated_result=aggregated_result,
                param_io=param_io,
                selected_client_ids=selected_client_ids,
            )
        elif apply_to == "custom":
            self._apply_aggregated_result_custom(
                param_name=param_name,
                aggregated_result=aggregated_result,
                param_io=param_io,
                selected_client_ids=selected_client_ids,
            )
        else:
            raise ValueError(
                f"Invalid ParamIO.apply_to='{apply_to}' for parameter '{param_name}'. "
                "Expected one of: 'server', 'clients', 'both', 'custom'."
        )

    def _update_server_parameter(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
    ):
        """
        Update a server/global parameter according to param_io.exchange.
        """
        if param_name not in self.global_parameters:
            raise KeyError(f"Server parameter '{param_name}' missing from self.global_parameters")

        old_value = self.global_parameters[param_name]
        new_value = self._apply_exchange_to_value(
            old_value=old_value,
            aggregated_result=aggregated_result,
            param_io=param_io,
            param_name=param_name,
        )
        self.global_parameters[param_name] = new_value

    def _update_client_states_parameter(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: List[int],
    ):
        """
        Update selected clients' local parameter named param_name according to param_io.exchange.

        Accepted aggregated_result formats:
        - dict[client_id -> value]
        - list/tuple aligned with selected_client_ids
        - single shared value broadcast to all selected clients
        """
        if isinstance(aggregated_result, dict):
            for client_id in selected_client_ids:
                if client_id not in aggregated_result:
                    raise KeyError(
                        f"Aggregated result for parameter '{param_name}' missing client_id={client_id}"
                    )

                old_value = self.client_states[client_id].local_parameters[param_name]
                new_value = self._apply_exchange_to_value(
                    old_value=old_value,
                    aggregated_result=aggregated_result[client_id],
                    param_io=param_io,
                    param_name=param_name,
                )
                self.client_states[client_id].local_parameters[param_name] = new_value
        elif isinstance(aggregated_result, (list, tuple)):
            if len(aggregated_result) != len(selected_client_ids):
                raise ValueError(
                    f"Aggregated result for parameter '{param_name}' has length {len(aggregated_result)} "
                    f"but there are {len(selected_client_ids)} selected clients"
                )

            for idx, client_id in enumerate(selected_client_ids):
                old_value = self.client_states[client_id].local_parameters[param_name]
                new_value = self._apply_exchange_to_value(
                    old_value=old_value,
                    aggregated_result=aggregated_result[idx],
                    param_io=param_io,
                    param_name=param_name,
                )
                self.client_states[client_id].local_parameters[param_name] = new_value
        else:
            # broadcast same value to all selected clients
            for client_id in selected_client_ids:
                old_value = self.client_states[client_id].local_parameters[param_name]
                new_value = self._apply_exchange_to_value(
                    old_value=old_value,
                    aggregated_result=aggregated_result,
                    param_io=param_io,
                    param_name=param_name,
                )
                self.client_states[client_id].local_parameters[param_name] = new_value

    def _apply_aggregated_result_custom(
        self,
        param_name: str,
        aggregated_result: Any,
        param_io: ParamIO,
        selected_client_ids: List[int],
    ):
        """
        Override in subclasses for methods that need non-standard application logic,
        e.g. updating a server-side memory bank, expanded user embedding pool,
        encrypted buffers, cluster parameters, etc.
        """
        raise NotImplementedError(
            f"Custom aggregated result application not implemented for parameter '{param_name}'"
        )

    def _apply_exchange_to_value(
        self,
        old_value: Any,
        aggregated_result: Any,
        param_io: ParamIO,
        param_name: str,
    ):
        """Apply an aggregated result to a parameter value per ``param_io.exchange``.

        Conventions:

        - ``"params"`` / ``"encrypted_params"``: the result is the new value.
        - ``"delta"`` / ``"encrypted_delta"``: ``old_value + result``.
        - ``"gradient"`` / ``"encrypted_gradient"``: ``old_value - lr * result``,
          with ``lr`` from ``_get_server_update_lr``. Clients must upload raw
          gradients; optimiser-ready steps should use ``"delta"``.
        - ``"none"``: ``old_value`` is returned unchanged.

        For the delta and gradient conventions, dict-of-tensor groups (e.g.
        a state-dict slice) are applied key by key with the same rule and
        learning rate. Params and none conventions do not inspect structure.

        Parameters
        ----------
        old_value : array-like, torch.Tensor or dict
            Current value of the group. A dict maps parameter names to
            tensors or arrays.
        aggregated_result : array-like, torch.Tensor or dict
            Aggregator output. Must be a dict with the same keys when
            ``old_value`` is a dict.
        param_io : ParamIO
            Declaration of the group; only ``exchange`` is read here.
        param_name : str
            Group name, used for the learning-rate lookup and error messages.

        Returns
        -------
        array-like, torch.Tensor or dict
            Updated value, with the same structure as ``old_value``.

        Raises
        ------
        TypeError
            If, for a delta or gradient exchange, exactly one of
            ``old_value`` and ``aggregated_result`` is a dict.
        ValueError
            If, for a delta or gradient exchange, dict keys differ; or if
            ``exchange`` is not a known convention.

        Examples
        --------
        >>> io = ParamIO(residency="server", exchange="delta", apply_to="server")
        >>> server._apply_exchange_to_value(
        ...     {"w": np.ones(2)}, {"w": np.ones(2)}, io, "g"
        ... )
        {'w': array([2., 2.])}
        """
        exchange = getattr(param_io, "exchange", None)

        if exchange in _ARITHMETIC_EXCHANGES and (
            isinstance(old_value, dict) or isinstance(aggregated_result, dict)
        ):
            if not (isinstance(old_value, dict) and isinstance(aggregated_result, dict)):
                raise TypeError(
                    f"Group '{param_name}': cannot apply exchange='{exchange}' "
                    f"between {type(old_value).__name__} (current value) and "
                    f"{type(aggregated_result).__name__} (aggregated result)"
                )
            if set(old_value) != set(aggregated_result):
                raise ValueError(
                    f"Group '{param_name}': aggregated keys "
                    f"{sorted(aggregated_result)} do not match parameter keys "
                    f"{sorted(old_value)}"
                )
            return {
                key: self._apply_exchange_to_value(
                    old_value=old_value[key],
                    aggregated_result=aggregated_result[key],
                    param_io=param_io,
                    param_name=param_name,
                )
                for key in old_value
            }

        if exchange == "params":
            return aggregated_result
        elif exchange == "delta":
            return old_value + aggregated_result
        elif exchange == "gradient":
            lr = self._get_server_update_lr(param_name=param_name, param_io=param_io)
            return old_value - lr * aggregated_result
        elif exchange == "encrypted_params":
            return aggregated_result
        elif exchange == "encrypted_delta":
            return old_value + aggregated_result
        elif exchange == "encrypted_gradient":
            lr = self._get_server_update_lr(param_name=param_name, param_io=param_io)
            return old_value - lr * aggregated_result
        elif exchange == "none":
            return old_value
        else:
            raise ValueError(
                f"Invalid exchange='{exchange}' for parameter '{param_name}'"
            )

    def _get_server_update_lr(
        self,
        param_name: str,
        param_io: ParamIO,
    ):
        """
        Learning rate used when applying aggregated gradients on the server/client side.

        Resolution order:
        1. self.server_learning_rates[param_name]
        2. self.server_learning_rate
        3. default = 1.0
        """
        if hasattr(self, "server_learning_rates") and self.server_learning_rates is not None:
            if param_name in self.server_learning_rates:
                return self.server_learning_rates[param_name]

        if hasattr(self, "server_learning_rate") and self.server_learning_rate is not None:
            return self.server_learning_rate

        return 1.0

    def __aggregation_round_callback(self, on_round_start: bool):
        for aggregation_strategy in self.aggregation_strategies.values():
            if on_round_start:
                aggregation_strategy.on_round_start(self.previous_round_statistics)
            else:
                aggregation_strategy.on_round_end(self.current_round_statistics)

    def _on_round_start_server(self):
        pass

    def _on_round_end_server(self):
        pass

    def _snapshot_best_state(self, current_round: int) -> None:
        """Deep-copy scoring inputs at the current best validation round.

        Live ``global_parameters``, ``client_states`` and ``extra_payload``
        continue to be updated in later rounds. Test evaluation after
        ``run()`` must use these copies so it scores the best validation
        round rather than the stop round.

        Parameters
        ----------
        current_round : int
            Round whose validation metric just became the best so far.

        Returns
        -------
        None

        Examples
        --------
        Called from ``_early_stopping_with_patience`` on improvement::

            self._snapshot_best_state(current_round)
        """
        self.early_stopping_best_round = current_round
        self.best_global_parameters = copy.deepcopy(self.global_parameters)
        self.best_client_states = copy.deepcopy(self.client_states)
        self.best_extra_payload = copy.deepcopy(self.extra_payload)

    def _early_stopping(self, current_round: int,):
        if self.early_stopping_custom:
             return self.early_stopping_callback(current_round, self.early_stopping_parameters, self)
        else:
            return self._early_stopping_with_patience(current_round)

    def _initialize_early_stopping(self):
        """Fill patience defaults; require an explicit ranking metric when stopping.

        Ranking evaluation does not produce a loss, so ``metric`` cannot
        default to ``"loss"``. When patience early stopping is on,
        ``early_stopping_parameters["metric"]`` must name a key that
        ``evaluate`` stores (for example ``"ndcg@10"``). Direction
        defaults to ``"max"``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``early_stopping`` is True, ``early_stopping_custom`` is
            False, and ``metric`` is missing or is ``"loss"``.

        Examples
        --------
        With ``early_stopping=True`` and ``{"metric": "ndcg@10"}``,
        direction defaults to ``"max"`` and patience to ``5``.
        """
        if self.early_stopping_custom:
            return

        self.early_stopping_best_round = None
        self._early_stopping_patience = 0

        if self.early_stopping_parameters is None:
            self.early_stopping_parameters = {}

        metric = self.early_stopping_parameters.get("metric")
        if self.early_stopping:
            if not metric:
                raise ValueError(
                    "early_stopping=True requires early_stopping_parameters['metric'] "
                    "to name a ranking result (for example 'ndcg@10'). Ranking "
                    "evaluation does not produce a loss."
                )
            if metric == "loss":
                raise ValueError(
                    "early_stopping metric 'loss' is not produced by ranking "
                    "evaluation. Pass a stored ranking key such as 'ndcg@10'."
                )

        if self.early_stopping_parameters.get("threshold") is None:
            self.early_stopping_parameters["threshold"] = 0
        if self.early_stopping_parameters.get("patience") is None:
            self.early_stopping_parameters["patience"] = 5
        if self.early_stopping_parameters.get("direction") is None:
            self.early_stopping_parameters["direction"] = "max"

        if self.early_stopping_parameters["direction"] == "min":
            self.early_stopping_best_result = float("inf")
        else:
            self.early_stopping_best_result = float("-inf")

    def _early_stopping_metric_value(self) -> float:
        """Return the monitored validation metric as a comparable scalar.

        Looks up ``early_stopping_parameters["metric"]`` in
        ``current_metrics``. Top-K names are stored as ``"{name}@{k}"``.
        If the entry is a confidence-interval dict, the ``mean`` is used.

        Returns
        -------
        float
            Scalar used for patience comparison.

        Raises
        ------
        KeyError
            If the configured metric is not in the validation results.
        TypeError
            If the stored value is not a scalar (or a CI dict with ``mean``).
        ValueError
            If the scalar is not finite.

        Examples
        --------
        After validation with ``current_metrics["ndcg@10"] == 0.4`` and
        ``early_stopping_parameters["metric"] == "ndcg@10"``, this returns
        ``0.4``. A CI entry ``{"mean": 0.4, ...}`` also returns ``0.4``.
        """
        metric_name = self.early_stopping_parameters["metric"]
        metrics = getattr(self, "current_metrics", None)
        if metrics is None:
            raise RuntimeError(
                "Early stopping ran without validation metrics. Pass an "
                "evaluator and validate before checking patience."
            )
        if metric_name not in metrics:
            available = sorted(metrics)
            raise KeyError(
                f"Early stopping metric {metric_name!r} is not in the "
                f"validation results {available}. Top-K metrics are stored "
                f"as '{{name}}@{{k}}' (for example 'ndcg@10')."
            )
        value = metrics[metric_name]
        if isinstance(value, dict):
            if "mean" not in value:
                raise TypeError(
                    f"Early stopping metric {metric_name!r} is a dict without "
                    f"'mean' (got keys {sorted(value)}). When "
                    "compute_confidence_intervals=True, the mean is used."
                )
            value = value["mean"]
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Early stopping metric {metric_name!r} must be a scalar, "
                f"got {type(value).__name__}: {value!r}"
            ) from exc
        if not np.isfinite(scalar):
            raise ValueError(
                f"Early stopping metric {metric_name!r} is not finite: {scalar}"
            )
        return scalar

    def _early_stopping_with_patience(self, current_round: int):
        metric_name = self.early_stopping_parameters["metric"]
        direction = self.early_stopping_parameters["direction"]
        threshold = self.early_stopping_parameters["threshold"]

        current_value = self._early_stopping_metric_value()
        best_value = self.early_stopping_best_result

        # Determine whether the monitored metric has improved by more than
        # the threshold. For "min", improvement == lower; for "max", higher.
        if direction == "min":
            improved = current_value < (best_value - threshold)
        elif direction == "max":
            improved = current_value > (best_value + threshold)
        else:
            raise ValueError(f"Invalid early stopping direction: {direction}")

        if improved:
            # Reset patience and update best result on improvement.
            if self._early_stopping_patience > 0:
                logger.info(f"Early stopping patience reset at round {current_round}")
            self._early_stopping_patience = 0
            self.early_stopping_best_result = current_value
            self._snapshot_best_state(current_round)
        else:
            # No sufficient improvement: increase patience counter.
            self._early_stopping_patience += 1
            logger.info(
                f"Early stopping patience incremented at round {current_round} "
                f"to {self._early_stopping_patience}. Current {metric_name}: "
                f"{current_value}, Best {metric_name}: {best_value}"
            )

        if self._early_stopping_patience >= self.early_stopping_parameters["patience"]:
            return True

        return False

    def _convert_to_framework_compatible(self, parameters: Any):
        """Convert stacked or listed parameter values to ``training_framework``.

        Dict-of-tensor groups stay a list of dicts (they are not stacked).
        Dense lists are stacked into one array or tensor.

        Parameters
        ----------
        parameters : list or array-like
            Per-client updates or previous values. A list of dicts is treated
            as state-dict groups; any other list is stacked.

        Returns
        -------
        Any
            Values on ``self.device`` if ``training_framework`` is ``"torch"``,
            or NumPy arrays if it is ``"numpy"``.

        Raises
        ------
        ValueError
            If ``training_framework`` is not ``"torch"`` or ``"numpy"``.

        Examples
        --------
        >>> server.training_framework = "numpy"
        >>> server._convert_to_framework_compatible(
        ...     [np.ones(2), np.zeros(2)]
        ... ).shape
        (2, 2)
        """
        # Dict-of-tensor values (state-dict parameter groups): convert each
        # value inside the dicts to the target framework, return as list of dicts.
        if isinstance(parameters, list) and len(parameters) > 0 and isinstance(parameters[0], dict):
            if self.training_framework == "torch":
                return [
                    {k: torch.as_tensor(v).to(self.device) if not isinstance(v, torch.Tensor) else v.to(self.device)
                     for k, v in d.items()}
                    for d in parameters
                ]
            elif self.training_framework == "numpy":
                return [
                    {k: np.asarray(v) for k, v in d.items()}
                    for d in parameters
                ]
            return parameters

        if self.training_framework == "torch":
            if isinstance(parameters, list):
                converted = [torch.as_tensor(p).to(self.device) if not isinstance(p, torch.Tensor) else p.to(self.device) for p in parameters]
                return torch.stack(converted)
            return torch.as_tensor(parameters).to(self.device)
        elif self.training_framework == "numpy":
            if isinstance(parameters, list):
                return np.stack([np.asarray(p) for p in parameters])
            return np.asarray(parameters)
        else:
            raise ValueError(f"Invalid training framework: {self.training_framework}")
