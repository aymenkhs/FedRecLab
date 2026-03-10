from typing import List, Callable, Dict, Any
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
from src.fedreclab.federated.utils import RoundStatistics, ParamIO

logger = logging.getLogger(__name__)


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
    ):
        self.interaction_dataset = interaction_dataset

        self.num_clients = interaction_dataset.num_clients
        self.num_rounds = num_rounds

        self.client_algorithm = client_algorithm

        self.param_ios = param_ios

        self.early_stopping = early_stopping
        self.early_stopping_custom = early_stopping_custom
        self.early_stopping_parameters = early_stopping_parameters
        self.early_stopping_callback = early_stopping_callback
        self._initialize_early_stopping()

        self.evaluator = evaluator
        self.validation_rounds_interval = validation_rounds_interval

        # Parse Client selection strategy
        if isinstance(client_selection_strategy, BaseClientSelection):
            self.client_selection_module = client_selection_strategy
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

        self.current_round_statistics = None
        self.previous_round_statistics = None

        self.client_state_class = client_state_class
        self.client_states = {}
        self._initialize_client_states()

        self.extra_payload = {}

        self.global_parameters = {}
        self.global_parameters = self._initialize_server_parameters()

        self.training_framework = training_framework
        self.device = device
        if self.training_framework == "torch" and self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_client_states(self):
        for client_id in range(self.num_clients):
            client_partition = self.interaction_dataset.client_partition
            users_in_client = client_partition.client_ids_2_user_ids[client_id]
            client_train_indices = self.interaction_dataset.get_client_view(client_id, include_validation=True, include_test=True)

            result = self.client_algorithm.initialize_local_parameters(
                client_id=client_id,
                total_num_users=self.interaction_dataset.num_users,
                num_users=len(users_in_client),
                num_items=self.interaction_dataset.num_items,
                client_dataset_view=client_train_indices,
                users_in_client=users_in_client,
            )
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
        If early stopping is enabled, stop training when the early stopping criteria is met.
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

            if self.early_stopping and self._early_stopping(current_round):
                logger.info(f"Early stopping triggered at round {current_round}")
                break

            self.previous_round_statistics = self.current_round_statistics

    def _before_training(self):
        pass

    def run_round(self, current_round: int):

        # Run start of round callbacks
        self._on_round_start_server()
        self.client_selection_module.on_round_start(self.previous_round_statistics)
        self.__aggregation_round_callback(on_round_start=True)

        # Select clients
        selected_client_ids = self.client_selection_module.select()

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
        validation_data = self.interaction_dataset.get_client_view(client_id, include_train=False, include_validation=True, include_test=False)
        training_request = ClientTrainingRequest(
            client_id=client_id,
            current_round=current_round,
            users_in_client=client_partition.client_ids_2_user_ids[client_id],
            client_state=client_state,
            training_data=training_data,
            validation_data=validation_data,
            global_payload=self._build_server_payload(client_id=client_id),
            extra_payload=self.extra_payload,
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
        for param_name, param_io in self.param_ios.items():
            if param_io.exchange is not None and param_io.exchange != "none":

                if param_io.residency == "server":
                    previous_parameters = self.global_parameters[param_name]
                elif param_io.residency == "client":
                    previous_parameters = [self.client_states[client_id].local_parameters[param_name] for client_id in self.current_round_statistics.selected]
                    previous_parameters = self.__convert_to_framework_compatible(previous_parameters)

                updates = [training_response.updates_packages[param_name].param_value for training_response in training_responses]
                updates = self.__convert_to_framework_compatible(updates)

                aggregated_results = self.aggregation_strategies[param_name].aggregate(
                    self.current_round_statistics,
                    previous_parameters,
                    updates,
                    param_io,
                )

                self._apply_aggregated_result(
                    param_name=param_name,
                    aggregated_result=aggregated_results,
                    param_io=param_io,
                    selected_client_ids=self.current_round_statistics.selected,
                )

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
        """
        Apply aggregated_result to old_value according to param_io.exchange.

        Supported conventions:
        - "params": aggregated_result is the new value
        - "delta": old_value + aggregated_result
        - "gradient": old_value - lr * aggregated_result
        - "none": return old_value unchanged

        Notes:
        - For "gradient", this assumes client returns raw gradients.
        - If your method returns optimizer-ready updates instead of raw gradients,
        prefer using exchange="delta".
        """
        exchange = getattr(param_io, "exchange", None)

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

    def _early_stopping(self, current_round: int,):
        if self.early_stopping_custom:
             return self.early_stopping_callback(current_round, self.early_stopping_parameters, self)
        else:
            return self._early_stopping_with_patience(current_round)

    def _initialize_early_stopping(self):
        if not self.early_stopping_custom:
            self.early_stopping_best_round = 0
            self._early_stopping_patience = 0

            if self.early_stopping_parameters is None:
                self.early_stopping_parameters = {}

            if self.early_stopping_parameters.get("metric") is None:
                self.early_stopping_parameters["metric"] = "loss"
            if self.early_stopping_parameters.get("threshold") is None:
                self.early_stopping_parameters["threshold"] = 0
            if self.early_stopping_parameters.get("patience") is None:
                self.early_stopping_parameters["patience"] = 5
            if self.early_stopping_parameters.get("direction") is None:
                if self.early_stopping_parameters["metric"] == "loss":
                    self.early_stopping_parameters["direction"] = "min"
                else:
                    self.early_stopping_parameters["direction"] = "max"

            if self.early_stopping_parameters["direction"] == "min":
                self.early_stopping_best_result = float('inf')
            else:
                self.early_stopping_best_result = float('-inf')

    def _early_stopping_with_patience(self, current_round: int):
        metric_name = self.early_stopping_parameters["metric"]
        direction = self.early_stopping_parameters["direction"]
        threshold = self.early_stopping_parameters["threshold"]

        current_value = self.current_metrics[metric_name]
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

    def __convert_to_framework_compatible(self, parameters: Any):
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
