from typing import List, Dict, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.fedreclab.federated.utils import ParamIO, UpdatePackage


@dataclass
class ClientStateBase:
    client_id: int
    local_parameters: Dict[str, Any]
    # Name-keyed optimiser moments ({param_name: {exp_avg, ...}}), never a live Optimizer.
    optimizer_state: Dict[str, Any] | None = None
    metadata: dict | None = None


@dataclass
class ClientTrainingRequest:
    client_id: int
    current_round: int
    users_in_client: List[int]
    client_state: ClientStateBase
    global_payload: dict
    training_data: pd.DataFrame
    validation_data: pd.DataFrame | None = None
    extra_payload: dict | None = None
    # Effective ParamIO map after experiment overrides. Clients read layout
    # from this and copy the objects into UpdatePackage; they do not build
    # their own ParamIO.
    param_ios: Dict[str, ParamIO] | None = None


@dataclass
class ClientTrainingResponse:
    client_id: int
    updated_client_state: ClientStateBase
    updates_packages: Dict[str, UpdatePackage]
    training_losses: List[float]
    training_duration: float
    num_training_examples: int
    validation_loss: float | None = None
    validation_metrics: Dict[str, float] | None = None
    extra_response: dict | None = None


class BaseClientAlgorithm:
    def __init__(self):
        pass

    def init_model_skeleton(self):
        raise NotImplementedError("Subclasses must implement this method")

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_train_data_view: pd.DataFrame,
        users_in_client: List[int],
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Any]]:
        raise NotImplementedError("Subclasses must implement this method")

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: Dict[int, np.ndarray],
        global_parameters: Dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: Dict[str, Any],
    ) -> Tuple[Dict[int, np.ndarray], float]:
        raise NotImplementedError("Subclasses must implement this method")

    def apply_privacy(
        self,
        training_request: ClientTrainingRequest,
        training_response: ClientTrainingResponse
    ) -> List[UpdatePackage]:
        pass

    def _update_package(
        self,
        training_request: ClientTrainingRequest,
        param_name: str,
        param_value: Any,
    ) -> UpdatePackage:
        """Build an update using the request's ParamIO, not a client-side copy."""
        param_ios = training_request.param_ios
        if not param_ios or param_name not in param_ios:
            raise KeyError(
                f"ClientTrainingRequest.param_ios is missing group {param_name!r}. "
                "The server must pass the effective ParamIO map on the request."
            )
        return UpdatePackage(
            param_name=param_name,
            param_io=param_ios[param_name],
            param_value=param_value,
            param_metadata={},
        )
