from typing import List, Dict, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.fedreclab.federated.utils import UpdatePackage


@dataclass
class ClientStateBase:
    client_id: int
    local_parameters: Dict[str, Any]
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
        client_dataset_view: pd.DataFrame,
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
