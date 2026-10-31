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
    seed: int | None = None


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
    """One algorithm object shared by every client.

    ``device`` is simulation runtime config. The server writes
    ``client_algorithm.device = server.device`` before
    ``initialize_local_parameters``. Standalone callers must set
    ``device`` themselves.
    """

    def __init__(self, device: str | None = None):
        """Store the simulation device.

        Parameters
        ----------
        device : str or None, optional
            Torch device string such as ``"cpu"`` or ``"cuda"``. ``None``
            until the server or a standalone caller assigns it.

        Returns
        -------
        None

        Examples
        --------
        >>> BaseClientAlgorithm(device="cpu").device
        'cpu'
        """
        self.device = device

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
    ) -> Dict[int, np.ndarray]:
        """Score every candidate item for each user on this client.

        Ranking evaluation calls this hook and ranks the returned scores
        against held-out positives. It does not compute a training loss:
        ``compute_scores`` is not given labels, ratings, or sampled
        negatives.

        Parameters
        ----------
        client_id : int
            Client being scored.
        candidate_items_per_user : dict of int to ndarray
            Item ids to score, keyed by remapped user id.
        global_parameters : dict
            Server-resident tensors for this round.
        client_state : ClientStateBase
            Client-resident tensors (user embeddings, and so on).
        extra_payload : dict
            Method-specific extras (for example expanded neighbour state).

        Returns
        -------
        dict of int to ndarray
            One score array per user, aligned with that user's candidate
            item ids. Higher is better.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> scores = client.compute_scores(
        ...     0, {0: np.array([1, 2])}, payload, state, {}
        ... )
        >>> scores[0].shape
        (2,)
        """
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
