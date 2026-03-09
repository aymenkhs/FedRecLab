from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any

import numpy as np
import torch

from src.fedreclab.federated.utils import RoundStatistics, ParamIO


class BaseAggregation:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients

    def aggregate(
        self,
        round_info: RoundStatistics,
        previous_parameters: Dict[str, Any],
        updates: List[torch.Tensor] | List[np.ndarray],
        param_io: ParamIO,
    ):
        raise NotImplementedError("Subclasses must implement this method")

    def on_round_start(self, previous_round_statistics: RoundStatistics):
        pass

    def on_round_end(self, current_round_statistics: RoundStatistics):
        pass

