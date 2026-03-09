from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RoundStatistics:
    current_round: int
    selected: List[int]
    responded: List[int]
    dropped: List[int]
    num_training_examples: Dict[int, int]
    training_losses: Dict[int, List[float]]
    training_durations: Dict[int, float]


class ParamIO:
    def __init__(self, residency, exchange, apply_to):
        self.residency = residency
        self.exchange = exchange
        self.apply_to = apply_to


@dataclass
class UpdatePackage:
    param_name: str
    param_io: ParamIO
    param_value: Any
    param_metadata: dict
