from typing import Dict, List, Any
from dataclasses import dataclass

import pandas as pd


@dataclass
class ClientPartition:
    num_clients: int
    client_ids_2_user_ids: Dict[int, List[int]]
    user_ids_2_client_ids: Dict[int, int]
    client_to_train_indices: Dict[int, List[int]]
    client_train_sizes: Dict[int, int]
    client_to_val_indices: Dict[int, List[int]]|None=None
    client_to_test_indices: Dict[int, List[int]]|None=None
    metadata: Dict[str, Any]|None=None


class BaseClientPartitionStrategy:
    def __init__(self, num_clients: int, user_col: str = "user_id", metadata: Dict[str, Any]|None=None):
        self.num_clients = num_clients
        self.user_col = user_col
        self.metadata = metadata

    def partition(self, train_interactions: pd.DataFrame) -> ClientPartition:
        raise NotImplementedError("Subclasses must implement this method")

