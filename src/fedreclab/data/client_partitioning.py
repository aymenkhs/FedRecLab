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


class CrossDeviceSplittingStrategy(BaseClientPartitionStrategy):
    """
    Each unique user is mapped to a distinct client.

    The number of clients is determined from the data as the number of
    unique user ids in the `u` column of `train_interactions`.
    """

    def __init__(self, user_col: str = "user_id", metadata: Dict[str, Any] | None = None) -> None:
        # num_clients is inferred during partition
        super().__init__(num_clients=0, user_col=user_col, metadata=metadata)

    def partition(self, train_interactions: pd.DataFrame) -> ClientPartition:
        if "user_id" not in train_interactions.columns:
            raise ValueError("train_interactions must contain a 'user_id' column with user identifiers")

        # Unique users define clients (one client per user)
        unique_users = train_interactions["user_id"].unique()
        num_clients = len(unique_users)

        # Map user ids to client ids (0-indexed)
        user_ids_2_client_ids: Dict[int, int] = {
            int(user_id): client_id for client_id, user_id in enumerate(unique_users)
        }

        # Inverse mapping: each client id has a single associated user id
        client_ids_2_user_ids: Dict[int, List[int]] = {
            client_id: [int(user_id)] for user_id, client_id in user_ids_2_client_ids.items()
        }

        # Group interactions by user to build index lists per client
        grouped = train_interactions.groupby("u").groups
        client_to_train_indices: Dict[int, List[int]] = {}
        client_train_sizes: Dict[int, int] = {}

        for user_id, indices in grouped.items():
            client_id = user_ids_2_client_ids[int(user_id)]
            index_list = list(indices)
            client_to_train_indices[client_id] = index_list
            client_train_sizes[client_id] = len(index_list)

        # Update num_clients on the strategy instance
        self.num_clients = num_clients

        return ClientPartition(
            num_clients=num_clients,
            client_ids_2_user_ids=client_ids_2_user_ids,
            user_ids_2_client_ids=user_ids_2_client_ids,
            client_to_train_indices=client_to_train_indices,
            client_train_sizes=client_train_sizes,
            client_to_val_indices=None,
            client_to_test_indices=None,
            metadata=self.metadata,
        )

