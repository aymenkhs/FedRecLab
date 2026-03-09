import pandas as pd

from src.fedreclab.data.split import BaseSplitStrategy
from src.fedreclab.data.client_partitioning import BaseClientPartitionStrategy


class InteractionDataset:

    def __init__(
        self,
        interaction_data: pd.DataFrame,
        splitting_strategy: BaseSplitStrategy,
        client_partitioning_strategy: BaseClientPartitionStrategy,
        include_validation: bool = False,
        include_test: bool = False,
        splits_included: bool = False,
        seed: int | None = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str | None = "rating",
        time_col: str | None = "timestamp",
    ):
        self._user_col = user_col
        self._item_col = item_col
        self._rating_col = rating_col
        self._time_col = time_col

        self.interaction_data = interaction_data
        self._remap_ids()

        self.num_users = len(self.interaction_data[self._user_col].unique())
        self.num_items = len(self.interaction_data[self._item_col].unique())
        self.num_interactions = len(interaction_data)

        self.splitting_strategy = splitting_strategy
        self.include_validation = include_validation
        self.include_test = include_test

        self.client_partitioning_strategy = client_partitioning_strategy

        self.seed = seed

        if splits_included:
            self.train_idx = self.interaction_data.index[self.interaction_data['split'] == 'train']
            if self.include_validation:
                self.val_idx = self.interaction_data.index[self.interaction_data['split'] == 'val']
            if self.include_test:
                self.test_idx = self.interaction_data.index[self.interaction_data['split'] == 'test']
        else:
            self.train_idx = None
            self.val_idx = None
            self.test_idx = None

    @property
    def train_interactions(self):
        if self.train_idx is None:
            raise ValueError("Train index is not set, call the method split_dataset to split the dataset")
        return self.interaction_data.iloc[self.train_idx]

    @property
    def val_interactions(self):
        if not self.include_validation:
            raise ValueError("Validation dataset is not included in this dataset")
        if self.val_idx is None:
            raise ValueError("Validation index is not set, call the method split_dataset to split the dataset")
        return self.interaction_data.iloc[self.val_idx]

    @property
    def test_interactions(self):
        if not self.include_test:
            raise ValueError("Test dataset is not included in this dataset")
        if self.test_idx is None:
            raise ValueError("Test index is not set, call the method split_dataset to split the dataset")
        return self.interaction_data.iloc[self.test_idx]

    def get_client_view(
        self,
        client_id: int,
        include_train: bool = True,
        include_validation: bool = False,
        include_test: bool = False,
    ) -> pd.DataFrame:
        if include_validation and not self.include_validation:
            include_validation = False
        if include_test and not self.include_test:
            include_test = False

        if not include_train and not include_validation and not include_test:
            raise ValueError("At least one of include_train, include_validation, or include_test must be True")

        client_users = self.client_partition.client_ids_2_user_ids[client_id]

        parts = []
        if include_train:
            parts.append(self.train_interactions.loc[self.client_partition.client_to_train_indices[client_id]])
        if include_validation:
            parts.append(self.val_interactions[self.val_interactions[self._user_col].isin(client_users)])
        if include_test:
            parts.append(self.test_interactions[self.test_interactions[self._user_col].isin(client_users)])

        if len(parts) == 1:
            return parts[0]
        return pd.concat(parts, ignore_index=True)

    def split_data(self):
        self.train_idx, val_idx, test_idx = self.splitting_strategy.split(self.interaction_data)
        self.interaction_data.loc[self.train_idx, 'split'] = 'train'
        if self.include_validation:
            self.val_idx = val_idx
            self.interaction_data.loc[self.val_idx, 'split'] = 'val'
        if self.include_test:
            self.test_idx = test_idx
            self.interaction_data.loc[self.test_idx, 'split'] = 'test'

    @property
    def num_clients(self):
        if not hasattr(self, 'client_partition'):
            raise ValueError("Client partition not set. Call partition_clients() first.")
        return self.client_partition.num_clients

    def _remap_ids(self):
        """Remap user and item IDs to contiguous 0-indexed integers for embedding lookups.

        All downstream code (splitting, partitioning, training, evaluation) sees
        remapped IDs. Original IDs can be recovered via reverse_user_id_map / reverse_item_id_map.
        """
        unique_users = sorted(self.interaction_data[self._user_col].unique())
        unique_items = sorted(self.interaction_data[self._item_col].unique())

        self.user_id_map = {orig: idx for idx, orig in enumerate(unique_users)}
        self.item_id_map = {orig: idx for idx, orig in enumerate(unique_items)}
        self.reverse_user_id_map = {idx: orig for orig, idx in self.user_id_map.items()}
        self.reverse_item_id_map = {idx: orig for orig, idx in self.item_id_map.items()}

        self.interaction_data[self._user_col] = self.interaction_data[self._user_col].map(self.user_id_map)
        self.interaction_data[self._item_col] = self.interaction_data[self._item_col].map(self.item_id_map)

    def partition_clients(self):
        self.client_partition = self.client_partitioning_strategy.partition(self.train_interactions)
        all_train_indices = [
            idx for indices in self.client_partition.client_to_train_indices.values()
            for idx in indices
        ]
        self.interaction_data.loc[all_train_indices, 'client_id'] = (
            self.interaction_data.loc[all_train_indices, self._user_col]
            .map(self.client_partition.user_ids_2_client_ids)
        )
