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

        self.num_users = len(interaction_data[self._user_col].unique())
        self.num_items = len(interaction_data[self._item_col].unique())
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

        parts = []
        if include_train:
            parts.append(self.train_interactions.iloc[self.client_partition.client_to_train_indices[client_id]])
        if include_validation:
            parts.append(self.val_interactions.iloc[self.client_partition.client_to_val_indices[client_id]])
        if include_test:
            parts.append(self.test_interactions.iloc[self.client_partition.client_to_test_indices[client_id]])

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

    def partition_clients(self):
        self.client_partition = self.client_partitioning_strategy.partition(self.train_interactions)
        self.interaction_data.loc[
            self.client_partition.client_to_train_indices,
            'client_id'
        ] = self.interaction_data[self._user_col].apply(lambda x: self.client_partition.user_ids_2_client_ids[x])

