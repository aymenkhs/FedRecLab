import numpy as np
import pandas as pd

from fedreclab.data.split import BaseSplitStrategy
from fedreclab.data.client_partitioning import BaseClientPartitionStrategy


class InteractionDataset:
    """Interaction table with optional frequency filters, ID remapping, splits and client partitions.

    When requested, frequency filters run on the input table before IDs are
    remapped: the minimum-interaction filter first, then the bipartite
    k-core. Both count rows, so duplicate ``(user, item)`` pairs contribute
    more than once. Deduplication is the caller's responsibility.

    Parameters
    ----------
    interaction_data : pandas.DataFrame
        Raw interactions. User and item IDs are remapped to contiguous
        0-based integers after filtering.
    splitting_strategy : BaseSplitStrategy
        Strategy used by ``split_data``.
    client_partitioning_strategy : BaseClientPartitionStrategy
        Strategy used by ``partition_clients``.
    k_core : int, optional
        Bipartite k-core threshold. ``0`` disables the k-core. Default is 0.
    min_user_interactions : int, optional
        One-pass minimum rows per user. ``0`` disables the user cut.
        Default is 0.
    min_item_interactions : int, optional
        One-pass minimum rows per item. ``0`` disables the item cut.
        Default is 0.
    include_validation : bool, optional
        Whether a validation split is part of this dataset. Default is False.
    include_test : bool, optional
        Whether a test split is part of this dataset. Default is False.
    splits_included : bool, optional
        If True, read an existing ``split`` column instead of calling
        ``split_data``. Default is False.
    seed : int or None, optional
        Seed stored on the dataset. Default is None.
    user_col : str, optional
        User-id column name. Default is ``"user_id"``.
    item_col : str, optional
        Item-id column name. Default is ``"item_id"``.
    rating_col : str or None, optional
        Rating column name. Default is ``"rating"``.
    time_col : str or None, optional
        Timestamp column name. Default is ``"timestamp"``.

    Raises
    ------
    ValueError
        If ``k_core``, ``min_user_interactions``, or
        ``min_item_interactions`` is not an integer ``>= 0``, or if a
        requested filter removes every interaction.
    """

    def __init__(
        self,
        interaction_data: pd.DataFrame,
        splitting_strategy: BaseSplitStrategy,
        client_partitioning_strategy: BaseClientPartitionStrategy,
        k_core: int = 0,
        min_user_interactions: int = 0,
        min_item_interactions: int = 0,
        include_validation: bool = False,
        include_test: bool = False,
        splits_included: bool = False,
        seed: int | None = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str | None = "rating",
        time_col: str | None = "timestamp",
    ):
        for name, value in (
            ("k_core", k_core),
            ("min_user_interactions", min_user_interactions),
            ("min_item_interactions", min_item_interactions),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}."
                )

        self._user_col = user_col
        self._item_col = item_col
        self._rating_col = rating_col
        self._time_col = time_col

        self.k_core = k_core
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

        self.interaction_data = interaction_data.reset_index(drop=True)
        if self.min_user_interactions > 0 or self.min_item_interactions > 0:
            self._apply_min_interaction_filter(
                self.min_user_interactions, self.min_item_interactions
            )
        if self.k_core > 0:
            self._apply_k_core_filter(self.k_core)
        self._remap_ids()

        self.num_users = len(self.interaction_data[self._user_col].unique())
        self.num_items = len(self.interaction_data[self._item_col].unique())
        self.num_interactions = len(self.interaction_data)

        self.splitting_strategy = splitting_strategy
        self.include_validation = include_validation
        self.include_test = include_test

        self.client_partitioning_strategy = client_partitioning_strategy

        self.seed = seed

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        if splits_included:
            split = self.interaction_data["split"].to_numpy()
            self.train_idx = np.flatnonzero(split == "train")
            if self.include_validation:
                self.val_idx = np.flatnonzero(split == "val")
            if self.include_test:
                self.test_idx = np.flatnonzero(split == "test")

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
        split_labels = np.empty(len(self.interaction_data), dtype=object)
        split_labels[self.train_idx] = "train"
        if self.include_validation:
            self.val_idx = val_idx
            if self.val_idx is not None and len(self.val_idx) > 0:
                split_labels[self.val_idx] = "val"
        if self.include_test:
            self.test_idx = test_idx
            if self.test_idx is not None and len(self.test_idx) > 0:
                split_labels[self.test_idx] = "test"
        self.interaction_data["split"] = split_labels

    @property
    def num_clients(self):
        if not hasattr(self, 'client_partition'):
            raise ValueError("Client partition not set. Call partition_clients() first.")
        return self.client_partition.num_clients

    def _apply_k_core_filter(self, k: int) -> None:
        """Apply a bipartite k-core, iterated to a fixed point.

        Repeatedly drop users and items with fewer than ``k`` rows until
        every remaining user and every remaining item has at least ``k``
        rows, or until nothing is left. The result is the largest
        sub-table of ``self.interaction_data`` with that property.

        Rows are counted, so duplicate ``(user, item)`` pairs contribute
        more than once. Deduplication is the caller's job.

        Parameters
        ----------
        k : int
            Minimum rows required of every remaining user and item.
            Callers with ``k <= 0`` should not invoke this method.

        Returns
        -------
        None
            Updates ``self.interaction_data`` in place and resets its index.

        Raises
        ------
        ValueError
            If the k-core is empty, i.e. every interaction is removed at
            threshold ``k``.

        Examples
        --------
        Users 0 and 1 each have items ``{10, 11}``; user 2 has
        ``{11, 12}``. Item 12 has one row, so it is dropped. User 2 then
        has one row and is dropped too::

            >>> # input:  (0,10), (0,11), (1,10), (1,11), (2,11), (2,12)
            >>> # 2-core: (0,10), (0,11), (1,10), (1,11)
        """
        df = self.interaction_data
        while True:
            user_counts = df.groupby(self._user_col)[self._item_col].count()
            valid_users = user_counts[user_counts >= k].index
            item_counts = df.groupby(self._item_col)[self._user_col].count()
            valid_items = item_counts[item_counts >= k].index
            prev_len = len(df)
            df = df[df[self._user_col].isin(valid_users) & df[self._item_col].isin(valid_items)]
            if len(df) == prev_len or len(df) == 0:
                break
        if len(df) == 0:
            raise ValueError(
                f"k-core filter with k_core={k} removed every interaction."
            )
        self.interaction_data = df.reset_index(drop=True)

    def _apply_min_interaction_filter(self, min_user: int, min_item: int) -> None:
        """Keep rows whose user and item meet one-pass frequency thresholds.

        ``min_user`` is the minimum number of rows a user must have in
        the input table to keep that user's rows; ``0`` disables the
        user cut. ``min_item`` is the same for items; ``0`` disables the
        item cut.

        Counts are taken once on the input table. There is no iteration,
        so a user who passed ``min_user`` may still end with fewer than
        ``min_user`` rows after the item cut. That is the defined
        behaviour of this filter and is what distinguishes it from
        ``k_core``.

        Rows are counted, so duplicate ``(user, item)`` pairs contribute
        more than once. Deduplication is the caller's job.

        Parameters
        ----------
        min_user : int
            Minimum input-table rows per user. ``0`` leaves users unfiltered.
        min_item : int
            Minimum input-table rows per item. ``0`` leaves items unfiltered.

        Returns
        -------
        None
            Updates ``self.interaction_data`` in place and resets its index.

        Raises
        ------
        ValueError
            If the filter removes every interaction at the given
            ``min_user`` and ``min_item`` thresholds.

        Examples
        --------
        Same table as the k-core cascade: users 0 and 1 have items
        ``{10, 11}``; user 2 has ``{11, 12}``. With both thresholds 2,
        item 12 is dropped, but user 2 is kept with one row because that
        user had two rows in the input::

            >>> # input:     (0,10), (0,11), (1,10), (1,11), (2,11), (2,12)
            >>> # after min: (0,10), (0,11), (1,10), (1,11), (2,11)
        """
        df = self.interaction_data
        keep = pd.Series(True, index=df.index)
        if min_user > 0:
            keep &= df.groupby(self._user_col)[self._item_col].transform("size") >= min_user
        if min_item > 0:
            keep &= df.groupby(self._item_col)[self._user_col].transform("size") >= min_item
        df = df[keep]
        if len(df) == 0:
            raise ValueError(
                "minimum-interaction filter with "
                f"min_user_interactions={min_user} and "
                f"min_item_interactions={min_item} removed every interaction."
            )
        self.interaction_data = df.reset_index(drop=True)

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
        self.interaction_data['client_id'] = (
            self.interaction_data[self._user_col]
            .map(self.client_partition.user_ids_2_client_ids)
        )
