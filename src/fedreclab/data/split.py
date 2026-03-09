import numpy as np
import pandas as pd
from cornac.eval_methods import RatioSplit
from cornac.eval_methods import StratifiedSplit


class BaseSplitStrategy:
    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str | None = "rating",
        time_col: str | None = "timestamp",
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.time_col = time_col

    def split(self, interaction_data: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement this method")


class __BaseTrainValTestSplit(BaseSplitStrategy):
    """
    Base class for strategies that split interactions into train / val / test.

    If both validation and test are disabled, all interactions are returned as train.
    Subclasses implement `_split_indices` for the actual splitting logic.
    """

    def __init__(
        self,
        include_validation: bool = False,
        include_test: bool = False,
        val_size: float | int = 0.0,
        test_size: float | int = 0.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_validation = include_validation
        self.include_test = include_test
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def split(self, interaction_data: pd.DataFrame):
        if not self.include_validation and not self.include_test:
            # All interactions go to train
            n = len(interaction_data)
            train_idx = np.arange(n, dtype=int)
            return train_idx, None, None

        # Reset index so positions 0..n-1 align with DataFrame row order
        df = interaction_data.reset_index(drop=True)
        n = len(df)

        if self.user_col not in df.columns or self.item_col not in df.columns:
            raise ValueError(
                f"Interaction data must contain '{self.user_col}' and '{self.item_col}' columns"
            )

        users = df[self.user_col].to_numpy()
        items = df[self.item_col].to_numpy()

        # Use row positions as a stable identifier; store them in the rating field
        row_ids = np.arange(n, dtype=np.int64)

        # Timestamp is optional; when missing, fill with zeros
        if self.time_col is not None and self.time_col in df.columns:
            times = df[self.time_col].to_numpy()
        else:
            times = np.zeros(n, dtype=np.int64)

        if self._requires_timestamp():
            # UIRT: (user, item, rating=row_id, timestamp)
            data_np = np.column_stack([users, items, row_ids, times])
        else:
            # UIR: (user, item, rating=row_id)
            data_np = np.column_stack([users, items, row_ids])

        splitter = self._build_cornac_splitter(data_np)

        def dataset_to_indices(dataset):
            if dataset is None:
                return None
            _, _, ratings = dataset.uir_tuple
            return np.asarray(ratings, dtype=int)

        train_idx = dataset_to_indices(splitter.train_set)
        val_idx = dataset_to_indices(splitter.val_set)
        test_idx = dataset_to_indices(splitter.test_set)

        return train_idx, val_idx, test_idx

    def _build_cornac_splitter(self, data_np: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")

    def _requires_timestamp(self) -> bool:
        """Whether the Cornac splitter expects UIRT data (with timestamp)."""
        return False


class RatioSplitStrategy(__BaseTrainValTestSplit):
    """
    Random ratio-based split of all interactions into train / val / test.
    """

    def __init__(
        self,
        include_validation: bool = False,
        include_test: bool = False,
        val_size: float | int = 0.0,
        test_size: float | int = 0.0,
        seed: int = 42,
        rating_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            include_validation=include_validation,
            include_test=include_test,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            **kwargs,
        )
        self.rating_threshold = rating_threshold

    def _build_cornac_splitter(self, data_np: np.ndarray):
        test_size = self.test_size if self.include_test else 0.0
        val_size = self.val_size if self.include_validation else 0.0

        splitter = RatioSplit(
            data=data_np,
            test_size=test_size,
            val_size=val_size,
            rating_threshold=self.rating_threshold,
            seed=self.seed,
            exclude_unknowns=True,
            verbose=False,
        )
        return splitter


class StratifiedSplitStrategy(__BaseTrainValTestSplit):
    """
    Stratified split per user or per item, with optional chronological ordering.

    Parameters
    ----------
    by : {'user', 'item'}
        Grouping dimension for stratification.
    chrono : bool
        If True, sort interactions within each group by timestamp before splitting.
        If False, use random ordering within each group.
    """

    def __init__(
        self,
        include_validation: bool = False,
        include_test: bool = False,
        val_size: float | int = 0.0,
        test_size: float | int = 0.0,
        seed: int = 42,
        by: str = "user",
        chrono: bool = False,
        rating_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            include_validation=include_validation,
            include_test=include_test,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            **kwargs,
        )

        self.by = by
        self.chrono = chrono
        self.rating_threshold = rating_threshold

    def _requires_timestamp(self) -> bool:
        return self.chrono

    def _build_cornac_splitter(self, data_np: np.ndarray):
        test_size = self.test_size if self.include_test else 0.0
        val_size = self.val_size if self.include_validation else 0.0

        splitter = StratifiedSplit(
            data=data_np,
            group_by=self.by,
            chrono=self.chrono,
            fmt="UIRT" if self.chrono else "UIR",
            test_size=test_size,
            val_size=val_size,
            rating_threshold=self.rating_threshold,
            seed=self.seed,
            exclude_unknowns=True,
            verbose=False,
        )
        return splitter
