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


class LeaveNOutSplitStrategy(__BaseTrainValTestSplit):
    """
    Leave-N-out split per user. For every user, N interactions go to test and N to validation; the rest are train.
    chronoTrue means the data will be split in chronological order for the different sets.
    """

    def __init__(
        self,
        include_validation: bool = False,
        include_test: bool = False,
        seed: int = 42,
        rating_threshold: float = 0.0,
        n_val: int = 1,
        n_test: int = 1,
        chrono: bool = False, 
        **kwargs,
    ):
        super().__init__(
            include_validation=include_validation,
            include_test=include_test, 
            seed=seed,
            **kwargs,
        )
        if n_val < 0 or n_test < 0:
            raise ValueError("n_val and n_test must be non-negative integers")
        if chrono and self.time_col is None:
            raise ValueError("chrono=True requires a column with timestamp")
        self.rating_threshold=rating_threshold
        self.n_val = n_val
        self.n_test = n_test
        self.chrono = chrono
        self.seed = seed
        
    def split(self, interaction_data: pd.DataFrame):
        df = interaction_data.reset_index(drop=True)
        n = len(df)

        if not self.include_validation and not self.include_test:
            return np.arange(n, dtype=int), None, None

        for col in (self.user_col, self.item_col):
            if col not in df.columns:
                raise ValueError(f"Interaction data must contain '{col}' column")

        n_val = self.n_val if self.include_validation else 0
        n_test = self.n_test if self.include_test else 0
        required = n_val + n_test

        # Which interactions may be held out (rating >= threshold)
        if self.rating_col is not None and self.rating_col in df.columns:
            eligible_mask = (df[self.rating_col].to_numpy() >= self.rating_threshold)
        elif self.rating_threshold > 0:
            raise ValueError(
                f"rating_threshold={self.rating_threshold} requires a "
                f"'{self.rating_col}' column in the interaction data"
            )
        else:
            eligible_mask = np.ones(n, dtype=bool)

        users = df[self.user_col]
        total_counts = users.groupby(users).size()
        eligible_counts = (
            pd.Series(eligible_mask, index=df.index).groupby(users).sum()
        )

        # Every user needs >= required eligible interactions to hold out,
        # and more than `required` interactions overall to keep one in train.
        too_few_eligible = eligible_counts[eligible_counts < required]
        too_few_total = total_counts[total_counts <= required]
        if len(too_few_eligible) > 0 or len(too_few_total) > 0:
            msgs = []
            if len(too_few_eligible) > 0:
                ex = ", ".join(f"{u!r} ({c})" for u, c in too_few_eligible.head(5).items())
                msgs.append(
                    f"{len(too_few_eligible)} user(s) have fewer than {required} "
                    f"interactions with rating >= {self.rating_threshold} "
                    f"(n_val={n_val}, n_test={n_test}). Examples (user, eligible count): {ex}"
                )
            if len(too_few_total) > 0:
                ex = ", ".join(f"{u!r} ({c})" for u, c in too_few_total.head(5).items())
                msgs.append(
                    f"{len(too_few_total)} user(s) have {required} or fewer interactions, "
                    f"so none would remain for training. Examples (user, count): {ex}"
                )
            raise ValueError(
                " | ".join(msgs)
                + ". Reduce n_val/n_test, lower rating_threshold, or filter out these users."
            )

        # Ordering key within each user
        if self.chrono:
            if self.time_col not in df.columns:
                raise ValueError(f"Interaction data must contain '{self.time_col}' column")
            key = df[self.time_col].to_numpy()
        else:
            key = np.random.default_rng(self.seed).random(n)

        # Rank only eligible interactions; non-eligible ones never leave train
        tmp = pd.DataFrame(
            {
                "user": df[self.user_col].to_numpy(),
                "key": key,
                "pos": np.arange(n, dtype=int),  # tie-breaker -> deterministic
            }
        )[eligible_mask].sort_values(["user", "key", "pos"], kind="mergesort")

        # 0 = last eligible interaction of the user (latest, or last in random order)
        rank_from_end = tmp.groupby("user", sort=False).cumcount(ascending=False).to_numpy()
        pos = tmp["pos"].to_numpy()

        test_pos = pos[rank_from_end < n_test]
        val_pos = pos[(rank_from_end >= n_test) & (rank_from_end < required)]

        held_out = np.zeros(n, dtype=bool)
        held_out[test_pos] = True
        held_out[val_pos] = True

        train_idx=np.flatnonzero(eligible_mask & ~held_out)
        val_idx = np.sort(val_pos) if self.include_validation else None
        test_idx = np.sort(test_pos) if self.include_test else None


        return train_idx, val_idx, test_idx