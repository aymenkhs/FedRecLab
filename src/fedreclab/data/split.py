import numpy as np
import pandas as pd


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

