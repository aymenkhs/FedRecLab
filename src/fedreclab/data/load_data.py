import logging

import pandas as pd
from cornac.datasets import (
    movielens,
    filmtrust,
    amazon_clothing,
    amazon_digital_music,
    amazon_office,
    amazon_toy,
    netflix,
    tradesy,
    epinions,
    citeulike,
)

from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import BaseSplitStrategy
from src.fedreclab.data.client_partitioning import BaseClientPartitionStrategy


CORNAC_DATASETS = {
    "movielens_100k": movielens,
	"movielens_1m": movielens,
	"movielens_10m": movielens,
	"movielens_20m": movielens,
    "filmtrust": filmtrust,
    "amazon_clothing": amazon_clothing,
    "amazon_digital_music": amazon_digital_music,
    "amazon_office": amazon_office,
    "amazon_toy": amazon_toy,
    "netflix_small": netflix,
    "netflix_original": netflix,
    "tradesy": tradesy,
    "epinions": epinions,
    "citeulike": citeulike,
}

CORNAC_DATASETS_VARIANTS = {
    "netflix_small": "small",
    "netflix_original": "original",
}

CORNAC_INCLUDE_TIMESTAMPS = {
    "movielens_100k": True,
	"movielens_1m": True,
	"movielens_10m": True,
	"movielens_20m": True,
    "filmtrust": False,
    "amazon_clothing": False,
    "amazon_digital_music": False,
    "amazon_office": False,
    "amazon_toy": False,
    "netflix_small": True,
    "netflix_original": True,
    "tradesy": False,
    "epinions": False,
    "citeulike": False,
}


def load_cornac_dataset(
    dataset_name: str,
    splitting_strategy: BaseSplitStrategy,
    client_partitioning_strategy: BaseClientPartitionStrategy,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    time_col: str | None = "timestamp",
    include_validation: bool = False,
    include_test: bool = False,
    seed: int = 42,
    rating_exclusion_threshold: float = 0.0,
):
    if dataset_name not in CORNAC_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not available in cornac datasets please provide one of the following: {list(CORNAC_DATASETS.keys())}")

    if CORNAC_INCLUDE_TIMESTAMPS[dataset_name]:
        if time_col is None:
            logging.warning(f"Dataset {dataset_name} includes timestamps but no time_col was provided, ignoring timestamps")
        timestamp = time_col is not None
    else:
        if time_col is not None:
            logging.warning(f"Dataset {dataset_name} does not include timestamps but time_col was provided, ignoring time_col")
            time_col = None
        timestamp = False

    if timestamp:
        columns = [user_col, item_col, rating_col, time_col]
    else:
        columns = [user_col, item_col, rating_col]

    if dataset_name in CORNAC_DATASETS_VARIANTS:
        if timestamp:
            data = CORNAC_DATASETS[dataset_name].load_feedback(variant=CORNAC_DATASETS_VARIANTS[dataset_name], fmt="UIRT")
        else:
            data = CORNAC_DATASETS[dataset_name].load_feedback(variant=CORNAC_DATASETS_VARIANTS[dataset_name])
    else:
        if timestamp:
            data = CORNAC_DATASETS[dataset_name].load_feedback(fmt="UIRT")
        else:
            data = CORNAC_DATASETS[dataset_name].load_feedback()
    df = pd.DataFrame(data, columns=columns)

    if rating_exclusion_threshold > 0.0:
        df = df[df[rating_col] >= rating_exclusion_threshold]

    interaction_dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=splitting_strategy,
        client_partitioning_strategy=client_partitioning_strategy,
        include_validation=include_validation,
        include_test=include_test,
        seed=seed,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        time_col=time_col,
    )
    interaction_dataset.split_data()
    interaction_dataset.partition_clients()
    return interaction_dataset
