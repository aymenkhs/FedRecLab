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
    "movielens_100k": "100K",
    "movielens_1m": "1M",
    "movielens_10m": "10M",
    "movielens_20m": "20M",
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
    k_core: int = 0,
    min_user_interactions: int = 0,
    min_item_interactions: int = 0,
    seed: int = 42,
    rating_exclusion_threshold: float = 0.0,
):
    """Load a Cornac dataset into an ``InteractionDataset``.

    The rating filter runs on the loaded table first. The dataset then
    applies the minimum-interaction filter and the bipartite k-core, so
    both count rows that remain after rating filtering. Defaults of 0
    leave the table unfiltered.

    Parameters
    ----------
    dataset_name : str
        Key in ``CORNAC_DATASETS``.
    splitting_strategy : BaseSplitStrategy
        Split strategy forwarded to ``InteractionDataset``.
    client_partitioning_strategy : BaseClientPartitionStrategy
        Client partition strategy forwarded to ``InteractionDataset``.
    user_col, item_col, rating_col : str, optional
        Column names for the constructed frame.
    time_col : str or None, optional
        Timestamp column name. Ignored when the Cornac source has no
        timestamps.
    include_validation, include_test : bool, optional
        Split flags forwarded to ``InteractionDataset``.
    k_core : int, optional
        Bipartite k-core threshold. ``0`` disables it. Default is 0.
    min_user_interactions, min_item_interactions : int, optional
        One-pass frequency thresholds. ``0`` disables that side.
        Default is 0.
    seed : int, optional
        Seed stored on the dataset. Default is 42.
    rating_exclusion_threshold : float, optional
        Keep rows with rating ``>=`` this value when ``> 0``. Default is 0.

    Returns
    -------
    InteractionDataset
        Split and client-partitioned dataset.

    Raises
    ------
    ValueError
        If ``dataset_name`` is unknown, or if a requested dataset filter
        removes every interaction.
    """
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
        k_core=k_core,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    interaction_dataset.split_data()
    interaction_dataset.partition_clients()
    return interaction_dataset


def load_dataframe_dataset(
    df: pd.DataFrame,
    splitting_strategy: BaseSplitStrategy,
    client_partitioning_strategy: BaseClientPartitionStrategy,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str | None = "rating",
    time_col: str | None = "timestamp",
    include_validation: bool = False,
    include_test: bool = False,
    seed: int = 42,
    rating_exclusion_threshold: float = 0.0,
    k_core: int = 0,
    min_user_interactions: int = 0,
    min_item_interactions: int = 0,
):
    """Build an ``InteractionDataset`` from a pandas DataFrame.

    The rating filter runs on ``df`` first. The dataset then applies the
    minimum-interaction filter and the bipartite k-core, so both count
    rows that remain after rating filtering. Defaults of 0 leave the
    table unfiltered.

    Parameters
    ----------
    df : pandas.DataFrame
        Interaction table.
    splitting_strategy : BaseSplitStrategy
        Split strategy forwarded to ``InteractionDataset``.
    client_partitioning_strategy : BaseClientPartitionStrategy
        Client partition strategy forwarded to ``InteractionDataset``.
    user_col, item_col : str, optional
        Column names that must exist on ``df``.
    rating_col : str or None, optional
        Rating column name. Missing columns are ignored.
    time_col : str or None, optional
        Timestamp column name. Missing columns are ignored.
    include_validation, include_test : bool, optional
        Split flags forwarded to ``InteractionDataset``.
    seed : int, optional
        Seed stored on the dataset. Default is 42.
    rating_exclusion_threshold : float, optional
        Keep rows with rating ``>=`` this value when ``> 0`` and
        ``rating_col`` is present. Default is 0.
    k_core : int, optional
        Bipartite k-core threshold. ``0`` disables it. Default is 0.
    min_user_interactions, min_item_interactions : int, optional
        One-pass frequency thresholds. ``0`` disables that side.
        Default is 0.

    Returns
    -------
    InteractionDataset
        Split and client-partitioned dataset.

    Raises
    ------
    ValueError
        If ``user_col`` or ``item_col`` is missing, or if a requested
        dataset filter removes every interaction.
    """
    if user_col not in df.columns:
        raise ValueError(f"user_col '{user_col}' not found in dataframe columns: {list(df.columns)}")
    if item_col not in df.columns:
        raise ValueError(f"item_col '{item_col}' not found in dataframe columns: {list(df.columns)}")

    # Check rating column
    if rating_col is not None:
        if rating_col not in df.columns:
            logging.warning(
                f"Rating column '{rating_col}' not found in dataframe, "
                f"ignoring ratings."
            )
            rating_col = None
        elif rating_exclusion_threshold > 0.0:
            df = df[df[rating_col] >= rating_exclusion_threshold]

    # Check timestamp column
    if time_col is not None and time_col not in df.columns:
        logging.warning(
            f"Timestamp column '{time_col}' not found in dataframe, "
            f"ignoring timestamps."
        )
        time_col = None

    interaction_dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=splitting_strategy,
        client_partitioning_strategy=client_partitioning_strategy,
        k_core=k_core,
        include_validation=include_validation,
        include_test=include_test,
        seed=seed,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        time_col=time_col,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    interaction_dataset.split_data()
    interaction_dataset.partition_clients()
    return interaction_dataset
