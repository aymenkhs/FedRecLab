"""Cornac loaders must request the named dataset variant, not the module default."""

from unittest.mock import patch

import pytest

from src.fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from src.fedreclab.data.load_data import CORNAC_DATASETS_VARIANTS, load_cornac_dataset
from src.fedreclab.data.split import RatioSplitStrategy

# Stub row counts, distinct so a missing variant (Cornac default "100K") fails.
VARIANT_SIZES = {
    "100K": 11,
    "1M": 23,
    "10M": 37,
    "20M": 41,
}

MOVIELENS_VARIANTS = {
    "movielens_100k": "100K",
    "movielens_1m": "1M",
    "movielens_10m": "10M",
    "movielens_20m": "20M",
}


def _stub_feedback(fmt="UIR", variant="100K", reader=None):
    n = VARIANT_SIZES[variant]
    if fmt == "UIRT":
        return [(i, i, 5.0, i) for i in range(n)]
    return [(i, i, 5.0) for i in range(n)]


def test_variants_map_covers_all_movielens_names():
    assert {name: CORNAC_DATASETS_VARIANTS[name] for name in MOVIELENS_VARIANTS} == MOVIELENS_VARIANTS


@pytest.mark.parametrize(
    "dataset_name, expected_variant",
    list(MOVIELENS_VARIANTS.items()),
    ids=list(MOVIELENS_VARIANTS),
)
def test_load_cornac_dataset_uses_variant_sizes(dataset_name, expected_variant):
    with patch(
        "src.fedreclab.data.load_data.movielens.load_feedback",
        side_effect=_stub_feedback,
    ) as mock_load:
        dataset = load_cornac_dataset(
            dataset_name=dataset_name,
            splitting_strategy=RatioSplitStrategy(
                include_validation=False, include_test=False
            ),
            client_partitioning_strategy=CrossDevicePartitionStrategy(),
            k_core=0,
        )

    assert dataset.num_interactions == VARIANT_SIZES[expected_variant]
    assert mock_load.call_args.kwargs["variant"] == expected_variant
    assert mock_load.call_args.kwargs["fmt"] == "UIRT"
