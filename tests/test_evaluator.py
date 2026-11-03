"""Full-ranking evaluation returns ranking metrics only, not a dummy loss."""

import numpy as np
import pandas as pd

from fedreclab.data.client_partitioning import CrossDevicePartitionStrategy
from fedreclab.data.dataset import InteractionDataset
from fedreclab.data.evaluator import FullRankingEvaluator
from fedreclab.data.split import RatioSplitStrategy
from fedreclab.federated.base_client import BaseClientAlgorithm, ClientStateBase


class _StubScorer(BaseClientAlgorithm):
    def compute_scores(
        self,
        client_id,
        candidate_items_per_user,
        global_parameters,
        client_state,
        extra_payload,
    ):
        return {
            user_id: np.arange(len(item_ids), dtype=np.float32)
            for user_id, item_ids in candidate_items_per_user.items()
        }


def _tiny_ranked_dataset():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1],
            "item_id": [0, 1, 2, 0, 1, 2],
            "rating": [5.0] * 6,
            "split": ["train", "val", "test", "train", "val", "test"],
        }
    )
    dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=True, include_test=True
        ),
        client_partitioning_strategy=CrossDevicePartitionStrategy(),
        k_core=0,
        include_validation=True,
        include_test=True,
        splits_included=True,
        time_col=None,
    )
    dataset.partition_clients()
    return dataset


def test_evaluate_results_do_not_include_loss():
    dataset = _tiny_ranked_dataset()
    evaluator = FullRankingEvaluator(
        metrics=["ndcg"],
        topk=1,
        interaction_dataset=dataset,
        client_algorithm=_StubScorer(),
    )
    client_states = {
        client_id: ClientStateBase(client_id=client_id, local_parameters={})
        for client_id in range(dataset.num_clients)
    }
    results = evaluator.evaluate(
        current_round=0,
        global_parameters={},
        client_states=client_states,
        extra_payload={},
        split="val",
    )
    assert "loss" not in results
    assert "ndcg@1" in results
