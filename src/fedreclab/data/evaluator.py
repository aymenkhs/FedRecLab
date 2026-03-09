from typing import List, Dict, Any, Callable

import numpy as np
import pandas as pd
from cornac.metrics import (
    Precision,
    Recall,
    NDCG,
    NCRR,
    MAP,
    HitRatio,
    FMeasure,
    AUC,
)

from src.fedreclab.federated.base_client import ClientStateBase, BaseClientAlgorithm
from src.fedreclab.data.dataset import InteractionDataset


class Evaluator:

    TOPK_METRIC_CLASSES = {
        "ndcg": NDCG,
        "precision": Precision,
        "recall": Recall,
        "hr": HitRatio,
        "ncrr": NCRR,
        "fmeasure": FMeasure,
    }

    NON_TOPK_METRIC_CLASSES = {
        # "map": MAP,
        # "auc": AUC,
    }

    def __init__(
        self,
        metrics: List[str],
        topk: int | List[int],
        interaction_dataset: InteractionDataset,
        client_algorithm: BaseClientAlgorithm,
        compute_confidence_intervals: bool = False,
        confidence_interval_alpha: float = 0.05,
        logging_callback: Callable | None = None,
        eval_mode: str = "server_side",
        verbose: bool = True,
    ):
        self.metrics = metrics
        self.topk = topk
        if isinstance(self.topk, int):
            self.topk = [self.topk]

        self.interaction_dataset = interaction_dataset
        self.client_algorithm = client_algorithm

        self.compute_confidence_intervals = compute_confidence_intervals
        self.confidence_interval_alpha = confidence_interval_alpha
        
        self.logging_callback = logging_callback
        self.eval_mode = eval_mode
        self.verbose = verbose

    def evaluate(
        self,
        current_round: int,
        global_parameters: Dict[str, Any],
        client_states: Dict[int, ClientStateBase],
        extra_payload: Dict[str, Any],
        split: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    def compute_metrics_per_user(
        self,
        item_rankings: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, Any]:
        metrics_results = {}
        for metric in self.metrics:
            if metric in self.TOPK_METRIC_CLASSES:
                for k in self.topk:
                    metric_instance = self.TOPK_METRIC_CLASSES[metric](k=k)
                    metrics_results[f"{metric}@{k}"] = metric_instance.compute(item_rankings, targets)
            elif metric in self.NON_TOPK_METRIC_CLASSES:
                metric_class = self.NON_TOPK_METRIC_CLASSES[metric]
                metrics_results[metric] = metric_class.compute(item_rankings, targets)
            else:
                raise ValueError(f"Invalid metric: {metric}")
        return metrics_results

    def compute_confidence_interval(
        self,
        metric_values: np.ndarray,
    ) -> Dict[str, float]:
        n = len(metric_values)
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        se = std / np.sqrt(n) if n > 0 else 0.0
        ei = 1.96 * se
        ci_lower = mean - ei
        ci_upper = mean + ei
        return {"mean": mean, "std": std, "ei": ei, "ci": [ci_lower, ci_upper]}


class FullRankingEvaluator(Evaluator):
    def __init__(
        self,
        metrics: List[str],
        topk: int | List[int],
        interaction_dataset: InteractionDataset,
        client_algorithm: BaseClientAlgorithm,
        compute_confidence_intervals: bool = False,
        confidence_interval_alpha: float = 0.05,
        logging_callback: Callable | None = None,
        eval_mode: str = "server_side",
        verbose: bool = True,
        threshold_rating: float = 0.0,
    ):
        super().__init__(metrics, topk, interaction_dataset, client_algorithm, compute_confidence_intervals, confidence_interval_alpha, logging_callback, eval_mode, verbose)
        
        self.full_item_list = self.interaction_dataset.interaction_data['item_id'].unique()
        self.user_candidate_items = {
            user_id: np.setdiff1d(
                self.full_item_list,
                self.interaction_dataset.get_client_view(
                    client_id=self.interaction_dataset.client_partition.user_ids_2_client_ids[user_id],
                    include_train=True,
                    include_validation=False,
                    include_test=False,
                )['item_id'].unique(),
            ) for user_id in self.interaction_dataset.interaction_data['user_id'].unique()
        }
        self.threshold_rating = threshold_rating

    def evaluate(
        self,
        current_round: int,
        global_parameters: Dict[str, Any],
        client_states: Dict[int, ClientStateBase],
        extra_payload: Dict[str, Any],
        split: str,
    ) -> Dict[str, Any]:
        if split == "test":
            evaluation_data = self.interaction_dataset.test_interactions
        elif split == "val":
            evaluation_data = self.interaction_dataset.val_interactions
        else:
            raise ValueError(f"Invalid split: {split}")

        metrics_results_per_user = {}
        losses = []

        for client_id, client_state in client_states.items():

            client_eval_interactions = evaluation_data[evaluation_data['client_id'] == client_id]
            users_in_client = self.interaction_dataset.client_partition.client_ids_2_user_ids[client_id]
            candidate_items_per_user = {
                user_id: self.user_candidate_items[user_id] for user_id in users_in_client
            }

            scores, loss = self.client_algorithm.compute_scores(
                client_id=client_id,
                candidate_items_per_user=candidate_items_per_user,
                global_parameters=global_parameters,
                client_state=client_state,
                extra_payload=extra_payload,
            )
            losses.append(loss)

            for user_id in users_in_client:
                eval_interactions_user = client_eval_interactions[
                    client_eval_interactions['user_id'] == user_id
                ]
                if "rating" in eval_interactions_user.columns and self.threshold_rating > 0.0:
                    eval_interactions_user = eval_interactions_user[eval_interactions_user['rating'] >= self.threshold_rating]                
                if len(eval_interactions_user) == 0:
                    continue
                eval_items = eval_interactions_user['item_id'].unique()

                candidate_items = candidate_items_per_user[user_id]
                user_scores = scores[user_id]
                rank_positions = np.argsort(-user_scores)
                ranked_item_ids = candidate_items[rank_positions]

                metrics_results_per_user[user_id] = self.compute_metrics_per_user(ranked_item_ids, eval_items)

        metrics_list = metrics_results_per_user[list(metrics_results_per_user.keys())[0]].keys()
        metric_results = {}
        for metric in metrics_list:
            metric_per_user_values = [metrics_results_per_user[user_id][metric] for user_id in metrics_results_per_user.keys()]
            if self.compute_confidence_intervals:
                metric_results[metric] = self.compute_confidence_interval(metric_per_user_values)
            else:
                metric_results[metric] = np.mean(metric_per_user_values)

        metric_results["loss"] = np.mean(losses)

        if self.logging_callback:
            self.logging_callback(current_round, metric_results, split, extra_payload)
        return metric_results
