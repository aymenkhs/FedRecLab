from typing import List, Dict, Any, Callable, Sequence
from statistics import NormalDist

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
                    metrics_results[f"{metric}@{k}"] = metric_instance.compute(targets, item_rankings)
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
        """Mean, std, and normal two-sided CI over per-user metric values.

        Half-width is ``z * se`` with ``se = std / sqrt(n)`` and
        ``z = Phi^{-1}(1 - alpha/2)``, where ``alpha`` is
        ``self.confidence_interval_alpha``. For ``alpha=0.05`` this is
        the usual 95% interval (``z ≈ 1.96``).

        Parameters
        ----------
        metric_values : array-like
            One scalar per evaluated user.

        Returns
        -------
        dict
            ``mean``, ``std``, ``ei`` (half-width), and ``ci``
            (``[lower, upper]``).

        Raises
        ------
        ValueError
            If ``confidence_interval_alpha`` is not in ``(0, 1)``.

        Examples
        --------
        >>> ev = Evaluator.__new__(Evaluator)
        >>> ev.confidence_interval_alpha = 0.05
        >>> out = ev.compute_confidence_interval(np.array([0.0, 2.0]))
        >>> round(out["mean"], 6), round(out["ei"] / (1 / np.sqrt(2)), 2)
        (1.0, 1.96)
        """
        alpha = self.confidence_interval_alpha
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"confidence_interval_alpha must be in (0, 1), got {alpha}."
            )
        n = len(metric_values)
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        se = std / np.sqrt(n) if n > 0 else 0.0
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        ei = z * se
        ci_lower = mean - ei
        ci_upper = mean + ei
        return {"mean": mean, "std": std, "ei": ei, "ci": [ci_lower, ci_upper]}


class FullRankingEvaluator(Evaluator):
    """Full-ranking evaluator: every catalogue item except a configured exclusion set.

    Protocol
    --------
    Candidates are ``all items in the dataset minus the user's items from the
    excluded splits``. The exclusion set is **per eval split**:

    * validation (``split="val"``): exclude train only, by default.
    * test (``split="test"``): exclude train and validation, by default.

    Validation items are therefore not treated as test negatives. Pass
    ``exclude_on_test=("train",)`` to restore the older “exclude train only”
    convention. The eval split itself must not appear in its exclusion set,
    or the positives would be dropped from the ranking.

    The catalogue is every item in ``interaction_dataset.interaction_data``,
    including items that never appear in train.
    """

    _VALID_EXCLUDE_SPLITS = frozenset({"train", "val", "test"})

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
        exclude_on_val: Sequence[str] = ("train",),
        exclude_on_test: Sequence[str] = ("train", "val"),
    ):
        """
        Parameters
        ----------
        metrics : list of str
            Ranking metric names (see ``Evaluator.TOPK_METRIC_CLASSES``).
        topk : int or list of int
            Cut-offs for top-k metrics.
        interaction_dataset : InteractionDataset
            Split and partitioned interactions.
        client_algorithm : BaseClientAlgorithm
            Used for ``compute_scores``.
        compute_confidence_intervals : bool, optional
            If True, aggregate per-user metrics as mean ± CI.
        confidence_interval_alpha : float, optional
            Two-sided significance level for the CI.
        logging_callback : callable or None, optional
            ``callback(round, metrics, split, extra_payload)``.
        eval_mode : str, optional
            Forwarded to ``Evaluator``.
        verbose : bool, optional
            Forwarded to ``Evaluator``.
        threshold_rating : float, optional
            If greater than 0, keep only eval interactions with
            ``rating >= threshold_rating``.
        exclude_on_val : sequence of {"train", "val", "test"}, optional
            Splits whose items are removed from candidates when
            ``evaluate(..., split="val")``. Default ``("train",)``.
        exclude_on_test : sequence of {"train", "val", "test"}, optional
            Splits whose items are removed from candidates when
            ``evaluate(..., split="test")``. Default ``("train", "val")``.

        Raises
        ------
        ValueError
            If an exclusion name is not a known split, or if an eval split
            excludes itself (positives would leave the candidate list).
        """
        super().__init__(metrics, topk, interaction_dataset, client_algorithm, compute_confidence_intervals, confidence_interval_alpha, logging_callback, eval_mode, verbose)

        self.threshold_rating = threshold_rating
        self.exclude_on_val = self._normalize_exclude(exclude_on_val, eval_split="val")
        self.exclude_on_test = self._normalize_exclude(exclude_on_test, eval_split="test")
        self.full_item_list = self.interaction_dataset.interaction_data['item_id'].unique()
        self.user_candidate_items = {
            "val": self._precompute_candidates(self.exclude_on_val),
            "test": self._precompute_candidates(self.exclude_on_test),
        }

    def _normalize_exclude(self, exclude: Sequence[str], eval_split: str) -> tuple[str, ...]:
        """Validate and freeze an exclusion list for one eval split.

        Parameters
        ----------
        exclude : sequence of str
            Split names whose items should leave the candidate list.
        eval_split : {"val", "test"}
            Split that will be ranked against those candidates.

        Returns
        -------
        tuple of str
            Deduplicated exclusion names, order preserved.

        Raises
        ------
        ValueError
            If a name is not ``train``/``val``/``test``, or equals ``eval_split``.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> evaluator._normalize_exclude(("train", "val"), eval_split="test")
        ('train', 'val')
        """
        seen = set()
        normalized = []
        for name in exclude:
            if name not in self._VALID_EXCLUDE_SPLITS:
                raise ValueError(
                    f"Invalid exclusion split {name!r}. "
                    f"Expected one of {sorted(self._VALID_EXCLUDE_SPLITS)}."
                )
            if name == eval_split:
                raise ValueError(
                    f"Cannot exclude {eval_split!r} items when evaluating "
                    f"{eval_split}: they are the ranking positives."
                )
            if name not in seen:
                seen.add(name)
                normalized.append(name)
        return tuple(normalized)

    def _items_to_exclude(self, user_id: int, exclude_splits: tuple[str, ...]) -> np.ndarray:
        """Item IDs of ``user_id`` that belong to any split in ``exclude_splits``.

        Parameters
        ----------
        user_id : int
            Remapped user id.
        exclude_splits : tuple of str
            Subset of ``{"train", "val", "test"}``. Splits not present on
            the dataset are ignored.

        Returns
        -------
        ndarray
            Unique item ids to drop from that user's candidate list.
            Empty if nothing is excluded.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> evaluator._items_to_exclude(0, ("train", "val"))
        array([1, 4, 7])
        """
        include_train = "train" in exclude_splits
        include_validation = (
            "val" in exclude_splits and self.interaction_dataset.include_validation
        )
        include_test = (
            "test" in exclude_splits and self.interaction_dataset.include_test
        )
        if not (include_train or include_validation or include_test):
            return np.array([], dtype=self.full_item_list.dtype)

        client_id = self.interaction_dataset.client_partition.user_ids_2_client_ids[user_id]
        view = self.interaction_dataset.get_client_view(
            client_id=client_id,
            include_train=include_train,
            include_validation=include_validation,
            include_test=include_test,
        )
        return view.loc[view["user_id"] == user_id, "item_id"].unique()

    def _precompute_candidates(self, exclude_splits: tuple[str, ...]) -> Dict[int, np.ndarray]:
        """Candidate items per user for one exclusion set.

        Parameters
        ----------
        exclude_splits : tuple of str
            Passed to ``_items_to_exclude``.

        Returns
        -------
        dict of int to ndarray
            ``user_id -> catalogue minus excluded items``.

        Examples
        --------
        >>> # doctest: +SKIP
        >>> evaluator._precompute_candidates(("train",))
        {0: array([2, 3]), 1: array([0, 4])}
        """
        return {
            user_id: np.setdiff1d(
                self.full_item_list,
                self._items_to_exclude(user_id, exclude_splits),
            )
            for user_id in self.interaction_dataset.interaction_data["user_id"].unique()
        }

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
                user_id: self.user_candidate_items[split][user_id] for user_id in users_in_client
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

        if not metrics_results_per_user:
            raise ValueError(
                f"No users had evaluation interactions for split {split!r}."
            )

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
