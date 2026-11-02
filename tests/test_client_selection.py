"""Client selectors: uniqueness, seeds, weighted sampling, round-robin, rpow-d."""

import logging
import random

import numpy as np
import pandas as pd
import pytest

from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.aggregation import FedAvg, SimpleAvg
from src.fedreclab.federated.base_client import BaseClientAlgorithm
from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.client_selection import (
    FullClientSelection,
    PowerOfChoiceClientSelection,
    RandomClientSelection,
    RoundRobinClientSelection,
)
from src.fedreclab.federated._compatibility_warnings import iter_selector_aggregator_warnings
from src.fedreclab.federated.utils import ParamIO, RoundStatistics


def _round_stats(selected, losses=None):
    selected = list(selected)
    return RoundStatistics(
        current_round=0,
        selected=selected,
        responded=selected,
        dropped=[],
        num_training_examples={cid: 1 for cid in selected},
        training_losses={cid: (losses or {}).get(cid, []) for cid in selected},
        training_durations={cid: 0.0 for cid in selected},
    )


def _tiny_dataset(n_users=2):
    df = pd.DataFrame(
        {
            "user_id": list(range(n_users)),
            "item_id": list(range(n_users)),
            "rating": [1.0] * n_users,
            "split": ["train"] * n_users,
        }
    )
    dataset = InteractionDataset(
        interaction_data=df,
        splitting_strategy=RatioSplitStrategy(
            include_validation=False, include_test=False
        ),
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
        k_core=0,
        include_validation=False,
        include_test=False,
        splits_included=True,
        time_col=None,
    )
    dataset.partition_clients()
    return dataset


class _DummyClient(BaseClientAlgorithm):
    def initialize_local_parameters(
        self,
        client_id,
        total_num_users,
        num_users,
        num_items,
        client_train_data_view,
        users_in_client,
    ):
        return {}


class _DummyServer(BaseFederatedServer):
    def _initialize_server_parameters(self):
        return {"w": np.zeros(1, dtype=np.float32)}


def test_integer_count_is_not_treated_as_a_fraction():
    chosen = RandomClientSelection(20, 5).select()
    assert len(chosen) == 5
    assert len(set(chosen)) == 5


def test_float_is_treated_as_a_fraction():
    chosen = RandomClientSelection(20, 0.25).select()
    assert len(chosen) == 5


def test_frac_greater_than_one_raises():
    with pytest.raises(ValueError, match="fraction in \\(0, 1]"):
        RandomClientSelection(20, 20, "frac")


def test_count_larger_than_population_raises():
    with pytest.raises(ValueError, match="without replacement"):
        RandomClientSelection(20, 25, "num")


def test_ids_are_unique_and_valid_for_all_selectors():
    selectors = [
        RandomClientSelection(10, 4, seed=0),
        RandomClientSelection(10, 4, seed=0, weights={i: i + 1 for i in range(10)}),
        RoundRobinClientSelection(10, 4, seed=0),
        PowerOfChoiceClientSelection(10, 4, d=6, seed=0),
        FullClientSelection(10),
    ]
    for sel in selectors:
        chosen = sel.select(0)
        assert len(chosen) == len(set(chosen))
        assert all(0 <= cid < 10 for cid in chosen)


def test_same_seed_reproduces_the_sequence():
    for cls, kwargs in (
        (RandomClientSelection, {}),
        (RandomClientSelection, {"weights": {i: i + 1 for i in range(8)}}),
        (RoundRobinClientSelection, {}),
        (PowerOfChoiceClientSelection, {"d": 5}),
    ):
        a = cls(8, 3, seed=0, **kwargs)
        b = cls(8, 3, seed=0, **kwargs)
        seq_a = [a.select(r) for r in range(5)]
        seq_b = [b.select(r) for r in range(5)]
        assert seq_a == seq_b


def test_replace_true_is_not_implemented():
    with pytest.raises(NotImplementedError, match="replace=True"):
        RandomClientSelection(4, 2, replace=True)


def test_weighted_m1_frequencies_match_p():
    weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    p = np.array([1.0, 2.0, 3.0, 4.0])
    p = p / p.sum()
    sel = RandomClientSelection(4, 1, seed=0, weights=weights)
    counts = np.zeros(4, dtype=np.float64)
    n_trials = 6000
    for r in range(n_trials):
        counts[sel.select(r)[0]] += 1
    np.testing.assert_allclose(counts / n_trials, p, atol=0.02)


def test_zero_weight_clients_are_never_selected():
    sel = RandomClientSelection(
        4, 2, seed=0, weights={0: 1.0, 1: 0.0, 2: 3.0, 3: 1.0}
    )
    for r in range(200):
        assert 1 not in sel.select(r)


def test_weighted_without_replacement_too_few_positive_weights_raises():
    with pytest.raises(ValueError, match="positive weight"):
        RandomClientSelection(
            5, 3, weights={0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0}
        )


def test_all_zero_weights_raise():
    with pytest.raises(ValueError, match="positive value"):
        RandomClientSelection(3, 1, weights={0: 0.0, 1: 0.0, 2: 0.0})


def test_non_finite_weights_raise():
    with pytest.raises(ValueError, match="finite"):
        RandomClientSelection(3, 1, weights={0: 1.0, 1: np.nan, 2: 1.0})
    with pytest.raises(ValueError, match="finite"):
        RandomClientSelection(3, 1, weights=[1.0, np.inf, 1.0])


def test_round_robin_short_last_visits_each_client_once_per_cycle():
    rr = RoundRobinClientSelection(5, 2, seed=0, remainder="short_last")
    first = rr.select(0)
    second = rr.select(1)
    third = rr.select(2)
    assert len(first) == 2
    assert len(second) == 2
    assert len(third) == 1
    cycle = first + second + third
    assert sorted(cycle) == [0, 1, 2, 3, 4]
    assert rr.select(3) == first


def test_round_robin_reshuffle_changes_the_next_cycle():
    rr = RoundRobinClientSelection(8, 2, seed=0, reshuffle_each_cycle=True)
    first = [tuple(rr.select(r)) for r in range(4)]
    second = [tuple(rr.select(r)) for r in range(4, 8)]
    assert sorted(c for cohort in first for c in cohort) == list(range(8))
    assert sorted(c for cohort in second for c in cohort) == list(range(8))
    assert first != second


def test_round_robin_wrap_keeps_m_and_repeats():
    rr = RoundRobinClientSelection(5, 3, seed=0, remainder="wrap")
    a = rr.select(0)
    b = rr.select(1)
    assert len(a) == 3
    assert len(b) == 3
    assert len(set(a + b)) <= 5


def test_round_robin_wrap_reshuffle_ids_are_unique_each_round():
    rr = RoundRobinClientSelection(
        5, 3, seed=0, remainder="wrap", reshuffle_each_cycle=True
    )
    for r in range(200):
        chosen = rr.select(r)
        assert len(chosen) == 3
        assert len(set(chosen)) == 3
        assert all(0 <= cid < 5 for cid in chosen)


def test_round_robin_cycp_groups_visit_whole_groups_in_order():
    rr = RoundRobinClientSelection(
        6, 2, seed=0, groups=[[0, 1, 2], [3, 4, 5]]
    )
    assert rr.select(0) == [0, 1, 2]
    assert rr.select(1) == [3, 4, 5]
    assert rr.select(2) == [0, 1, 2]


def test_round_robin_cycp_subsamples_each_group():
    rr = RoundRobinClientSelection(
        6, 2, seed=0, groups=[[0, 1, 2], [3, 4, 5]], clients_per_group=2
    )
    first = rr.select(0)
    second = rr.select(1)
    assert len(first) == 2 and set(first).issubset({0, 1, 2})
    assert len(second) == 2 and set(second).issubset({3, 4, 5})


def test_power_of_choice_prefers_unseen_infinite_loss():
    poc = PowerOfChoiceClientSelection(4, 2, d=4, seed=0)
    poc.last_loss[0] = 1.0
    chosen = poc.select(0)
    assert 0 not in chosen
    assert set(chosen).issubset({1, 2, 3})


def test_power_of_choice_returns_top_m_when_all_losses_are_finite():
    poc = PowerOfChoiceClientSelection(4, 2, d=4, seed=0)
    poc.last_loss[:] = [1.0, 5.0, 3.0, 4.0]
    assert set(poc.select(0)) == {1, 3}


def test_power_of_choice_breaks_ties_randomly():
    counts = np.zeros(2, dtype=int)
    for seed in range(40):
        poc = PowerOfChoiceClientSelection(2, 1, d=2, seed=seed)
        poc.last_loss[:] = [1.0, 1.0]
        chosen = poc.select(0)
        counts[chosen[0]] += 1
    assert counts.min() > 0
    assert counts.max() < 40


def test_power_of_choice_d_schedule_is_clamped(caplog):
    def too_small(_round):
        return 1

    def too_large(_round):
        return 99

    caplog.set_level(logging.INFO, logger="src.fedreclab.federated.client_selection")
    small = PowerOfChoiceClientSelection(5, 2, d=3, seed=0, d_schedule=too_small)
    small.select(0)
    assert "of 2 candidates" in caplog.text
    caplog.clear()
    large = PowerOfChoiceClientSelection(5, 2, d=3, seed=0, d_schedule=too_large)
    large.select(0)
    assert "of 5 candidates" in caplog.text


def test_power_of_choice_d_equals_m_matches_weighted_without_replacement():
    weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    n_trials = 4000
    m = 2
    poc = PowerOfChoiceClientSelection(4, m, d=m, seed=0, weights=weights)
    rnd = RandomClientSelection(4, m, seed=1, weights=weights, replace=False)
    freq_p = np.zeros(4)
    freq_r = np.zeros(4)
    for t in range(n_trials):
        for c in poc.select(t):
            freq_p[c] += 1
        for c in rnd.select(t):
            freq_r[c] += 1
    np.testing.assert_allclose(freq_p / n_trials, freq_r / n_trials, atol=0.04)


def test_power_of_choice_updates_loss_from_responded_clients():
    poc = PowerOfChoiceClientSelection(3, 1, d=2, seed=0)
    stats = _round_stats([1], losses={1: [2.0, 4.0]})
    poc.on_round_end(stats)
    assert poc.last_loss[1] == pytest.approx(3.0)
    assert np.isinf(poc.last_loss[0])
    assert np.isinf(poc.last_loss[2])


def test_power_of_choice_non_finite_mean_loss_raises():
    poc = PowerOfChoiceClientSelection(3, 1, d=2, seed=0)
    with pytest.raises(ValueError, match="not finite"):
        poc.on_round_end(_round_stats([1], losses={1: [1.0, np.nan]}))


def test_simpleavg_is_an_equal_mean():
    pio = ParamIO(residency="server", exchange="params", apply_to="server")
    info = _round_stats([0, 1])
    updates = [np.array([0.0]), np.array([1.0])]
    out = SimpleAvg(2).aggregate(info, None, updates, pio)
    np.testing.assert_allclose(out, [0.5])


def test_compatibility_table_warns_on_exact_class_pairs():
    weighted = RandomClientSelection(
        4, 2, weights={0: 1, 1: 2, 2: 3, 3: 4}
    )
    msgs = list(iter_selector_aggregator_warnings(weighted, [FedAvg(4)]))
    assert len(msgs) == 1
    assert "counts data size twice" in msgs[0]
    assert "Use SimpleAvg" in msgs[0]
    assert list(iter_selector_aggregator_warnings(weighted, [SimpleAvg(4)])) == []
    uniform = RandomClientSelection(4, 2)
    assert list(iter_selector_aggregator_warnings(uniform, [FedAvg(4)])) == []
    rr = RoundRobinClientSelection(4, 2, seed=0)
    rr_msgs = list(iter_selector_aggregator_warnings(rr, [FedAvg(4)]))
    assert rr_msgs
    assert "plain mean" in rr_msgs[0]
    assert "Use SimpleAvg" in rr_msgs[0]
    assert list(iter_selector_aggregator_warnings(rr, [SimpleAvg(4)])) == []
    poc_uniform = PowerOfChoiceClientSelection(4, 2, d=3, seed=0)
    assert list(iter_selector_aggregator_warnings(poc_uniform, [FedAvg(4)])) == []
    poc_weighted = PowerOfChoiceClientSelection(
        4, 2, d=3, seed=0, weights={0: 1, 1: 2, 2: 3, 3: 4}
    )
    assert list(iter_selector_aggregator_warnings(poc_weighted, [FedAvg(4)])) != []
    assert list(iter_selector_aggregator_warnings(poc_weighted, [SimpleAvg(4)])) == []


def test_server_logs_pairing_warning(caplog):
    dataset = _tiny_dataset(2)
    caplog.set_level(logging.WARNING)
    _DummyServer(
        interaction_dataset=dataset,
        client_algorithm=_DummyClient(),
        num_rounds=1,
        param_ios={
            "w": ParamIO(residency="server", exchange="params", apply_to="server"),
        },
        client_selection_strategy=RandomClientSelection(
            dataset.num_clients,
            1,
            weights=dataset.client_partition.client_train_sizes,
        ),
        aggregation_strategies=FedAvg(dataset.num_clients),
        device="cpu",
        training_framework="numpy",
    )
    assert "already samples by client size" in caplog.text


def test_server_does_not_warn_weighted_sampling_with_simpleavg(caplog):
    dataset = _tiny_dataset(2)
    caplog.set_level(logging.WARNING)
    _DummyServer(
        interaction_dataset=dataset,
        client_algorithm=_DummyClient(),
        num_rounds=1,
        param_ios={
            "w": ParamIO(residency="server", exchange="params", apply_to="server"),
        },
        client_selection_strategy=RandomClientSelection(
            dataset.num_clients,
            1,
            weights=dataset.client_partition.client_train_sizes,
        ),
        aggregation_strategies=SimpleAvg(dataset.num_clients),
        device="cpu",
        training_framework="numpy",
    )
    assert "already samples by client size" not in caplog.text


def test_selectors_ignore_the_random_module():
    kwargs = dict(num_clients=12, number_selected_clients=3, seed=0)
    random.seed(1)
    a = RandomClientSelection(**kwargs).select(0)
    random.seed(99)
    b = RandomClientSelection(**kwargs).select(0)
    assert a == b
    random.seed(1)
    c = RoundRobinClientSelection(**kwargs).select(0)
    random.seed(99)
    d = RoundRobinClientSelection(**kwargs).select(0)
    assert c == d
