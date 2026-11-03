'''Catalogue: items 0..3, categories 0:[a]  1:[b]  2:[a,b]  3:[c]
  -> weight matrix rows: 0=[1,0,0] 1=[0,1,0] 2=[.5,.5,0] 3=[0,0,1]
  -> total category mass per (a, b, c) = [1.5, 1.5, 1]
Users 0..3, groups f={0,1}, m={2,3}
Train: u0={0,1} u1={0} u2={0,2,3} u3={0,1,2,3}
  -> item counts [4,2,2,2], popularity phi = counts / 4 users = [1, .5, .5, .5]
  -> buckets (head_share=tail_share=.2): head={0} mid={1,2} tail={3}
  -> user activity u3 > u2 > u0 > u1 ; advantaged (top 25%) = {3}
  -> profile popularity u1=1.0 u0=.75 u2=.667 u3=.625
     -> groups: u1 blockbuster, u0/u2 diverse, u3 niche'''
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from src.fedreclab.data.metric import (
    BiasMetrics,
    CategoryMatrix,
    beyond_parity_metrics,
    category_user_metrics,
)

CATS = {0: ["action"], 1: ["romance"], 2: ["action", "romance"], 3: ["comedy"]}
CATALOG = np.arange(4)
GROUPS = {0: "f", 1: "f", 2: "m", 3: "m"}

RANK_MIXED = {
    0: np.array([0, 1, 2, 3]),
    1: np.array([3, 2, 1, 0]),
    2: np.array([1, 0, 3, 2]),
    3: np.array([2, 3, 0, 1]),
}
RANK_SPLIT = {
    0: np.array([0, 2, 1, 3]),
    1: np.array([0, 2, 3, 1]),
    2: np.array([1, 3, 0, 2]),
    3: np.array([1, 3, 2, 0]),
}
RANK_SAME = {u: np.array([0, 1, 2, 3]) for u in range(4)}



def train():
    rows = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def make_bm(train, **kw):
    args = dict(
        item_categories=CATS,
        user_groups=GROUPS,
        topk=[2],
        activity_top_frac=0.25,
        popularity_user_fracs=(0.25, 0.5, 0.25),
    )
    args.update(kw)
    return BiasMetrics(train, CATALOG, **args)


def kl(p, q, alpha=0.01):
    qt = [(1 - alpha) * qi + alpha * pi for pi, qi in zip(p, q)]
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, qt) if pi > 0)


# =====================================================================
# CategoryMatrix
# =====================================================================
class TestCategoryMatrix:
    def test_weights_split_over_multiple_categories(self):
        cm = CategoryMatrix(CATS, CATALOG)
        print(cm.categories)
        assert cm.categories == ["action", "romance", "comedy"]
        np.testing.assert_allclose(cm.W[2], [0.5, 0.5, 0])
        np.testing.assert_allclose(cm.W[3], [0, 0, 1])

    def test_total_mass(self):
        cm = CategoryMatrix(CATS, CATALOG)
        np.testing.assert_allclose(cm.total_mass, [1.5, 1.5, 1.0])

    def test_unknown_and_uncategorised_items_are_zero_rows(self):
        cm = CategoryMatrix({0: ["action"], 1: []}, [0, 1, 2])
        out = cm.weights(np.array([0, 1, 2, 99]))
        np.testing.assert_allclose(out, [[1], [0], [0], [0]])

    def test_duplicate_categories_are_deduplicated(self):
        cm = CategoryMatrix({0: ["action", "action", "romance"]}, [0])
        np.testing.assert_allclose(cm.W[0], [0.5, 0.5])

    def test_item_not_in_catalogue_ignored(self):
        cm = CategoryMatrix({0: ["action"], 42: ["z"]}, [0, 1])
        assert cm.W.shape == (2, 2)  # 'z' still registered as a category
        np.testing.assert_allclose(cm.W[:, 1], 0)

TestCategoryMatrixTest = TestCategoryMatrix()
TestCategoryMatrixTest.test_weights_split_over_multiple_categories()
TestCategoryMatrixTest.test_unknown_and_uncategorised_items_are_zero_rows()
TestCategoryMatrixTest.test_total_mass()
TestCategoryMatrixTest.test_duplicate_categories_are_deduplicated()
TestCategoryMatrixTest.test_item_not_in_catalogue_ignored()

# # =====================================================================
# # Category-aware per-user metrics (hand computed)
# # =====================================================================
class TestCategoryUserMetrics:
    def out(self):
        cm = CategoryMatrix(CATS, CATALOG)
        # top-2 = items [2, 0] -> weights [[.5,.5,0],[1,0,0]]
        return category_user_metrics(cm, np.array([2, 0, 1, 3]), [2], max_len=4)

    def test_cc(self, out):
        np.testing.assert_allclose(out["cc@2"], [0.75, 0.25, 0])

    def test_rcr(self, out):
        np.testing.assert_allclose(out["rcr@2"], [1.0, 1 / 3, 0])

    def test_cmap(self, out):
        np.testing.assert_allclose(out["cmap@2"], [2 / 3, 1 / 6, 0])

    def test_cdcg(self, out):
        np.testing.assert_allclose(
            out["cdcg@2"], [(0.5 + 1 / math.log2(3)) / 2, 0.25, 0]
        )

    def test_cdcg_log_base_changes_value(self):
        cm = CategoryMatrix(CATS, CATALOG)
        r = np.array([2, 0, 1, 3])
        e = category_user_metrics(cm, r, [2], 4, log_base=math.e)["cdcg@2"]
        # discount = ln(j+1): position 1 -> ln 2, position 2 -> ln 3
        np.testing.assert_allclose(e[0], (0.5 / math.log(2) + 1 / math.log(3)) / 2)

    def test_cmrr(self, out):
        np.testing.assert_allclose(out["cmrr@2"], [0.5, 0.25, 0])

    def test_crp(self, out):
        # R_c = floor([1.5,1.5,1]) = [1,1,1]; first item weights / 1
        np.testing.assert_allclose(out["crp"], [0.5, 0.5, 0])

    def test_crp_can_be_disabled(self):
        cm = CategoryMatrix(CATS, CATALOG)
        out = category_user_metrics(cm, np.array([0, 1]), [2], 4, compute_crp=False)
        assert "crp" not in out

    def test_empty_ranking_returns_empty(self):
        cm = CategoryMatrix(CATS, CATALOG)
        assert category_user_metrics(cm, np.array([], dtype=int), [2], 4) == {}

    def test_list_shorter_than_k_uses_actual_length(self):
        cm = CategoryMatrix(CATS, CATALOG)
        out = category_user_metrics(cm, np.array([0]), [3], 3)
        np.testing.assert_allclose(out["cc@3"], [1, 0, 0])

    def test_multiple_k(self):
        cm = CategoryMatrix(CATS, CATALOG)
        out = category_user_metrics(cm, np.array([0, 1, 3]), [1, 3], 3)
        np.testing.assert_allclose(out["cc@1"], [1, 0, 0])
        np.testing.assert_allclose(out["cc@3"], [1 / 3, 1 / 3, 1 / 3])

    def test_cmap_ranking_sensitive(self):
        cm = CategoryMatrix(CATS, CATALOG)
        a_first = category_user_metrics(cm, np.array([0, 1]), [2], 2)["cmap@2"][0]
        a_last = category_user_metrics(cm, np.array([1, 0]), [2], 2)["cmap@2"][0]
        assert a_first > a_last

    def test_non_ranking_metrics_order_invariant(self):
        cm = CategoryMatrix(CATS, CATALOG)
        x = category_user_metrics(cm, np.array([0, 1]), [2], 2)
        y = category_user_metrics(cm, np.array([1, 0]), [2], 2)
        np.testing.assert_allclose(x["cc@2"], y["cc@2"])
        np.testing.assert_allclose(x["rcr@2"], y["rcr@2"])


# # =====================================================================
# # Gender-balance score (GBS)
# # =====================================================================
class TestGBS:
    def test_gbs_cc_hand_computed(self, train):
        s, _ = make_bm(train).compute(RANK_SPLIT)
        # f: [.75,.25,0]  m: [0,.5,.5]  -> |diff| sum = .75+.25+.5
        assert s["gbs_cc@2"] == pytest.approx(1.5)

    def test_gbs_rcr_hand_computed(self, train):
        s, _ = make_bm(train).compute(RANK_SPLIT)
        # f: [1,1/3,0]  m: [0,2/3,1] -> 1 + 1/3 + 1
        assert s["gbs_rcr@2"] == pytest.approx(7 / 3)

    def test_gbs_crp_hand_computed(self, train):
        s, _ = make_bm(train).compute(RANK_SPLIT)
        # f first item=0 -> [1,0,0]; m first item=1 -> [0,1,0]
        assert s["gbs_crp"] == pytest.approx(2.0)

    def test_identical_groups_give_zero(self, train):
        s, _ = make_bm(train).compute(RANK_SAME)
        for key, val in s.items():
            if key.startswith("gbs_"):
                assert val == pytest.approx(0.0), key

    def test_all_gbs_keys_present(self, train):
        s, _ = make_bm(train).compute(RANK_SPLIT)
        for name in ["cc", "rcr", "cmap", "cdcg", "cmrr"]:
            assert f"gbs_{name}@2" in s
        assert "gbs_crp" in s

    def test_gbs_is_non_negative_and_label_symmetric(self, train):
        s1, _ = make_bm(train).compute(RANK_SPLIT)
        flipped = {u: ("m" if g == "f" else "f") for u, g in GROUPS.items()}
        s2, _ = make_bm(train, user_groups=flipped).compute(RANK_SPLIT)
        for k in s1:
            if k.startswith("gbs_"):
                assert s1[k] >= 0
                assert s1[k] == pytest.approx(s2[k])

    def test_details_table(self, train):
        _, d = make_bm(train).compute(RANK_SPLIT)
        t = d["category/cc@2"]
        assert list(t.index) == ["action", "romance", "comedy"]
        assert list(t.columns) == ["all", "f", "m", "abs_diff"]
        np.testing.assert_allclose(t["f"], [0.75, 0.25, 0])
        np.testing.assert_allclose(t["m"], [0, 0.5, 0.5])
        np.testing.assert_allclose(t["abs_diff"], [0.75, 0.25, 0.5])
        np.testing.assert_allclose(t["all"], [0.375, 0.375, 0.25])

    def test_without_groups_no_gbs_but_details_exist(self, train):
        s, d = make_bm(train, user_groups=None).compute(RANK_SPLIT)
        assert not any(k.startswith("gbs_") for k in s)
        assert list(d["category/cc@2"].columns) == ["all"]

    def test_without_categories_no_category_metrics(self, train):
        s, d = make_bm(train, item_categories=None).compute(RANK_SPLIT)
        assert not any(k.startswith(("gbs_", "calib_")) for k in s)
        assert not any(k.startswith("category/") for k in d)

    def test_missing_group_warns_and_skips(self, train):
        only_f = {0: RANK_SPLIT[0], 1: RANK_SPLIT[1]}
        with pytest.warns(UserWarning):
            s, _ = make_bm(train).compute(only_f)
        assert not any(k.startswith("gbs_") for k in s)

    def test_users_without_group_label_are_ignored_in_gap(self, train):
        groups = {0: "f", 1: "f", 2: "m"}  # user 3 unlabeled
        s, _ = make_bm(train, user_groups=groups).compute(RANK_SPLIT)
        # m = only user 2 -> m cc = [0, .5, .5]
        assert s["gbs_cc@2"] == pytest.approx(1.5)


# # =====================================================================
# # Beyond Parity
# # =====================================================================
def _bp_pairs():
    # users 0,1 = f ; user 2 = m
    return pd.DataFrame(
        [
            (0, 0, 4.0, 5.0),
            (1, 0, 2.0, 3.0),  # f item0: pred mean 3, rating mean 4
            (2, 0, 3.0, 3.0),  # m item0: pred 3, rating 3
            (0, 1, 2.0, 1.0),  # f item1
            (2, 1, 4.0, 2.0),  # m item1
            (0, 2, 5.0, 5.0),  # item2 only f -> dropped
        ],
        columns=["user_id", "item_id", "pred", "rating"],
    )


class TestBeyondParity:
    # item0: ea=-1 eb=0 ; item1: ea=+1 eb=+2 ; pred f=[3,2] pred m=[3,4]
    def test_hand_computed(self):
        out = beyond_parity_metrics(_bp_pairs(), GROUPS, "f", "m")
        assert out["bp_uval"] == pytest.approx(1.0)
        assert out["bp_uabs"] == pytest.approx(1.0)
        assert out["bp_uunder"] == pytest.approx(0.5)
        assert out["bp_uover"] == pytest.approx(0.5)
        assert out["bp_upar"] == pytest.approx(0.25)

    def test_symmetric_in_group_order(self):
        a = beyond_parity_metrics(_bp_pairs(), GROUPS, "f", "m")
        b = beyond_parity_metrics(_bp_pairs(), GROUPS, "m", "f")
        for k in a:
            assert a[k] == pytest.approx(b[k])

    def test_identical_groups_are_zero(self):
        rows = []
        for u, g in [(0, "f"), (1, "m")]:
            rows += [(u, 0, 3.0, 4.0), (u, 1, 2.0, 1.0)]
        df = pd.DataFrame(rows, columns=["user_id", "item_id", "pred", "rating"])
        out = beyond_parity_metrics(df, {0: "f", 1: "m"}, "f", "m")
        assert all(v == pytest.approx(0.0) for v in out.values())

    def test_no_item_rated_by_both_groups_returns_empty(self):
        df = pd.DataFrame(
            [(0, 0, 1.0, 1.0), (2, 1, 1.0, 1.0)],
            columns=["user_id", "item_id", "pred", "rating"],
        )
        assert beyond_parity_metrics(df, GROUPS, "f", "m") == {}

    def test_other_group_labels_ignored(self):
        groups = {**GROUPS, 9: "x"}
        df = pd.concat(
            [_bp_pairs(), pd.DataFrame([(9, 0, 100.0, 0.0)], columns=_bp_pairs().columns)]
        )
        a = beyond_parity_metrics(df, groups, "f", "m")
        b = beyond_parity_metrics(_bp_pairs(), GROUPS, "f", "m")
        for k in a:
            assert a[k] == pytest.approx(b[k])

    def test_one_group_absent_returns_empty(self):
        df = _bp_pairs()
        df = df[df["user_id"].isin([0, 1])]
        assert beyond_parity_metrics(df, GROUPS, "f", "m") == {}

    def test_empty_frame(self):
        df = pd.DataFrame(columns=["user_id", "item_id", "pred", "rating"])
        assert beyond_parity_metrics(df, GROUPS, "f", "m") == {}

    def test_through_compute(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED, eval_pairs=_bp_pairs())
        assert s["bp_uval"] == pytest.approx(1.0)

    def test_compute_skips_bp_without_groups_or_pairs(self, train):
        s1, _ = make_bm(train, user_groups=None).compute(RANK_MIXED, eval_pairs=_bp_pairs())
        s2, _ = make_bm(train).compute(RANK_MIXED)
        assert not any(k.startswith("bp_") for k in s1)
        assert not any(k.startswith("bp_") for k in s2)


# # =====================================================================
# # UGF
# # =====================================================================
class TestUGF:
    PUM = {
        0: {"ndcg@2": 0.2},
        1: {"ndcg@2": 0.4},
        2: {"ndcg@2": 0.6},
        3: {"ndcg@2": 0.9},
    }

    def test_activity_groups(self, train):
        assert make_bm(train).advantaged == {3}

    def test_activity_top_frac_ceil(self, train):
        assert make_bm(train, activity_top_frac=0.5).advantaged == {3, 2}
        assert make_bm(train, activity_top_frac=0.01).advantaged == {3}  # at least 1

    def test_ugf_hand_computed(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED, per_user_metrics=self.PUM)
        # adv = u3 (.9); others mean = .4
        assert s["ugf_ndcg@2"] == pytest.approx(0.5)

    def test_group_gap_hand_computed(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED, per_user_metrics=self.PUM)
        # f mean .3 ; m mean .75
        assert s["group_gap_ndcg@2"] == pytest.approx(0.45)

    def test_ugf_uses_all_metric_names(self, train):
        pum = {u: {"ndcg@2": v["ndcg@2"], "recall@2": 1.0} for u, v in self.PUM.items()}
        s, _ = make_bm(train).compute(RANK_MIXED, per_user_metrics=pum)
        assert s["ugf_recall@2"] == pytest.approx(0.0)

    def test_equal_performance_gives_zero(self, train):
        pum = {u: {"ndcg@2": 0.5} for u in range(4)}
        s, _ = make_bm(train).compute(RANK_MIXED, per_user_metrics=pum)
        assert s["ugf_ndcg@2"] == pytest.approx(0.0)
        assert s["group_gap_ndcg@2"] == pytest.approx(0.0)

    def test_no_ugf_when_all_users_advantaged(self, train):
        bm = make_bm(train, activity_top_frac=1.0)
        s, _ = bm.compute(RANK_MIXED, per_user_metrics=self.PUM)
        assert "ugf_ndcg@2" not in s

    def test_no_per_user_metrics_no_ugf(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED)
        assert not any(k.startswith(("ugf_", "group_gap_")) for k in s)


# # =====================================================================
# # Calibration
# # =====================================================================
class TestCalibration:
    def test_hand_computed_mean(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED)
        #the catagory dist for train and recolists(topk)
        expected = [
            kl([0.5, 0.5, 0], [0.5, 0.5, 0]),
            kl([1, 0, 0], [0.25, 0.25, 0.5]),
            kl([0.5, 1 / 6, 1 / 3], [0.5, 0.5, 0]),
            kl([0.375, 0.375, 0.25], [0.25, 0.25, 0.5]),
        ]
        assert s["calib_kl@2"] == pytest.approx(np.mean(expected))
        assert expected[0] == pytest.approx(0.0)

    def test_group_gap(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED)
        e = [
            kl([0.5, 0.5, 0], [0.5, 0.5, 0]),
            kl([1, 0, 0], [0.25, 0.25, 0.5]),
            kl([0.5, 1 / 6, 1 / 3], [0.5, 0.5, 0]),
            kl([0.375, 0.375, 0.25], [0.25, 0.25, 0.5]),
        ]
        assert s["calib_kl_gap@2"] == pytest.approx(abs(np.mean(e[:2]) - np.mean(e[2:])))

    def test_perfectly_calibrated_is_zero(self):
        tr = pd.DataFrame({"user_id": [0, 0], "item_id": [0, 1]})
        bm = BiasMetrics(tr, CATALOG, item_categories=CATS, topk=[2])
        s, _ = bm.compute({0: np.array([0, 1, 2, 3])})
        assert s["calib_kl@2"] == pytest.approx(0.0, abs=1e-12)

    def test_missing_category_in_recs_is_finite_thanks_to_smoothing(self):
        tr = pd.DataFrame({"user_id": [0], "item_id": [3]})  # profile = category c
        bm = BiasMetrics(tr, CATALOG, item_categories=CATS, topk=[2])
        s, _ = bm.compute({0: np.array([0, 1])})  # recs contain no c
        assert s["calib_kl@2"] == pytest.approx(-math.log(0.01))

    def test_alpha_parameter(self):
        tr = pd.DataFrame({"user_id": [0], "item_id": [3]})
        bm = BiasMetrics(tr, CATALOG, item_categories=CATS, topk=[2], calibration_alpha=0.1)
        s, _ = bm.compute({0: np.array([0, 1])})
        assert s["calib_kl@2"] == pytest.approx(-math.log(0.1))

    def test_rank_discount(self):
        tr = pd.DataFrame({"user_id": [0], "item_id": [0]})  # p = [1,0,0]
        w = np.array([1.0, 1 / math.log2(3)])
        q = w[0] * np.array([0, 0, 1]) + w[1] * np.array([0.5, 0.5, 0])
        q = q / q.sum()
        bm = BiasMetrics(
            tr, CATALOG, item_categories=CATS, topk=[2], calibration_rank_discount=True
        )
        s, _ = bm.compute({0: np.array([3, 2])})
        assert s["calib_kl@2"] == pytest.approx(kl([1, 0, 0], q))
        plain = BiasMetrics(tr, CATALOG, item_categories=CATS, topk=[2])
        s2, _ = plain.compute({0: np.array([3, 2])})
        assert s["calib_kl@2"] != pytest.approx(s2["calib_kl@2"])

    def test_non_negative(self, train):
        s, _ = make_bm(train).compute(RANK_MIXED)
        assert s["calib_kl@2"] >= 0

    def test_user_without_profile_skipped(self, train):
        rk = {**RANK_MIXED, 99: np.array([0, 1])}
        s, _ = make_bm(train).compute(rk)
        s0, _ = make_bm(train).compute(RANK_MIXED)
        assert s["calib_kl@2"] == pytest.approx(s0["calib_kl@2"])

    def test_profile_without_categorised_items_skipped(self):
        cats = {0: ["action"]}
        tr = pd.DataFrame({"user_id": [0], "item_id": [3]})  # item 3 uncategorised
        bm = BiasMetrics(tr, CATALOG, item_categories=cats, topk=[2])
        s, _ = bm.compute({0: np.array([0, 1])})
        assert "calib_kl@2" not in s

