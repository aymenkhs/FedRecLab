"""GPFedRec relation graph: neighbour rules, normalisation, propagation, failures."""

import numpy as np
import pytest

from src.fedreclab.methods.gpfedrec.gpfedrec import (
    build_relation_graph,
    cosine_similarity_matrix,
    propagate_tables,
)

# a and b point almost the same way; c is orthogonal to a.
TABLES = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]], dtype=np.float32)


def test_cosine_similarity_is_scale_invariant_with_unit_diagonal():
    s = cosine_similarity_matrix(TABLES)
    scaled = cosine_similarity_matrix(TABLES * np.array([[3.0], [0.5], [7.0]], dtype=np.float32))
    np.testing.assert_allclose(s, scaled, atol=1e-6)
    np.testing.assert_array_equal(np.diag(s), np.ones(3))
    np.testing.assert_allclose(s, s.T)
    assert s[0, 1] == pytest.approx(1.0 / np.sqrt(1.01), abs=1e-6)
    assert s[0, 2] == pytest.approx(0.0, abs=1e-7)


def test_similarity_rule_keeps_similar_users_and_self():
    # mean(S) ~ 0.577: a and b pick each other and themselves; c only itself.
    adjacency = build_relation_graph(cosine_similarity_matrix(TABLES), "similarity", 1.0)
    expected = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(adjacency, expected)


def test_reference_distance_rule_keeps_dissimilar_users_and_excludes_self():
    # mean(1 - S) ~ 0.423: a and b pick c; c picks a and b; nobody picks itself.
    adjacency = build_relation_graph(
        cosine_similarity_matrix(TABLES), "reference_distance", 1.0
    )
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    np.testing.assert_allclose(adjacency, expected)


@pytest.mark.parametrize("rule, threshold", [("similarity", 2.0), ("reference_distance", 10.0)])
def test_rows_without_neighbours_fall_back_to_a_self_loop(rule, threshold):
    adjacency = build_relation_graph(cosine_similarity_matrix(TABLES), rule, threshold)
    np.testing.assert_array_equal(adjacency, np.eye(3))


def test_rows_are_normalised_to_sum_to_one():
    rng = np.random.default_rng(0)
    tables = rng.normal(size=(6, 5)).astype(np.float32)
    for rule in ("similarity", "reference_distance"):
        adjacency = build_relation_graph(cosine_similarity_matrix(tables), rule, 0.5)
        np.testing.assert_allclose(adjacency.sum(axis=1), np.ones(6))


def test_two_layers_equal_squared_adjacency_times_tables():
    rng = np.random.default_rng(1)
    tables = rng.normal(size=(4, 6)).astype(np.float32)
    adjacency = build_relation_graph(cosine_similarity_matrix(tables), "similarity", 0.5)
    out = propagate_tables(adjacency, tables, layers=2)
    a32 = adjacency.astype(np.float32)
    np.testing.assert_allclose(out, a32 @ a32 @ tables, rtol=1e-6)
    assert out.dtype == np.float32


def test_zero_norm_upload_raises():
    with pytest.raises(ValueError, match="zero norm"):
        cosine_similarity_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_non_finite_upload_raises():
    with pytest.raises(ValueError, match="non-finite"):
        cosine_similarity_matrix(np.array([[np.nan, 0.0], [1.0, 0.0]]))


def test_unknown_rule_and_zero_layers_raise():
    s = cosine_similarity_matrix(TABLES)
    with pytest.raises(ValueError, match="neighbor_rule"):
        build_relation_graph(s, "knn", 1.0)
    with pytest.raises(ValueError, match="layers"):
        propagate_tables(np.eye(3), TABLES, layers=0)
