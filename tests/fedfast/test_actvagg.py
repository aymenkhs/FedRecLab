"""FedFast-private ActvAGG subordinate updates and reclustering."""

import numpy as np

from src.fedreclab.methods.fedfast._actvagg import _ActvAgg, _FedFastClusterState


def test_subordinate_receives_discounted_delegate_delta():
    state = _FedFastClusterState(num_clusters=2, seed=0)
    helper = _ActvAgg(state)
    # Two tight pairs so k-means keeps {0,1} and {2,3}.
    emb0 = {
        0: np.array([0.0, 0.0]),
        1: np.array([0.1, 0.0]),
        2: np.array([10.0, 0.0]),
        3: np.array([10.1, 0.0]),
    }
    helper.snapshot_user_embeddings(emb0)

    emb1 = {cid: vec.copy() for cid, vec in emb0.items()}
    emb1[0] = np.array([2.0, 0.0])  # delegate delta = (2, 0)

    pending, groups = helper.compute_subordinate_updates(
        emb1, selected_client_ids=[0], current_round=0
    )
    assert len(groups) == 2
    # Client 1 should share a cluster with 0 and get gamma=1 * delta.
    assert 1 in pending
    np.testing.assert_allclose(pending[1], np.array([0.1, 0.0]) + np.array([2.0, 0.0]))
    assert 0 not in pending
    assert state.last_gamma == 1.0


def test_gamma_decays_with_round():
    state = _FedFastClusterState(num_clusters=1, seed=0)
    helper = _ActvAgg(state)
    emb = {0: np.zeros(2), 1: np.zeros(2)}
    helper.snapshot_user_embeddings(emb)
    emb_after = {0: np.array([1.0, 0.0]), 1: np.zeros(2)}
    helper.compute_subordinate_updates(emb_after, [0], current_round=1)
    np.testing.assert_allclose(state.last_gamma, np.exp(-1.0))
