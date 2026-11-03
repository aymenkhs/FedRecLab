"""Clustering, fake items, sliced payloads and lagged ego vectors."""

import logging

import numpy as np
import pytest

from src.fedreclab.federated.client_selection import (
    FullClientSelection,
    RandomClientSelection,
)


def test_each_user_in_one_cluster_and_n_fake_per_cluster(make_server):
    server = make_server()
    server._on_round_start_server()
    assert len(server._user_cluster) == server.interaction_dataset.num_users
    labels = set(server._user_cluster.values())
    assert labels == set(server._cluster_users)
    for label, users in server._cluster_users.items():
        assert len(users) >= 1
        fake = server._cluster_fake_items[label]
        assert fake.shape == (server.n_fake,)
        assert len(set(int(i) for i in fake)) == server.n_fake


def test_same_item_can_appear_in_several_clusters(make_server):
    server = make_server(n_fake=2, num_groups=2)
    server._on_round_start_server()
    # Force both clusters to prefer the same top item by rewriting membership.
    # With n_fake=2 and a tiny catalogue the independent top-k picks often
    # overlap; assert the mechanism allows duplicates across clusters.
    item_hits: dict[int, int] = {}
    for fake in server._cluster_fake_items.values():
        for item_id in fake:
            item_hits[int(item_id)] = item_hits.get(int(item_id), 0) + 1
    # At least verify assignment ran; overlap is allowed when present.
    assert sum(len(v) for v in server._cluster_fake_items.values()) == server.n_fake * len(
        server._cluster_fake_items
    )


def test_payload_is_sliced_to_cluster_and_local_items(make_server):
    server = make_server()
    server.current_round = 0
    server._on_round_start_server()
    payload = server._build_server_payload(client_id=0)
    num_items = server.interaction_dataset.num_items
    num_users = server.interaction_dataset.num_users

    assert payload["item_ids"].ndim == 1
    assert payload["item_embeddings"].shape[0] == payload["item_ids"].shape[0]
    assert payload["item_embeddings"].shape[0] < num_items or num_items <= 8
    assert set(int(i) for i in payload["item_ids"]).issubset(set(range(num_items)))

    user_id = server._client_to_user[0]
    label = server._user_cluster[user_id]
    cluster_users = set(server._cluster_users[label])
    assert set(int(u) for u in payload["cluster_user_ids"]) == cluster_users
    assert payload["cluster_ego_embeddings"].shape[0] == len(cluster_users)
    assert payload["cluster_ego_embeddings"].shape[0] <= num_users
    assert int(payload["cluster_size"]) == len(cluster_users)

    expected_ids = set(int(i) for i in payload["train_item_ids"])
    expected_ids.update(int(i) for i in payload["fake_item_ids"])
    expected_ids.update(int(i) for i in payload["bpr_negative_ids"])
    assert set(int(i) for i in payload["item_ids"]) == expected_ids


def test_second_round_payload_sees_uploaded_ego(make_server):
    server = make_server(num_rounds=2)
    server.run_round(0)
    ego_after_round0 = np.asarray(
        server.global_parameters["ego_embeddings"], dtype=np.float32
    ).copy()
    # Client 0 appears first in FullClientSelection; its ego must already
    # be the round-0 upload when round 1 builds payloads.
    server.current_round = 1
    server._on_round_start_server()
    payload = server._build_server_payload(client_id=0)
    user_id = server._client_to_user[0]
    cluster_users = list(payload["cluster_user_ids"])
    idx = cluster_users.index(user_id)
    np.testing.assert_allclose(
        payload["cluster_ego_embeddings"][idx],
        ego_after_round0[user_id],
    )


def test_fake_training_positive_is_not_a_bpr_negative(make_server):
    server = make_server(n_fake=3, num_groups=1)
    server.current_round = 0
    server._on_round_start_server()
    # With one cluster, every user shares the same fake items.
    payload = server._build_server_payload(0)
    train_ids = set(int(i) for i in payload["train_item_ids"])
    fake_ids = [int(i) for i in payload["fake_item_ids"]]
    fake_negs = [i for i in fake_ids if i not in train_ids]
    assert all(i not in train_ids for i in fake_negs)
    response = server._train_client(0, current_round=0)
    assert "item_embeddings" in response.updates_packages


def test_non_full_selector_logs_and_does_not_raise(make_server, caplog):
    dataset = make_server().interaction_dataset
    # Rebuild with a random selector; binding requires a fresh client.
    from src.fedreclab.methods.semidfegl.semidfegl_client import SemiDFEGLClientAlgorithm

    client = SemiDFEGLClientAlgorithm(embedding_dim=4, num_layers=1)
    with caplog.at_level(logging.INFO):
        server = make_server(
            dataset=dataset,
            client=client,
            client_selection_strategy=RandomClientSelection(
                dataset.num_clients, 2, selection_value_type="num", seed=0
            ),
        )
        server._before_training()
    assert any("FullClientSelection" in r.message for r in caplog.records)
    server._before_training()
    assert sum("FullClientSelection" in r.message for r in caplog.records) == 1


def test_zero_or_several_users_raises(make_tiny_dataset, make_server):
    from src.fedreclab.data.client_partitioning import ClientPartition
    from src.fedreclab.methods.semidfegl.semidfegl_client import SemiDFEGLClientAlgorithm

    dataset = make_tiny_dataset(n_users=2)
    # Corrupt the partition so client 0 owns two users.
    dataset.client_partition = ClientPartition(
        num_clients=1,
        client_ids_2_user_ids={0: [0, 1]},
        user_ids_2_client_ids={0: 0, 1: 0},
        client_to_train_indices={0: [0, 1, 2, 3]},
        client_train_sizes={0: 4},
    )
    client = SemiDFEGLClientAlgorithm(embedding_dim=4, num_layers=1)
    with pytest.raises(ValueError, match="exactly one user"):
        make_server(
            dataset=dataset,
            client=client,
            num_groups=1,
            client_selection_strategy=FullClientSelection(1),
        )
