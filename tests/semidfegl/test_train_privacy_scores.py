"""Local train uploads, privacy replacement and ego-dot scoring."""

import numpy as np

from src.fedreclab.methods.semidfegl.semidfegl_client import SemiDFEGLClientAlgorithm


def test_privacy_off_upload_is_exactly_loss_updated_rows(make_server):
    server = make_server()
    server.current_round = 0
    server._on_round_start_server()
    payload = server._build_server_payload(0)
    response = server._train_client(0, current_round=0)
    upload = response.updates_packages["item_embeddings"].param_value
    rows = set(int(r) for r in upload["rows"])
    expected = set(int(i) for i in payload["train_item_ids"])
    expected.update(int(i) for i in payload["fake_item_ids"])
    expected.update(int(i) for i in payload["bpr_negative_ids"])
    assert rows == expected
    assert server.client_algorithm.apply_privacy(None, response) is response


def test_privacy_pseudo_matches_single_positive_and_replaces_row(make_server):
    client = SemiDFEGLClientAlgorithm(
        embedding_dim=4,
        num_layers=1,
        num_pseudo_items=1,
    )
    server = make_server(client=client)
    server.current_round = 0
    server._on_round_start_server()
    # Train client 0; apply privacy afterwards.
    request_payload = server._build_server_payload(0)
    # Build a real training request through the server path, then privacy.
    from src.fedreclab.federated.base_client import ClientTrainingRequest

    client_state = server.client_states[0]
    training_data = server.interaction_dataset.get_client_view(
        0, include_validation=False, include_test=False
    )
    request = ClientTrainingRequest(
        client_id=0,
        current_round=0,
        users_in_client=server.interaction_dataset.client_partition.client_ids_2_user_ids[0],
        client_state=client_state,
        training_data=training_data,
        global_payload=request_payload,
        extra_payload=server.extra_payload,
        param_ios=server.param_ios,
        seed=server.seed,
    )
    response = client.local_train(request)
    raw = response.updates_packages["item_embeddings"].param_value
    raw_rows = np.asarray(raw["rows"]).copy()
    raw_values = np.asarray(raw["values"], dtype=np.float32).copy()
    train_ids = set(int(i) for i in request_payload["train_item_ids"])
    pos_mask = np.array([int(r) in train_ids for r in raw_rows])
    assert pos_mask.any()
    # One positive embedding: variance 0, so the pseudo copy equals that row
    # when there is a single positive; with two positives it matches the
    # empirical diagonal Gaussian. Force one positive for the equality case.
    if int(pos_mask.sum()) != 1:
        # Keep only the first positive row for the match check by rewriting.
        first_pos = int(np.flatnonzero(pos_mask)[0])
        raw["rows"] = raw_rows[[first_pos]]
        raw["values"] = raw_values[[first_pos]]
        response.updates_packages["item_embeddings"] = client._update_package(
            request, "item_embeddings", raw
        )
        raw_rows = np.asarray(raw["rows"]).copy()
        raw_values = np.asarray(raw["values"], dtype=np.float32).copy()
        train_ids = {int(raw_rows[0])}
        response.updated_client_state.metadata["train_item_ids"] = np.asarray(
            list(train_ids), dtype=np.int64
        )

    response = client.apply_privacy(request, response)
    upload = response.updates_packages["item_embeddings"].param_value
    rows = np.asarray(upload["rows"])
    values = np.asarray(upload["values"], dtype=np.float32)
    assert rows.shape[0] == raw_rows.shape[0] + 1
    np.testing.assert_allclose(values[0], raw_values[0])
    np.testing.assert_allclose(values[-1], raw_values[0])
    assert int(rows[-1]) not in set(int(r) for r in raw_rows)
    assert int(rows[-1]) not in train_ids

    # Sole uploader of the pseudo id replaces the global row under params.
    prev = np.asarray(server.global_parameters["item_embeddings"], dtype=np.float32).copy()
    pseudo_id = int(rows[-1])
    server.current_round_statistics = server._update_statistics(0, [0], [response])
    server.aggregate_parameters([response])
    after = np.asarray(server.global_parameters["item_embeddings"], dtype=np.float32)
    np.testing.assert_allclose(after[pseudo_id], values[-1])
    assert not np.allclose(after[pseudo_id], prev[pseudo_id])


def test_compute_scores_is_ego_dot_product(make_server):
    server = make_server()
    server.current_round = 0
    server._on_round_start_server()
    server.run_round(0)
    egos = np.asarray(server.global_parameters["ego_embeddings"], dtype=np.float32)
    items = np.asarray(server.global_parameters["item_embeddings"], dtype=np.float32)
    user_id = server._client_to_user[0]
    candidates = np.arange(server.interaction_dataset.num_items, dtype=np.int64)
    scores = server.client_algorithm.compute_scores(
        client_id=0,
        candidate_items_per_user={user_id: candidates},
        global_parameters=server.global_parameters,
        client_state=server.client_states[0],
        extra_payload=server.extra_payload,
    )
    expected = items[candidates] @ egos[user_id]
    np.testing.assert_allclose(scores[user_id], expected)


def test_validation_edge_is_not_a_graph_edge(make_server, make_tiny_dataset):
    import pandas as pd

    from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
    from src.fedreclab.data.dataset import InteractionDataset
    from src.fedreclab.data.split import RatioSplitStrategy

    rows = []
    for user_id in range(3):
        rows.append(
            {
                "user_id": user_id,
                "item_id": user_id,
                "rating": 1.0,
                "split": "train",
            }
        )
        rows.append(
            {
                "user_id": user_id,
                "item_id": user_id + 3,
                "rating": 1.0,
                "split": "val",
            }
        )
    dataset = InteractionDataset(
        interaction_data=pd.DataFrame(rows),
        splitting_strategy=RatioSplitStrategy(
            include_validation=True, include_test=False
        ),
        client_partitioning_strategy=CrossDeviceSplittingStrategy(),
        k_core=0,
        include_validation=True,
        include_test=False,
        splits_included=True,
        time_col=None,
    )
    dataset.partition_clients()
    server = make_server(
        dataset=dataset,
        num_groups=1,
        n_fake=1,
        num_rounds=1,
    )
    server.current_round = 0
    server._on_round_start_server()
    train_items = set(int(i) for i in server._user_train_items[0])
    assert 0 in train_items
    assert 3 not in train_items
    payload = server._build_server_payload(0)
    assert 3 not in set(int(i) for i in payload["train_item_ids"])
