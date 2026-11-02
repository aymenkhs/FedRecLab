"""GPFedRec server: graph step, aggregator boundary, r_i store, payload, config checks."""

import numpy as np
import pytest
import torch

from src.fedreclab.federated.aggregation import BaseAggregation, FedAtt, SimpleAvg
from src.fedreclab.federated.client_selection import (
    BaseClientSelection,
    RandomClientSelection,
)
from src.fedreclab.methods.gpfedrec.gpfedrec import (
    GPFedRec,
    build_relation_graph,
    cosine_similarity_matrix,
    default_param_ios,
)

LATENT = 4


class _RecordingGPFedRec(GPFedRec):
    """Keeps a copy of the uploads seen by the graph step."""

    def _aggregate_group(self, param_name, param_io, updates, participant_ids):
        self.recorded_uploads = [torch.as_tensor(u).clone() for u in updates]
        self.recorded_ids = list(participant_ids)
        super()._aggregate_group(param_name, param_io, updates, participant_ids)


class _ScriptedSelection(BaseClientSelection):
    """Selects one client per round from a fixed schedule."""

    def __init__(self, num_clients, schedule):
        super().__init__(num_clients, 1, selection_value_type="num")
        self.schedule = schedule

    def select(self, current_round=0):
        return [self.schedule[current_round]]


class _FlatAggregator(BaseAggregation):
    """Returns the mean flattened, which is the wrong shape for a table."""

    def aggregate(self, round_info, previous_parameters, updates, param_io):
        return torch.as_tensor(updates).mean(dim=0).reshape(-1)


def _expected_r(uploads, rule="similarity", threshold=1.0):
    flat = np.stack([u.numpy().reshape(-1) for u in uploads])
    adjacency = build_relation_graph(cosine_similarity_matrix(flat), rule, threshold)
    return (adjacency.astype(np.float32) @ flat).reshape((len(uploads),) + tuple(uploads[0].shape))


def test_default_param_ios_declare_only_item_embeddings():
    pios = default_param_ios()
    assert list(pios) == ["item_embeddings"]
    pio = pios["item_embeddings"]
    assert (pio.residency, pio.exchange, pio.apply_to, pio.layout) == (
        "server", "params", "server", "dense",
    )


@pytest.mark.parametrize("rule", ["similarity", "reference_distance"])
def test_simple_avg_gives_mean_of_propagated_tables(make_server, rule):
    server = make_server(server_class=_RecordingGPFedRec, neighbor_rule=rule)
    server.run_round(0)
    expected = _expected_r(server.recorded_uploads, rule)
    for row, client_id in enumerate(server.recorded_ids):
        np.testing.assert_allclose(
            server.personalized_item_embeddings[client_id].numpy(),
            expected[row], rtol=1e-5, atol=1e-6,
        )
    np.testing.assert_allclose(
        server.global_parameters["item_embeddings"].numpy(),
        expected.mean(axis=0), rtol=1e-5, atol=1e-6,
    )


def test_global_table_keeps_type_shape_dtype_and_device(make_server):
    server = make_server()
    before = server.global_parameters["item_embeddings"]
    server.run_round(0)
    after = server.global_parameters["item_embeddings"]
    assert isinstance(after, torch.Tensor)
    assert after.shape == before.shape
    assert after.dtype == before.dtype
    assert after.device == before.device
    server.run_round(1)
    assert server.global_parameters["item_embeddings"].shape == before.shape


def test_fedatt_receives_table_shaped_updates(make_server, make_tiny_dataset):
    dataset = make_tiny_dataset()
    server = make_server(
        dataset=dataset, aggregation_strategies=FedAtt(num_clients=dataset.num_clients)
    )
    server.run_round(0)
    table = server.global_parameters["item_embeddings"]
    assert table.shape == (dataset.num_items, LATENT)
    assert torch.isfinite(table).all()


def test_wrong_shape_from_aggregator_raises(make_server, make_tiny_dataset):
    dataset = make_tiny_dataset()
    server = make_server(
        dataset=dataset, aggregation_strategies=_FlatAggregator(dataset.num_clients)
    )
    with pytest.raises(ValueError, match="shape"):
        server.run_round(0)


def test_stored_r_i_are_owned_cpu_float32_tables(make_server):
    server = make_server()
    server.run_round(0)
    store = server.personalized_item_embeddings
    assert sorted(store) == list(range(server.num_clients))
    pointers = set()
    for r_i in store.values():
        assert r_i.shape == server.global_parameters["item_embeddings"].shape
        assert r_i.dtype == torch.float32
        assert r_i.device.type == "cpu"
        assert r_i._base is None
        pointers.add(r_i.untyped_storage().data_ptr())
    assert len(pointers) == len(store)


def test_non_participants_keep_their_last_r_i(make_server, make_tiny_dataset):
    dataset = make_tiny_dataset()
    server = make_server(
        dataset=dataset,
        client_selection_strategy=_ScriptedSelection(dataset.num_clients, [0, 1]),
        client_kwargs={"user_lr": 1.0},
    )
    server.run_round(0)
    assert set(server.personalized_item_embeddings) == {0}
    payload = server._build_server_payload(client_id=1)
    assert payload["personalized_item_embeddings"] is payload["item_embeddings"]
    r_0 = server.personalized_item_embeddings[0].clone()

    server.run_round(1)
    assert set(server.personalized_item_embeddings) == {0, 1}
    assert torch.equal(server.personalized_item_embeddings[0], r_0)


def test_payload_carries_q_global_and_the_clients_own_r_i(make_server):
    server = make_server()
    server.run_round(0)
    for client_id in range(server.num_clients):
        payload = server._build_server_payload(client_id=client_id)
        assert payload["item_embeddings"] is server.global_parameters["item_embeddings"]
        assert (
            payload["personalized_item_embeddings"]
            is server.personalized_item_embeddings[client_id]
        )


def test_q_global_is_written_through_the_server_apply_path(make_server, monkeypatch):
    server = make_server()
    calls = []
    original = server._update_server_parameter

    def spy(*args, **kwargs):
        calls.append(kwargs.get("param_name", args[0] if args else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "_update_server_parameter", spy)
    server.run_round(0)
    assert calls == ["item_embeddings"]


@pytest.mark.parametrize(
    "fields",
    [{"layout": "rows"}, {"apply_to": "custom"}, {"exchange": "delta"}],
)
def test_overriding_the_item_embeddings_declaration_raises(make_server, fields):
    with pytest.raises(ValueError, match="GPFedRec requires item_embeddings"):
        make_server(param_io_overrides={"item_embeddings": fields})


def test_participation_ratio_disagreeing_with_selector_raises(make_server, make_tiny_dataset):
    dataset = make_tiny_dataset(10)
    with pytest.raises(ValueError, match="participation_ratio=1.0 implies 10"):
        make_server(
            dataset=dataset,
            client_selection_strategy=RandomClientSelection(10, 0.3),
        )


def test_participation_ratio_matching_a_floored_selector_passes(make_server, make_tiny_dataset):
    # The selector floors 0.5 * 7 to 3 clients; the ratio implies the same 3.
    dataset = make_tiny_dataset(7)
    selector = RandomClientSelection(7, 0.5)
    assert selector.number_selected_clients == 3
    make_server(
        dataset=dataset,
        client_selection_strategy=selector,
        client_kwargs={"participation_ratio": 0.5},
    )


def test_participation_ratio_check_is_skipped_when_user_lr_is_set(make_server, make_tiny_dataset):
    dataset = make_tiny_dataset(10)
    make_server(
        dataset=dataset,
        client_selection_strategy=RandomClientSelection(10, 0.3),
        client_kwargs={"user_lr": 1.0},
    )


def test_latent_dim_mismatch_raises(make_server):
    with pytest.raises(ValueError, match="latent_dim"):
        make_server(latent_dim=LATENT + 1)
