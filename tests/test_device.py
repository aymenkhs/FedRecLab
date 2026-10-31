"""Server.device is resolved before init and copied onto the client algorithm."""

import pandas as pd

from src.fedreclab.data.client_partitioning import CrossDeviceSplittingStrategy
from src.fedreclab.data.dataset import InteractionDataset
from src.fedreclab.data.split import RatioSplitStrategy
from src.fedreclab.federated.aggregation import SimpleAvg
from src.fedreclab.federated.base_client import BaseClientAlgorithm
from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.client_selection import FullClientSelection
from src.fedreclab.federated.utils import ParamIO


def _tiny_dataset():
    df = pd.DataFrame(
        {
            "user_id": [0, 1],
            "item_id": [0, 1],
            "rating": [1.0, 1.0],
            "split": ["train", "train"],
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


class _RecordingClient(BaseClientAlgorithm):
    def __init__(self):
        super().__init__()
        self.device_at_local_init = None

    def initialize_local_parameters(
        self,
        client_id,
        total_num_users,
        num_users,
        num_items,
        client_train_data_view,
        users_in_client,
    ):
        self.device_at_local_init = self.device
        return {}


class _RecordingServer(BaseFederatedServer):
    def __init__(self, *args, **kwargs):
        self.device_at_server_init = None
        self.framework_at_server_init = None
        super().__init__(*args, **kwargs)

    def _initialize_server_parameters(self):
        self.device_at_server_init = self.device
        self.framework_at_server_init = self.training_framework
        return {"w": None}


def test_device_is_available_during_server_and_client_init():
    dataset = _tiny_dataset()
    client = _RecordingClient()
    server = _RecordingServer(
        interaction_dataset=dataset,
        client_algorithm=client,
        num_rounds=1,
        param_ios={
            "w": ParamIO(residency="server", exchange="params", apply_to="server"),
        },
        client_selection_strategy=FullClientSelection(dataset.num_clients),
        aggregation_strategies=SimpleAvg(dataset.num_clients),
        device="cpu",
    )
    assert server.device_at_server_init == "cpu"
    assert server.framework_at_server_init == "torch"
    assert client.device_at_local_init == "cpu"
    assert client.device == "cpu"
