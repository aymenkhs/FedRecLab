from typing import Dict, Any

import numpy as np

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


class FedPerGNN(BaseFederatedServer):

    def __init__(
        self,
        hidden_dim: int = 64,
        history_len: int = 20,
        neighbor_len: int = 10,
        **kwargs: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len

        # One logical parameter group for all server-side model parameters.
        param_ios = {
            "model_state": ParamIO(
                residency="server",
                exchange="params",
                apply_to="server",
            ),
        }

        super().__init__(param_ios=param_ios, **kwargs)

    def _initialize_server_parameters(self) -> Dict[str, Any]:
        """
        Create initial global model parameters using a temporary model.

        The temporary model is never used directly for training; only its
        server-side state is extracted and stored in ``self.global_parameters``.
        """
        model = FedPerGNNModel(
            num_items=self.interaction_dataset.num_items,
            hidden_dim=self.hidden_dim,
            history_len=self.history_len,
            neighbor_len=self.neighbor_len,
        )
        server_state = model.get_server_state()

        # Ensure everything is converted to numpy for aggregation backends.
        numpy_state: Dict[str, Any] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in server_state.items()
        }

        return {"model_state": numpy_state}

    def _on_round_start_server(self) -> None:
        """
        Hook called at the beginning of each round.

        Users can override this method in a subclass to implement expansion
        logic and populate ``self.extra_payload`` with per-client structures
        consumed by the FedPerGNN client algorithm.
        """
        # Default implementation: do nothing. Expansion is opt-in.
        return

