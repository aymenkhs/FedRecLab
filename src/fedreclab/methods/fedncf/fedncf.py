""" Federated Neural Collaborative Filtering (FedNCF) server.

Source : Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.

Minimal BaseFederatedServer subclass.  The only required override is
_initialize_server_parameters(), which creates the initial server-side
weights. All round orchestration, payload building, aggregation, and statistics
are handled by the base class through the ParamIO / aggregation system.
No hooks (_on_round_start_server, _on_round_end_server, etc.) are
needed because FedNCF has no custom per-round server logic.
"""

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering


class FedNCF(BaseFederatedServer):
    """Federated NCF. Server holds item tables, MLP and output weights.

    User embedding tables and their optimiser state live in ClientState.
    """


    def __init__(self, predictive_factor: int = 32, **kwargs):
        self.predictive_factor = predictive_factor

        # TODO: Add a ParamIO for local parameters too with exchange=None
        # TODO: Offer the option for user to provide a custom io_spec and keep this one as default
        # Three logical parameter groups for te server
        param_ios = {
            "item_embeddings": ParamIO(residency="server", exchange="params", apply_to="server"),
            "shared_layers":   ParamIO(residency="server", exchange="params", apply_to="server"),
            "output_weights":  ParamIO(residency="server", exchange="params", apply_to="server"),
        }

        super().__init__(param_ios=param_ios, **kwargs)

    def _initialize_server_parameters(self):
        """Create initial server weights using a temporary skeleton."""
        model = NeuralCollaborativeFiltering(
            num_items=self.interaction_dataset.num_items,
            predictive_factor=self.predictive_factor,
        )
        return model.get_server_state_groups()
