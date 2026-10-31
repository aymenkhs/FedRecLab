""" Federated Neural Collaborative Filtering (FedNCF) server.

Source : Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.

Minimal BaseFederatedServer subclass.  The only required override is
_initialize_server_parameters(), which creates the initial server-side
weights. All round orchestration, payload building, aggregation, and statistics
are handled by the base class through the ParamIO / aggregation system.
No hooks (_on_round_start_server, _on_round_end_server, etc.) are
needed because FedNCF has no custom per-round server logic.

Training protocol lives on the client: paper default is implicit binary
labels and BCE (FedNCFClientAlgorithm). Rating-valued MSE is an opt-in
deviation, not the KBS experiments.

Faithful aggregation: ``SimpleAvg`` on ``item_embeddings`` (uniform mean
over the clients that updated each item; Algorithm 1 / MF-FedAvg). Neural
groups (``shared_layers``, ``output_weights``) use ordinary dense FedAvg
in the paper; passing one ``SimpleAvg`` for every group is an experiment
choice, not the paper default for the MLP.

Simulation shortcuts:
  Item embeddings use plaintext sparse ``{rows, values}`` uploads. In the
  default configuration (positives only), the server sees each client's
  exact positive item set. The paper hides participation behind a full
  table + mask under secure aggregation; that blurring property is absent
  here. Default follows Algorithm 2 (P = O). §3.5.2's blurring argument
  relies on negative-item updates being visible to the server;
  ``item_upload_include_negatives`` on the client includes negatives in
  the upload.

Deviations from the paper:
  Labels and loss follow §4.2 (implicit binary targets, BCE, four
  negatives). ``predictive_factor=32``, ``lr=5e-4`` and ``local_epochs=5``
  on the client are the public-code defaults, not the paper's D=12,
  Adam 0.001 and two local epochs.
"""

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering


def default_param_ios() -> dict[str, ParamIO]:
    return {
        "item_embeddings": ParamIO(
            residency="server", exchange="params", apply_to="server", layout="rows",
        ),
        "shared_layers": ParamIO(
            residency="server", exchange="params", apply_to="server",
        ),
        "output_weights": ParamIO(
            residency="server", exchange="params", apply_to="server",
        ),
    }


class FedNCF(BaseFederatedServer):
    """Federated NCF. Server holds item tables, MLP and output weights.

    User embedding tables and their optimiser state live in ClientState.
    Item embedding uploads are row-layout by default; override with
    ``param_io_overrides={"item_embeddings": {"layout": "dense"}}`` for the
    diluted dense-average ablation.
    """

    def __init__(self, predictive_factor: int = 32, **kwargs):
        self.predictive_factor = predictive_factor
        super().__init__(param_ios=default_param_ios(), **kwargs)

    def _initialize_server_parameters(self):
        """Create initial server weights using a temporary skeleton."""
        model = NeuralCollaborativeFiltering(
            num_items=self.interaction_dataset.num_items,
            predictive_factor=self.predictive_factor,
        )
        return model.get_server_state_groups()
