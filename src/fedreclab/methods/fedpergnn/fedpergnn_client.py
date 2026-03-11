import time
from typing import Dict, Any, Tuple, List

import numpy as np
import torch

from src.fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from src.fedreclab.federated.utils import ParamIO, UpdatePackage
from src.fedreclab.methods.fedpergnn.fedpergnn_model import FedPerGNNModel


class FedPerGNNClientAlgorithm(BaseClientAlgorithm):

    def __init__(
        self,
        hidden_dim: int = 64,
        history_len: int = 20,
        neighbor_len: int = 10,
        local_epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.neighbor_len = neighbor_len
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size

        self._num_items: int | None = None

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_dataset_view,
        users_in_client: List[int],
    ) -> Tuple[Dict[str, Any], Any]:
        if self._num_items is None:
            self._num_items = num_items

        # TODO: Allow for custom MLP and GAT layers with a custom activation function, and parameters.
        # TODO: Currently we are sending all paramters as local paramters but later on we will only store 
        # completely local parameters in the client state.
        model = FedPerGNNModel(
            num_items=num_items,
            hidden_dim=self.hidden_dim,
            history_len=self.history_len,
            neighbor_len=self.neighbor_len,
        )
        # TODO: Allow for custom optimizer with a custom learning rate, and parameters.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        local_params = {
            "model": model,
        }
        return local_params, optimizer

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data

        model: FedPerGNNModel = client_state.local_parameters["model"]
        model.load_server_state(training_request.global_payload["model_state"])
        model.to(device)
        model.train()

        optimizer = client_state.optimizer_state
        if not isinstance(optimizer, torch.optim.Optimizer):
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        user_col, item_col = "user_id", "item_id"
        rating_col = "rating" if "rating" in training_data.columns else None

        # All interactions belong to the same user for CrossDeviceSplitting.
        user_ids_np = training_data[user_col].values.astype(np.int64)
        item_ids_np = training_data[item_col].values.astype(np.int64)
        if rating_col is not None:
            ratings_np = training_data[rating_col].values.astype(np.float32)
        else:
            ratings_np = np.ones(len(training_data), dtype=np.float32)

        # History and neighbor embeddings are expected from extra_payload.
        expanded_payload_all = training_request.extra_payload or {}
        expanded_for_client = expanded_payload_all.get("expanded_user_data", {}).get(client_id, None)

        if expanded_for_client is not None:
            history_items_np = expanded_for_client.get("history_item_ids")
            neighbor_emb_np = expanded_for_client.get("neighbor_embeddings")
        else:
            # Fallback: simple history with no neighbors.
            # History is the most recent `history_len` interactions for this user,
            # determined by timestamp if available, otherwise by original order.
            if "timestamp" in training_data.columns:
                base = training_data.sort_values(by="timestamp")
            else:
                base = training_data

            hist_items = base[item_col].values.astype(np.int64)[-self.history_len :]
            if hist_items.shape[0] < self.history_len:
                pad_len = self.history_len - hist_items.shape[0]
                hist_items = np.pad(hist_items, (0, pad_len), mode="constant", constant_values=0)
            history_items_np = hist_items
            neighbor_emb_np = None

        # History and neighbor embeddings are shared across all interactions
        # for this user, so we broadcast them per-batch when constructing
        # tensors.
        num_examples = len(item_ids_np)
        all_losses: List[float] = []
        num_training_examples = 0

        for _ in range(self.local_epochs):
            indices = np.arange(num_examples)
            np.random.shuffle(indices)

            user_ids_np_shuffled = user_ids_np[indices]
            item_ids_np_shuffled = item_ids_np[indices]
            ratings_np_shuffled = ratings_np[indices]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_examples, self.batch_size):
                end = min(start + self.batch_size, num_examples)
                if start >= end:
                    break

                b_user_ids = torch.as_tensor(
                    user_ids_np_shuffled[start:end], dtype=torch.long, device=device
                )
                b_item_ids = torch.as_tensor(
                    item_ids_np_shuffled[start:end], dtype=torch.long, device=device
                )
                b_ratings = torch.as_tensor(
                    ratings_np_shuffled[start:end], dtype=torch.float32, device=device
                )

                # Broadcast history / neighbor tensors to the batch size.
                b_hist_items = torch.as_tensor(
                    np.repeat(history_items_np[None, :], repeats=b_item_ids.size(0), axis=0),
                    dtype=torch.long,
                    device=device,
                )

                if neighbor_emb_np is not None:
                    b_neighbor_emb = torch.as_tensor(
                        np.repeat(neighbor_emb_np[None, :, :, :], repeats=b_item_ids.size(0), axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    b_neighbor_emb = None

                optimizer.zero_grad()
                preds = model(
                    user_ids=b_user_ids,
                    item_ids=b_item_ids,
                    history_item_ids=b_hist_items,
                    neighbor_embeddings=b_neighbor_emb,
                )
                loss = torch.mean((preds - b_ratings) ** 2)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                num_training_examples += b_item_ids.size(0)

            if num_batches > 0:
                all_losses.append(epoch_loss / num_batches)

        model = model.cpu()

        server_state = model.get_server_state()

        numpy_state: Dict[str, Any] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
            for k, v in server_state.items()
        }

        updates_packages: Dict[str, UpdatePackage] = {
            "model_state": UpdatePackage(
                param_name="model_state",
                param_io=ParamIO(residency="server", exchange="params", apply_to="server"),
                param_value=numpy_state,
                param_metadata={},
            )
        }

        updated_state = ClientStateBase(
            client_id=client_id,
            local_parameters={
                "model": model,
            },
            optimizer_state=optimizer,
        )

        return ClientTrainingResponse(
            client_id=client_id,
            updated_client_state=updated_state,
            updates_packages=updates_packages,
            training_losses=all_losses,
            training_duration=time.time() - t0,
            num_training_examples=num_training_examples,
        )

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: Dict[int, np.ndarray],
        global_parameters: Dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: Dict[str, Any],
    ) -> Tuple[Dict[int, np.ndarray], float]:
        """
        Compute scores for ranking evaluation.

        For now we ignore neighbor information during scoring for simplicity
        and only use a fixed history per user. This can be extended later to
        align with the training-time expansion logic.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model: FedPerGNNModel = client_state.local_parameters["model"]
        model.load_server_state(global_parameters["model_state"])
        model.to(device)
        model.eval()

        # Build a simple history from the client's training data if available
        # via extra_payload, otherwise use zeros.
        expanded_payload_all = extra_payload or {}
        expanded_for_client = expanded_payload_all.get("expanded_user_data", {}).get(client_id, None)
        if expanded_for_client is not None:
            history_items_np = expanded_for_client.get("history_item_ids")
            neighbor_emb_np = expanded_for_client.get("neighbor_embeddings")
        else:
            history_items_np = np.zeros((self.history_len,), dtype=np.int64)
            neighbor_emb_np = None

        scores: Dict[int, np.ndarray] = {}

        with torch.no_grad():
            for user_id, cand_items in candidate_items_per_user.items():
                b = len(cand_items)
                user_ids = torch.full((b,), int(user_id), dtype=torch.long, device=device)
                item_ids = torch.as_tensor(cand_items, dtype=torch.long, device=device)
                hist_items = torch.as_tensor(
                    np.repeat(history_items_np[None, :], repeats=b, axis=0),
                    dtype=torch.long,
                    device=device,
                )

                if neighbor_emb_np is not None:
                    neigh = torch.as_tensor(
                        np.repeat(neighbor_emb_np[None, :, :, :], repeats=b, axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    neigh = None

                preds = model(
                    user_ids=user_ids,
                    item_ids=item_ids,
                    history_item_ids=hist_items,
                    neighbor_embeddings=neigh,
                )
                scores[user_id] = preds.cpu().numpy()

        model = model.cpu()
        return scores, 0.0

