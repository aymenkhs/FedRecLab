"""FedNCF client algorithm -- local training and scoring for Federated NCF.

Source : Perifanis, Vasileios, and Pavlos S. Efraimidis. “Federated Neural Collaborative Filtering.”
Knowledge-Based Systems 242 (April 2022): 108441. https://doi.org/10.1016/j.knosys.2022.108441.

Implements BaseClientAlgorithm with three responsibilities:
  1. initialize_local_parameters: create per-client user embeddings, model, and optimizer (both persisted across rounds in ClientState)
  2. local_train: load latest server/user weights into the persistent model, train with negative sampling, return update packages
  3. compute_scores: load weights into persistent model and score candidates
"""

import time
from typing import List, Dict, Tuple, Union, Any

import numpy as np
import torch

from src.fedreclab.federated.base_client import (
    BaseClientAlgorithm,
    ClientStateBase,
    ClientTrainingRequest,
    ClientTrainingResponse,
)
from src.fedreclab.federated.utils import ParamIO, UpdatePackage
from src.fedreclab.methods.fedncf.fedncf_model import NeuralCollaborativeFiltering


class FedNCFClientAlgorithm(BaseClientAlgorithm):

    def __init__(
        self,
        predictive_factor: int = 32,
        local_epochs: int = 5,
        lr: float = 5e-4,
        batch_size: int = 128,
        neg_samples: int = 3,
    ):
        super().__init__()
        self.predictive_factor = predictive_factor
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.num_items = None

    def initialize_local_parameters(
        self,
        client_id: int,
        total_num_users: int,
        num_users: int,
        num_items: int,
        client_dataset_view: Any,
        users_in_client: List[int],
    ) -> Tuple[Dict[str, Any], torch.optim.Optimizer]:
        if self.num_items is None:
            self.num_items = num_items

        model = NeuralCollaborativeFiltering(num_users, num_items, self.predictive_factor)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        user_id_to_local = {uid: idx for idx, uid in enumerate(users_in_client)}

        local_params = {
            "user_embeddings": model.get_user_state(),
            "user_id_to_local": user_id_to_local,
            "model": model,
        }
        return local_params, optimizer

    def local_train(self, training_request: ClientTrainingRequest) -> ClientTrainingResponse:
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        client_id = training_request.client_id
        client_state = training_request.client_state
        training_data = training_request.training_data
        user_id_to_local = client_state.local_parameters["user_id_to_local"]

        # ---- 1. Reuse persistent model; load latest weights in-place ----
        model = client_state.local_parameters["model"]
        model.load_server_weights(training_request.global_payload)
        model.load_user_weights(client_state.local_parameters["user_embeddings"])
        model.join_output_weights()
        model = model.to(device)

        optimizer = client_state.optimizer_state
        if not isinstance(optimizer, torch.optim.Optimizer):
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            if client_state.optimizer_state is not None:
                try:
                    optimizer.load_state_dict(client_state.optimizer_state)
                except ValueError:
                    pass

        user_col, item_col = "user_id", "item_id"
        rating_col = "rating" if "rating" in training_data.columns else None

        pos_global_uids = training_data[user_col].values
        pos_item_ids = training_data[item_col].values
        pos_ratings = training_data[rating_col].values if rating_col else np.ones(len(training_data), dtype=np.float32)

        pos_local_uids = np.array([user_id_to_local[u] for u in pos_global_uids])

        user_pos_items: Dict[int, set] = {}
        for local_uid, iid in zip(pos_local_uids, pos_item_ids):
            user_pos_items.setdefault(local_uid, set()).add(iid)

        model.train()
        all_losses: List[float] = []
        num_training_examples = 0

        for _ in range(self.local_epochs):
            samples_x, samples_y = self._build_epoch_samples(
                pos_local_uids, pos_item_ids, pos_ratings, user_pos_items,
            )
            num_training_examples += len(samples_x)

            perm = np.random.permutation(len(samples_x))
            samples_x = samples_x[perm]
            samples_y = samples_y[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, len(samples_x), self.batch_size):
                bx = torch.tensor(samples_x[start:start + self.batch_size], dtype=torch.long, device=device)
                by = torch.tensor(samples_y[start:start + self.batch_size], dtype=torch.float32, device=device)

                preds = model(bx)
                loss = torch.nn.functional.mse_loss(preds, by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                all_losses.append(epoch_loss / num_batches)

        model = model.cpu()
        server_groups = model.get_server_state_groups()

        updates_packages = {}
        for group_name, group_value in server_groups.items():
            updates_packages[group_name] = UpdatePackage(
                param_name=group_name,
                param_io=ParamIO(residency="server", exchange="params", apply_to="server"),
                param_value=group_value,
                param_metadata={},
            )

        updated_state = ClientStateBase(
            client_id=client_id,
            local_parameters={
                "user_embeddings": model.get_user_state(),
                "user_id_to_local": user_id_to_local,
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

    def _build_epoch_samples(
        self,
        pos_local_uids: np.ndarray,
        pos_item_ids: np.ndarray,
        pos_ratings: np.ndarray,
        user_pos_items: Dict[int, set],
    ):
        """Build one epoch of training samples (positives + negatives).

        For each positive interaction, self.neg_samples random items that the
        user has NOT interacted with are sampled as negatives (label=0).
        """
        n_pos = len(pos_local_uids)
        n_total = n_pos * (1 + self.neg_samples)

        samples_x = np.empty((n_total, 2), dtype=np.int64)
        samples_y = np.empty(n_total, dtype=np.float32)

        idx = 0
        for i in range(n_pos):
            local_uid = pos_local_uids[i]
            pos_item = pos_item_ids[i]
            rating = pos_ratings[i]

            samples_x[idx] = [local_uid, pos_item]
            samples_y[idx] = rating
            idx += 1

            excl = user_pos_items.get(local_uid, set())
            for _ in range(self.neg_samples):
                neg_item = np.random.randint(0, self.num_items)
                while neg_item in excl:
                    neg_item = np.random.randint(0, self.num_items)
                samples_x[idx] = [local_uid, neg_item]
                samples_y[idx] = 0.0
                idx += 1

        return samples_x[:idx], samples_y[:idx]

    def compute_scores(
        self,
        client_id: int,
        candidate_items_per_user: Dict[int, np.ndarray],
        global_parameters: Dict[str, Any],
        client_state: ClientStateBase,
        extra_payload: Dict[str, Any],
    ) -> Tuple[Dict[int, np.ndarray], float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        user_id_to_local = client_state.local_parameters["user_id_to_local"]

        model = client_state.local_parameters["model"]
        model.load_server_weights(global_parameters)
        model.load_user_weights(client_state.local_parameters["user_embeddings"])
        model.join_output_weights()
        model = model.to(device)
        model.eval()

        scores: Dict[int, np.ndarray] = {}

        with torch.no_grad():
            for user_id, candidate_items in candidate_items_per_user.items():
                local_uid = user_id_to_local[user_id]
                n_items = len(candidate_items)

                user_ids = torch.full((n_items,), local_uid, dtype=torch.long, device=device)
                item_ids = torch.tensor(candidate_items, dtype=torch.long, device=device)
                x = torch.stack([user_ids, item_ids], dim=1)

                scores[user_id] = model(x).cpu().numpy()

        return scores, 0.0
