# Federated recommendation methods

Nine federated recommenders live under `[src/fedreclab/methods/](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/)`. The package `__init__.py` is empty, so import the concrete modules (for example `fedreclab.methods.fedncf.fedncf`).

Every server subclasses `[BaseFederatedServer](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/base_server.py)` and takes the shared arguments below. Method-specific constructor fields are listed per method. Client algorithms, selectors, aggregators and evaluators are single-use: build new ones for every server. The dataset may be shared.

**Shared server arguments**


| Argument                     | Role                                                                    |
| ---------------------------- | ----------------------------------------------------------------------- |
| `interaction_dataset`        | Split and partitioned `InteractionDataset`                              |
| `client_algorithm`           | Matching `BaseClientAlgorithm` subclass                                 |
| `num_rounds`                 | Federated rounds                                                        |
| `client_selection_strategy`  | Selector from `[client_selection.md](client_selection.md)`              |
| `aggregation_strategies`     | One aggregator, or a `{group: aggregator}` dict                         |
| `evaluator`                  | Optional evaluator (required for early stopping on a validation metric) |
| `seed`                       | Optional; also written onto the selector when its seed is still `None`  |
| `device`                     | Optional; defaults to CUDA when available                               |
| `param_io_overrides`         | Optional dict of field overrides for declared `ParamIO` groups          |
| `validation_rounds_interval` | Default `1`                                                             |
| `early_stopping`             | Default `False`                                                         |
| `early_stopping_parameters`  | e.g. `{"metric": "ndcg@10", "direction": "max", "patience": 10}`        |




## Overview


| Method    | Year | Code                                                                                                                                                                                                                                                                                                                                                                                                                     | Paper                                                                                                                                                                | DOI / link                                                                         |
| --------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| FedMF     | 2021 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedmf/fedmf.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedmf/fedmf_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedmf/fedmf_model.py)                                                                                                       | Chai, Wang, Chen, Yang. Secure Federated Matrix Factorization. *IEEE Intelligent Systems* 36(5):11–20                                                                | [10.1109/MIS.2020.3014880](https://doi.org/10.1109/MIS.2020.3014880)               |
| FedNCF    | 2022 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedncf/fedncf.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedncf/fedncf_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedncf/fedncf_model.py)                                                                                                 | Perifanis, Efraimidis. Federated Neural Collaborative Filtering. *Knowledge-Based Systems* 242:108441                                                                | [10.1016/j.knosys.2022.108441](https://doi.org/10.1016/j.knosys.2022.108441)       |
| FedFast   | 2020 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedfast/fedfast.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedfast/fedfast_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedfast/fedfast_model.py)                                                                                           | Muhammad, Wang, O'Reilly-Morgan, Tragos, Smyth, Hurley, Geraci, Lawlor. FedFast: Going Beyond Average for Faster Training of Federated Recommender Systems. KDD 2020 | [10.1145/3394486.3403176](https://doi.org/10.1145/3394486.3403176)                 |
| FedVAE    | 2021 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedvae/fedvae.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedvae/fedvae_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedvae/fedvae_model.py)                                                                                                 | Polato. Federated Variational Autoencoder for Collaborative Filtering. IJCNN 2021                                                                                    | [10.1109/IJCNN52387.2021.9533358](https://doi.org/10.1109/IJCNN52387.2021.9533358) |
| EFVAE     | 2024 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/efvae/efvae.py) · [FCIS](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/efvae/efvae_fcis.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/efvae/efvae_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/efvae/efvae_model.py) | Zhang, Rong, Ding, Li, Yuan. EFVAE: Efficient Federated Variational Autoencoder for Collaborative Filtering. CIKM 2024, pp. 3176–3185                                | [10.1145/3627673.3679818](https://doi.org/10.1145/3627673.3679818)                 |
| GPFedRec  | 2024 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/gpfedrec/gpfedrec.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/gpfedrec/gpfedrec_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/gpfedrec/gpfedrec_model.py)                                                                                     | Zhang, Long, Zhou, Zhang, Yan, Yang. GPFedRec: Graph-Guided Personalization for Federated Recommendation. KDD 2024                                                   | [10.1145/3637528.3671702](https://doi.org/10.1145/3637528.3671702)                 |
| FedPerGNN | 2022 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedpergnn/fedpergnn.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedpergnn/fedpergnn_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedpergnn/fedpergnn_model.py)                                                                               | Wu, Wu, Lyu, Qi, Huang, Xie. A federated graph neural network framework for privacy-preserving personalization. *Nature Communications* 13:3091                      | [10.1038/s41467-022-30714-9](https://doi.org/10.1038/s41467-022-30714-9)           |
| MetaMF    | 2020 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/metamf/metamf.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/metamf/metamf_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/metamf/metamf_model.py)                                                                                                 | Lin, Ren, Chen, Ren, Yu, Ma, de Rijke, Cheng. Meta Matrix Factorization for Federated Rating Predictions. SIGIR 2020                                                 | [10.1145/3397271.3401081](https://doi.org/10.1145/3397271.3401081)                 |
| SemiDFEGL | 2023 | [server](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/semidfegl/semidfegl.py) · [client](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/semidfegl/semidfegl_client.py) · [model](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/semidfegl/semidfegl_model.py)                                                                               | Qu, Tang, Zheng, Nguyen, Huang, Shi, Yin. Semi-decentralized Federated Ego Graph Learning for Recommendation. WWW 2023, pp. 339–348                                  | [10.1145/3543507.3583337](https://doi.org/10.1145/3543507.3583337)                 |


---



## FedMF

**Task.** Explicit rating prediction. Score is `⟨u_i, v_j⟩`. Loss is regularised least squares.

**Exchange.** ParamIO group `item_embeddings`: residency `server`, exchange `gradient`, apply to `server`, layout `rows` (PartText). FullText is `param_io_overrides={"item_embeddings": {"layout": "dense"}}`. Faithful aggregation is `[Sum](aggregation.md)` on the item gradients.

**Server defaults.** `latent_dim=100`, `init_mode="constant"`, `init_value=0.01`, `init_std=0.01`.

**Client defaults** (`FedMFClientAlgorithm`). `latent_dim=100`, `lr=1e-3`, `reg_u=1e-4`, `reg_v=1e-4`, `local_epochs=1`, `local_update="sgd_sequential"`, `u_regularisation="per_rating"`, `v_regularisation="per_rating"`, `lr_placement="client"`, `init_mode="constant"`, `init_value=0.01`, `init_std=0.01`, `init_seed=None`, `loss_scale="none"`.

**Limitations.** Homomorphic encryption is omitted; gradients travel in plaintext, so the paper's Section 4 inversion attack applies. Plaintext FullText still reveals rated-item support because unrated rows are exact zeros. There is no server-side once-per-round `2μV` term. RegularMF's per-user sequential apply of `V` is not implemented. The library evaluator is top-K ranking; the paper's RMSE is not a library metric.

**Instantiation**

```python
from fedreclab.methods.fedmf.fedmf import FedMF
from fedreclab.methods.fedmf.fedmf_client import FedMFClientAlgorithm
from fedreclab.federated.aggregation import Sum
from fedreclab.federated.client_selection import RandomClientSelection

client = FedMFClientAlgorithm(latent_dim=100)
server = FedMF(
    latent_dim=100,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=50,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.1, "frac"),
    aggregation_strategies=Sum(num_clients=dataset.num_clients),
    evaluator=evaluator,
)
```

No training notebook yet. The method is exercised in `tests/fedmf/`.

---



## FedNCF

**Task.** Implicit-feedback ranking (paper default). Binary labels, BCE with four negatives per positive. Rating-valued MSE is an opt-in deviation (`label_mode="rating"`, `loss="mse"`). NeuMF backbone follows He et al., WWW 2017.

**Exchange.** `item_embeddings` (rows, params; faithful `[SimpleAvg](aggregation.md)`); `shared_layers` and `output_weights` (dense, params; faithful `[FedAvg](aggregation.md)`).

**Server defaults.** `predictive_factor=32`.

**Client defaults** (`FedNCFClientAlgorithm`). `predictive_factor=32`, `local_epochs=5`, `lr=5e-4`, `batch_size=128`, `neg_samples=4`, `label_mode="binary"`, `loss="bce"`, `optimizer_state_scope="client_resident"`, `item_upload_include_negatives=False`.

**Limitations.** Plaintext sparse uploads show each client's positive item set (no secure-aggregation mask). Learning rate and local epochs follow the public code (`5e-4`, five epochs), not Adam `0.001` and two epochs. The NeuMF tower cannot express the paper MLP `24→48→24→12→6` until widths are configurable (NCF-03).

**Instantiation**

```python
from fedreclab.methods.fedncf.fedncf import FedNCF
from fedreclab.methods.fedncf.fedncf_client import FedNCFClientAlgorithm
from fedreclab.federated.aggregation import FedAvg, SimpleAvg
from fedreclab.federated.client_selection import RandomClientSelection

client = FedNCFClientAlgorithm(predictive_factor=32, local_epochs=5, lr=5e-4,
                               batch_size=128, neg_samples=4)
server = FedNCF(
    predictive_factor=32,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=100,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.02, "frac"),
    aggregation_strategies={
        "item_embeddings": SimpleAvg(num_clients=dataset.num_clients),
        "shared_layers": FedAvg(num_clients=dataset.num_clients),
        "output_weights": FedAvg(num_clients=dataset.num_clients),
    },
    evaluator=evaluator,
)
```

**Examples.** `[examples/quickstart.ipynb](https://github.com/aymenkhs/FedRecLab/blob/master/examples/quickstart.ipynb)`, `[examples/custom_data.ipynb](https://github.com/aymenkhs/FedRecLab/blob/master/examples/custom_data.ipynb)`, `[examples/custom_aggregator.ipynb](https://github.com/aymenkhs/FedRecLab/blob/master/examples/custom_aggregator.ipynb)`, `[examples/extend_fedncf.ipynb](https://github.com/aymenkhs/FedRecLab/blob/master/examples/extend_fedncf.ipynb)`.

---



## FedFast

**Task.** Implicit ranking with a GMF model (BCE). ActvSAMP client sampling and ActvAGG item aggregation. See also `[ActvSampClientSelection](client_selection.md)` and `[ChangeMagnitudeAvg](aggregation.md)`.

**Exchange.** `item_embeddings` (rows, params); `output_weights` (dense, params). User embeddings stay in `ClientState` and are not ParamIO groups. With `actv_agg=True` (default), the server installs `ChangeMagnitudeAvg` on items and `FedAvg` on the head, and runs subordinate user updates on round end.

**Server defaults.** `embedding_dim=10`, `num_clusters=20`, `actv_agg=True`, `change_eps=0.0`.

**Client defaults** (`FedFastClientAlgorithm`). `embedding_dim=10`, `local_epochs=1`, `lr=1e-3`, `batch_size=128`, `neg_samples=4`, `label_mode="binary"`, `loss="bce"`, `optimizer_state_scope="client_resident"`, `item_upload_include_negatives=False`.

**Limitations.** The server reads every client's user embeddings for k-means and writes non-selected (subordinate) user embeddings after each round. Initial clusters come from centralised train-profile statistics. User embeddings are client-resident (Option A). The paper's round-robin one-per-cluster sampling is not implemented. NeuMF / NCF is not implemented. Pair with `ActvSampClientSelection` and choose `m` divisible by `num_clusters`. Private helper: `[_actvagg.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/fedfast/_actvagg.py)`.

**Instantiation**

```python
from fedreclab.methods.fedfast.fedfast import FedFast
from fedreclab.methods.fedfast.fedfast_client import FedFastClientAlgorithm
from fedreclab.federated.client_selection import ActvSampClientSelection

client = FedFastClientAlgorithm(embedding_dim=10)
selector = ActvSampClientSelection(dataset.num_clients, 40, "num", seed=42)
server = FedFast(
    embedding_dim=10,
    num_clusters=20,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=50,
    seed=42,
    client_selection_strategy=selector,
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/fedfast/`.

---



## FedVAE

**Task.** Mult-VAE ELBO collaborative filtering (Liang et al., WWW 2018). Selected clients download the whole network, compute one dense gradient on their training histories, and upload it. The server applies Adam (optional FedVAEdec schedule) in `_apply_aggregated_result_custom`.

**Exchange.** Single group `vae`: server, gradient, `apply_to="custom"`, dense. Faithful aggregation is `[SimpleAvg](aggregation.md)`. `FedAvg` weights by history length and is a deviation. Multi-Krum is the paper's optional SecureFilter.

**Server defaults.** `hidden_dims=(600,)`, `latent_dim=200`, `dropout=0.5`, `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`, `lr_multiplier=0.0`, `lr_decay=0.9`, `decay_every_rounds=None`, `extrapolate="all"`.

**Client defaults** (`FedVAEClientAlgorithm`). `hidden_dims=(600,)`, `latent_dim=200`, `dropout=0.5`, `beta=0.2`.

Paper binarisation for MovieLens (ratings ≥ 4) is `rating_exclusion_threshold=4` at load time.

**Limitations.** Dense gradient memory grows with `clients_per_round * |I|` (about 18 MB per client at 3.5k items and hidden width 600). Plaintext encoder gradients reveal about half of each user's positives. Decay indexing and `extrapolate` differ from a literal reading of the paper. Splitting the VAE into several ParamIO groups breaks Krum / Bulyan because they would keep or drop clients per group.

**Instantiation**

```python
from fedreclab.methods.fedvae.fedvae import FedVAE
from fedreclab.methods.fedvae.fedvae_client import FedVAEClientAlgorithm
from fedreclab.federated.aggregation import SimpleAvg
from fedreclab.federated.client_selection import RandomClientSelection

client = FedVAEClientAlgorithm(hidden_dims=(600,), latent_dim=200, beta=0.2)
server = FedVAE(
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=100,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.1, "frac"),
    aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/fedvae/`.

---



## EFVAE

**Task.** Efficient Mult-VAE with Federated Collaborative Importance Sampling (FCIS). Clients upload sparse encoder / decoder row gradients and a dense middle gradient. `FIMSUniSampler` in `[efvae_sampler.py](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/methods/efvae/efvae_sampler.py)` is item proposal sampling, not client selection.

**Exchange.** Groups `enc_embeddings`, `dec_embeddings`, `dec_bias` (row gradients, `apply_to="custom"`) and `middle` (dense gradient, `apply_to="custom"`). Faithful aggregation is `[SimpleAvg](aggregation.md)` on every group.

`EFVAE` **defaults.** `hidden_dims=(120,)`, `latent_dim=40`, `dropout=0.5`, `anneal_cap=0.2`, `total_anneal_steps=200000`.

`EFVAEFCISServer` **defaults** (forwarded). `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`, `num_cluster=16`, `num_sample=800`.

**Client defaults** (`EFVAEClientAlgorithm`). `hidden_dims=(120,)`, `latent_dim=40`, `dropout=0.5`.

**Limitations.** Full parameter tables are distributed every round; secret sharing / MPC and communication-cost meters are omitted. Paper Alg. 1–2 multi-step sparse download is not enforced. Fake-item count is clamped when the catalogue is too small. Id `0` is never used as padding. MultDAE and RecVAE are named as future subclasses of `EFVAEFCISServer` and are not implemented.

**Instantiation**

```python
from fedreclab.methods.efvae.efvae import EFVAE
from fedreclab.methods.efvae.efvae_client import EFVAEClientAlgorithm
from fedreclab.federated.aggregation import SimpleAvg
from fedreclab.federated.client_selection import RandomClientSelection

client = EFVAEClientAlgorithm(hidden_dims=(120,), latent_dim=40)
server = EFVAE(
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=100,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.1, "frac"),
    aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/efvae/`.

---



## GPFedRec

**Task.** Implicit ranking. Clients train BCE over positives and sampled negatives, plus `reg * MSE(item_table, r_i)`, and upload a full item table. The server builds a cosine similarity graph, computes personalised tables `R`, and aggregates `q_global` with the configured aggregator.

**Exchange.** Only `item_embeddings` is ParamIO (dense, params). Personalised tables `r_i` are server state under the payload key `personalized_item_embeddings`; they are not aggregated and are not copied into the early-stopping snapshot. Faithful pair: `[FullClientSelection](client_selection.md)` + `[SimpleAvg](aggregation.md)`. The graph step needs every individual upload (not secure-aggregation compatible).

**Server defaults.** `latent_dim=32`, `item_init_std=1.0`, `neighbor_rule="similarity"`, `neighborhood_threshold=1.0`, `mp_layers=1`. A `seed` is required for the initial table.

**Client defaults** (`GPFedRecClientAlgorithm`). `latent_dim=32`, `hidden_layers=(64, 32, 16, 8)`, `local_epochs=1`, `batch_size=256`, `neg_samples=4`, `lr=0.1`, `lr_eta=80.0`, `participation_ratio=1.0`, `user_lr=None`, `item_lr=None`, `reg=1.0`, `ldp_laplace_scale=0.0`, `init_seed=None`.

**Limitations.** Simulation shortcuts: none declared. `neighbor_rule="similarity"` follows the paper; `"reference_distance"` follows the reference code, and paper `gamma` values do not transfer across rules. Light-GPFedRec (refreshing the graph less often), top-k neighbourhoods and non-cosine similarity are not implemented. Memory at `latent_dim=32`: about 0.2 GB / 2.9 GB / 2.6 GB per dense table copy on ML-100K / ML-1M / Lastfm-2K.

**Instantiation**

```python
from fedreclab.methods.gpfedrec.gpfedrec import GPFedRec
from fedreclab.methods.gpfedrec.gpfedrec_client import GPFedRecClientAlgorithm
from fedreclab.federated.aggregation import SimpleAvg
from fedreclab.federated.client_selection import FullClientSelection

client = GPFedRecClientAlgorithm(latent_dim=32)
server = GPFedRec(
    latent_dim=32,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=50,
    seed=42,
    client_selection_strategy=FullClientSelection(dataset.num_clients),
    aggregation_strategies=SimpleAvg(num_clients=dataset.num_clients),
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/gpfedrec/`.

---



## FedPerGNN

**Task.** Federated GNN personalisation. Default `task="rating"` trains MSE on observed ratings (paper). `task="ranking"` is the library ranking protocol (binary labels, negatives, BCE). Score is the SI inner product of the user hidden state and the target item embedding.

**Exchange.** `model_state` (dense, params); `item_embeddings` and `user_embeddings` (rows, params). Architecture widths come from the client only. The anchor user's embedding is always trainable; `neighbor_embeddings_trainable` covers neighbour rows only.

**Server defaults** (keyword-only). `warmup_rounds=1`, `expanded_users=None`, `history_item_ids=None`, `expansion_seed=None`.

**Client defaults** (`FedPerGNNClientAlgorithm`). `hidden_dim=64`, `history_len=20`, `neighbor_len=10`, `local_epochs=1`, `lr=1e-3`, `batch_size=128`, `neighbor_embeddings_trainable=False`, `optimizer_state_scope="client_resident"`, `task="rating"`, `neg_samples=None` (resolves to 0 for rating and 4 for ranking), `num_pseudo_items=0`, `ldp_clip=None`, `ldp_scale=0.0`.

**Limitations.** The server holds a canonical user-embedding table used as an expansion mirror (simulation shortcut). Uploads are parameters; a TODO on `default_param_ios` notes that the paper averages gradients and applies `Theta ← Theta − α g`. The Keras concat-plus-`Dense(1, sigmoid)` head is not used. Privacy defaults are off (paper uses M=1000, L1 clip 0.1, Laplace scale 0.2). Pseudo-items never enter the loss; they are not ranking negatives. Empty history slots are `-1`; catalogue item `i` is embedding row `i + 1`.

**Instantiation**

```python
from fedreclab.methods.fedpergnn.fedpergnn import FedPerGNN
from fedreclab.methods.fedpergnn.fedpergnn_client import FedPerGNNClientAlgorithm
from fedreclab.federated.aggregation import FedAvg, SimpleAvg
from fedreclab.federated.client_selection import RandomClientSelection

client = FedPerGNNClientAlgorithm(hidden_dim=64, history_len=20, neighbor_len=10)
server = FedPerGNN(
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=50,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.1, "frac"),
    aggregation_strategies={
        "model_state": FedAvg(num_clients=dataset.num_clients),
        "item_embeddings": SimpleAvg(num_clients=dataset.num_clients),
        "user_embeddings": SimpleAvg(num_clients=dataset.num_clients),
    },
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/fedpergnn/`.

---



## MetaMF

**Task.** Explicit rating prediction. A server-resident meta network `Theta` generates per-user item embeddings and a small rating MLP (`Phi_u`). Clients upload the gradient of a local MSE with respect to `Phi_u`; the server translates it into a gradient on `Theta` and applies Adam.

**Exchange.** Group `meta_network`: server, gradient, `apply_to="custom"`, dense. Faithful aggregation is `[FedAvg](aggregation.md)`. `Sum` matches the paper's accumulate but rescales the data gradient by the round size.

**Server defaults.** `user_emb_dim=32`, `item_emb_dim=32`, `item_mem_num=8`, `mem_size=128`, `hidden_size=512`, `rp_hidden_dim=None`, `item_payload="factors"`, `lr=1e-4`, `weight_decay=1e-3`, `max_grad_norm=5.0`, `betas=(0.9, 0.999)`, `eps=1e-8`.

**Client defaults** (`MetaMFClientAlgorithm`). Same architecture fields as the server, plus `local_batch_size=None`, `local_batch_replacement=False`.

**Limitations.** The server holds `e_u` and `Theta` (not secure-aggregation compatible). `Phi` is regenerated at score time. Robust aggregators on sparse factor rows are a known limitation. The library evaluator is ranking, not RMSE.

**Instantiation**

```python
from fedreclab.methods.metamf.metamf import MetaMF
from fedreclab.methods.metamf.metamf_client import MetaMFClientAlgorithm
from fedreclab.federated.aggregation import FedAvg
from fedreclab.federated.client_selection import RandomClientSelection

client = MetaMFClientAlgorithm(user_emb_dim=32, item_emb_dim=32)
server = MetaMF(
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=50,
    seed=42,
    client_selection_strategy=RandomClientSelection(dataset.num_clients, 0.1, "frac"),
    aggregation_strategies=FedAvg(num_clients=dataset.num_clients),
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/metamf/`.

---



## SemiDFEGL

**Task.** Semi-decentralized ego-graph learning with BPR. The implemented encoder is LightGCN (He et al., SIGIR 2020); GCN, GAT and GraphSAGE from the paper's §4.6 are not implemented. Recommend `[FullClientSelection](client_selection.md)`.

**Exchange.** `item_embeddings` and `ego_embeddings`, both row-layout params.

**Server defaults.** `embedding_dim=64`, `num_groups=100`, `n_fake=1`, `num_layers=4`, `fuzziness=2.0`, `cmeans_max_iter=100`, `cmeans_tol=1e-5`, `bpr_negatives=1`. A `seed` is required for Xavier init, BPR negatives and c-means.

**Client defaults** (`SemiDFEGLClientAlgorithm`). `embedding_dim=64`, `num_layers=4`, `local_epochs=1`, `lr=0.001`, `adam_weight_decay=1e-4`, `bpr_reg=0.0`, `fake_items_as_negatives=True`, `num_pseudo_items=0`, `clip_threshold=None`, `laplace_scale=0.0`.

**Limitations.** The server copies each cluster member's stored ego vector into the payload in place of a device-to-device link. Neighbour messages are one round old. There is no in-round multi-hop relay. `n_fake` means items per cluster. Ranking score dots the layer-combined user vector with layer-0 item rows. Local item updates use SGD; the user embedding uses Adam. Privacy is off by default and the paper publishes no clip or noise scale.

**Instantiation**

```python
from fedreclab.methods.semidfegl.semidfegl import SemiDFEGL
from fedreclab.methods.semidfegl.semidfegl_client import SemiDFEGLClientAlgorithm
from fedreclab.federated.aggregation import FedAvg
from fedreclab.federated.client_selection import FullClientSelection

client = SemiDFEGLClientAlgorithm(embedding_dim=64, num_layers=4)
server = SemiDFEGL(
    embedding_dim=64,
    interaction_dataset=dataset,
    client_algorithm=client,
    num_rounds=50,
    seed=42,
    client_selection_strategy=FullClientSelection(dataset.num_clients),
    aggregation_strategies=FedAvg(num_clients=dataset.num_clients),
    evaluator=evaluator,
)
```

No training notebook yet. Exercised in `tests/semidfegl/`.

---



## Writing a custom method

Subclass `[BaseFederatedServer](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/base_server.py)` and `[BaseClientAlgorithm](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/base_client.py)`, declare each exchanged parameter group with `[ParamIO](https://github.com/aymenkhs/FedRecLab/blob/master/src/fedreclab/federated/utils.py)`, and put method-specific logic in the server hooks (`_initialize_server_parameters`, `_build_server_payload`, `_apply_aggregated_result_custom`, `_on_round_end_server`, …). Prefer hooks over overriding pipeline methods. Aggregators stay experiment configuration: do not special-case aggregator classes inside the method.

**Worked examples**

- New method from scratch (popularity baseline): `[examples/implement_your_fed_rec.ipynb](https://github.com/aymenkhs/FedRecLab/blob/master/examples/implement_your_fed_rec.ipynb)`.
- Change the loss of an existing method, or keep layers on the client: `[examples/extend_fedncf.ipynb](https://github.com/aymenkhs/FedRecLab/blob/master/examples/extend_fedncf.ipynb)`.

