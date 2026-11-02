"""Graph-guided personalisation for federated recommendation (GPFedRec) server.

Source: Zhang, Long, Zhou, Zhang, Yan and Yang, "GPFedRec: Graph-Guided
Personalization for Federated Recommendation", KDD 2024
(https://doi.org/10.1145/3637528.3671702); reference code
``GPFedRec-main`` (``engine.py``, ``utils.py``, ``train.py``).

Task: implicit-feedback ranking. Every client holds one user vector per
local user, a full item table and a personal MLP score function. Clients
train on BCE over observed positives and sampled negatives, plus
``reg * MSE(item_table, r_i)``, and upload their full trained item table.

Server step, each round, over the tables ``Q`` uploaded this round:

1. Cosine similarity ``S`` between the flattened tables.
2. Relation graph with an adaptive threshold ``gamma * mean(S)``,
   row-normalised; a client with no neighbour gets a self-loop.
3. User-specific tables ``R = A_hat^l Q``. Row ``r_i`` is sent back to
   client ``i`` as its regularisation target.
4. Shared table ``q_global`` = configured aggregator over ``R``. Clients
   start each round from ``q_global``.

Faithful configuration: ``FullClientSelection`` with ``SimpleAvg``
(``q_global = mean(R)``, as in the reference). Defaults follow the
reference ``train.py``: ``latent_dim=32``, ``neighborhood_threshold=1.0``,
``mp_layers=1``; client defaults are listed on
:class:`~src.fedreclab.methods.gpfedrec.gpfedrec_client.GPFedRecClientAlgorithm`.

Exchanged data: ``item_embeddings`` is the only ParamIO group (download
``q_global``, upload the full table). ``r_i`` is server-held per-client
state, not a ParamIO group: it has no upload, aggregator or exchange
rule. It is kept in ``GPFedRec.personalized_item_embeddings`` and sent
under the payload key ``personalized_item_embeddings``. It is not stored
in ``global_parameters``, so the early-stopping snapshot does not copy it.

The graph step needs every individual upload, so GPFedRec cannot run
under secure aggregation. Local differential privacy (Laplace noise on
the upload, paper Eq. 9) is a client option and is off by default.

Simulation shortcuts: none.

Deviations from the paper and the reference code:

- Neighbour rule. ``neighbor_rule="similarity"`` (default) follows the
  paper (Eq. 3-4): ``S_ij > gamma * mean(S)``, the user itself included.
  The reference code thresholds the cosine *distance* ``D = 1 - S``
  instead (``D_ij > gamma * mean(D)``), which keeps dissimilar users and
  excludes the user itself; this is ``neighbor_rule="reference_distance"``.
  The best ``gamma`` and ``reg`` reported in the paper (0.5 each) were
  measured with the reference code, so ``gamma`` values do not carry over
  to the paper rule.
- The adjacency is row-normalised and ``q_global`` is the aggregator over
  ``R`` (``mean(R)`` under ``SimpleAvg``), as in the reference code. The
  mean of ``R`` weights each upload by how often it is chosen as a
  neighbour.
- Negatives exclude only the user's training items. The reference also
  excludes validation and test items, which uses held-out data in
  training.
- Evaluation is the library's evaluator, not the reference's
  leave-one-out with 99 sampled negatives.
- Partial participation (the reference supports only full
  participation): the graph covers this round's participants only.
  Non-participants keep their last ``r_i``; clients never aggregated
  receive ``q_global`` as ``r_i``, which is the reference's round-0
  behaviour.
- A client that has never trained scores with ``q_global``.
- Clients with several users are supported; all users of a client share
  one item table and score function.
- Not implemented: Light-GPFedRec (refreshing the graph less often), the
  reference's top-k neighbourhood option, and similarity metrics other
  than cosine.

Memory: every client keeps a dense ``(num_items, latent_dim)`` table, the
server keeps one ``r_i`` per client, and an early-stopping snapshot
copies the client tables again. One such copy is about 0.2 GB on
ML-100K, 2.9 GB on ML-1M and 2.6 GB on Lastfm-2K at ``latent_dim=32``.
During aggregation the server also holds the stacked uploads, their
normalised copy and ``R``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.fedreclab.federated.base_server import BaseFederatedServer
from src.fedreclab.federated.utils import ParamIO
from src.fedreclab.methods.gpfedrec.gpfedrec_model import initial_item_table

NEIGHBOR_RULES = ("similarity", "reference_distance")


def default_param_ios() -> dict[str, ParamIO]:
    """Return the GPFedRec ParamIO map.

    ``item_embeddings`` is server-resident, exchanged as full dense
    parameters and written on the server. It must stay dense: the MSE
    regulariser moves every row of the table.

    Returns
    -------
    dict of str to ParamIO
        One group, ``item_embeddings``.

    Examples
    --------
    >>> pio = default_param_ios()["item_embeddings"]
    >>> pio.residency, pio.exchange, pio.apply_to, pio.layout
    ('server', 'params', 'server', 'dense')
    """
    return {
        "item_embeddings": ParamIO(
            residency="server",
            exchange="params",
            apply_to="server",
            layout="dense",
        ),
    }


def cosine_similarity_matrix(tables: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities between the rows of ``tables``.

    Parameters
    ----------
    tables : ndarray
        Shape ``(n, m)``; one flattened item table per row.

    Returns
    -------
    ndarray
        float64 array of shape ``(n, n)``, clipped to ``[-1, 1]``, with an
        exact diagonal of 1.

    Raises
    ------
    ValueError
        If ``tables`` is not 2-d, contains non-finite values, or a row has
        zero norm (its cosine similarity is undefined).

    Examples
    --------
    >>> cosine_similarity_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    array([[1., 0.],
           [0., 1.]])
    """
    tables = np.asarray(tables)
    if tables.ndim != 2:
        raise ValueError(f"tables must be 2-d (n, m), got shape {tables.shape}")
    if not np.isfinite(tables).all():
        raise ValueError("Uploaded item tables contain non-finite values")
    norms = np.linalg.norm(tables, axis=1)
    zero = np.flatnonzero(norms == 0.0)
    if zero.size:
        raise ValueError(
            f"Uploaded item tables at positions {zero.tolist()} have zero norm; "
            "cosine similarity is undefined"
        )
    normalized = tables / norms[:, None].astype(tables.dtype)
    similarity = (normalized @ normalized.T).astype(np.float64)
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def build_relation_graph(
    similarity: np.ndarray,
    neighbor_rule: str = "similarity",
    threshold: float = 1.0,
) -> np.ndarray:
    """Build the row-normalised relation graph from a similarity matrix.

    The cut-off is ``threshold`` times the mean over the whole matrix,
    diagonal included.

    - ``"similarity"``: ``j`` is a neighbour of ``i`` if
      ``S_ij > threshold * mean(S)``. The diagonal is 1, so ``i`` is its own
      neighbour whenever the cut-off is below 1.
    - ``"reference_distance"``: ``j`` is a neighbour of ``i`` if
      ``D_ij > threshold * mean(D)`` with ``D = 1 - S`` and a zero
      diagonal, so ``i`` is never its own neighbour.

    A row without neighbours gets a self-loop. Rows are then divided by
    their neighbour count.

    Parameters
    ----------
    similarity : ndarray
        Square cosine-similarity matrix from
        :func:`cosine_similarity_matrix`.
    neighbor_rule : {"similarity", "reference_distance"}, optional
        Neighbour selection rule. Default ``"similarity"``.
    threshold : float, optional
        Multiplier ``gamma`` of the mean. Default 1.0.

    Returns
    -------
    ndarray
        float64 array of the same shape; each row sums to 1.

    Raises
    ------
    ValueError
        If ``similarity`` is not square or ``neighbor_rule`` is unknown.

    Examples
    --------
    >>> s = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> build_relation_graph(s, "similarity", 1.0)
    array([[0.5, 0.5, 0. ],
           [0.5, 0.5, 0. ],
           [0. , 0. , 1. ]])
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must be square, got shape {similarity.shape}")
    if neighbor_rule == "similarity":
        adjacency = similarity > float(threshold) * similarity.mean()
    elif neighbor_rule == "reference_distance":
        distance = 1.0 - similarity
        np.fill_diagonal(distance, 0.0)
        adjacency = distance > float(threshold) * distance.mean()
    else:
        raise ValueError(
            f"neighbor_rule must be one of {NEIGHBOR_RULES}, got {neighbor_rule!r}"
        )
    isolated = ~adjacency.any(axis=1)
    adjacency[isolated, isolated] = True
    adjacency = adjacency.astype(np.float64)
    return adjacency / adjacency.sum(axis=1, keepdims=True)


def propagate_tables(
    adjacency: np.ndarray,
    tables: np.ndarray,
    layers: int = 1,
) -> np.ndarray:
    """Return ``adjacency^layers @ tables``.

    Parameters
    ----------
    adjacency : ndarray
        Row-normalised graph of shape ``(n, n)``.
    tables : ndarray
        Flattened tables of shape ``(n, m)``.
    layers : int, optional
        Number of propagation steps ``l``. Default 1.

    Returns
    -------
    ndarray
        Array of shape ``(n, m)`` with the dtype of ``tables``.

    Raises
    ------
    ValueError
        If ``layers < 1`` or the shapes do not match.

    Examples
    --------
    >>> propagate_tables(np.full((2, 2), 0.5), np.array([[0.0], [2.0]]))
    array([[1.],
           [1.]])
    """
    if int(layers) < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    adjacency = np.asarray(adjacency)
    tables = np.asarray(tables)
    if adjacency.shape != (tables.shape[0], tables.shape[0]):
        raise ValueError(
            f"adjacency shape {adjacency.shape} does not match "
            f"{tables.shape[0]} tables"
        )
    adjacency = adjacency.astype(tables.dtype, copy=False)
    out = tables
    for _ in range(int(layers)):
        out = adjacency @ out
    return out


class GPFedRec(BaseFederatedServer):
    """GPFedRec server: relation graph over uploads, ``q_global`` and ``r_i``.

    The server sets ``client_algorithm.init_seed`` from ``seed`` when the
    client leaves it unset, so every client starts from the same seeded
    template.

    Robust aggregators: any aggregator runs, but robustness guarantees do
    not carry over.

    - The graph step runs before the configured aggregator. A malicious
      upload enters every row ``r_i`` that has it as a neighbour before
      Krum, Median or Bulyan see anything, so one attacker can corrupt
      many of the rows the aggregator receives. The honest-majority
      assumption of those rules no longer holds.
    - ``r_i`` never passes through an aggregator; it goes directly to
      the client as its regularisation target.
    - Cosine similarity ignores the norm. A large-norm upload passes the
      threshold as easily as a normal one and then dominates every row it
      enters, because ``R`` mixes the raw tables.
    - Under ``neighbor_rule="reference_distance"``, an outlier has an
      above-average distance to almost everyone and is chosen as a
      neighbour by nearly every client.

    Parameters
    ----------
    latent_dim : int, optional
        Item-vector width. Must match the client. Default 32.
    item_init_std : float, optional
        Standard deviation of the initial ``N(0, std)`` item table.
        Default 1.0 (``nn.Embedding`` default in the reference).
    neighbor_rule : {"similarity", "reference_distance"}, optional
        Neighbour rule, see :func:`build_relation_graph`. Default
        ``"similarity"`` (paper).
    neighborhood_threshold : float, optional
        Multiplier ``gamma`` of the mean similarity or distance. Default 1.0.
    mp_layers : int, optional
        Propagation steps ``l``. Default 1.
    **kwargs
        Passed to :class:`BaseFederatedServer`. ``seed`` is required.

    Attributes
    ----------
    personalized_item_embeddings : dict of int to torch.Tensor
        Last ``r_i`` per client, each an owned CPU float32 tensor of
        shape ``(num_items, latent_dim)``.

    Raises
    ------
    ValueError
        If a hyper-parameter is invalid, ``seed`` is ``None``, the client's
        ``latent_dim`` differs, ``param_io_overrides`` changes the
        ``item_embeddings`` declaration, or the client's
        ``participation_ratio`` disagrees with the selector while
        ``user_lr`` is not set.

    Examples
    --------
    >>> server = GPFedRec(
    ...     latent_dim=32,
    ...     interaction_dataset=data,
    ...     client_algorithm=GPFedRecClientAlgorithm(),
    ...     num_rounds=100,
    ...     client_selection_strategy=FullClientSelection(data.num_clients),
    ...     aggregation_strategies=SimpleAvg(num_clients=data.num_clients),
    ...     seed=0,
    ... )  # doctest: +SKIP
    >>> server.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        latent_dim: int = 32,
        item_init_std: float = 1.0,
        neighbor_rule: str = "similarity",
        neighborhood_threshold: float = 1.0,
        mp_layers: int = 1,
        **kwargs,
    ):
        if neighbor_rule not in NEIGHBOR_RULES:
            raise ValueError(
                f"neighbor_rule must be one of {NEIGHBOR_RULES}, got {neighbor_rule!r}"
            )
        if int(mp_layers) < 1:
            raise ValueError(f"mp_layers must be >= 1, got {mp_layers}")
        if float(item_init_std) <= 0.0:
            raise ValueError(f"item_init_std must be > 0, got {item_init_std}")
        self.latent_dim = int(latent_dim)
        self.item_init_std = float(item_init_std)
        self.neighbor_rule = neighbor_rule
        self.neighborhood_threshold = float(neighborhood_threshold)
        self.mp_layers = int(mp_layers)
        self.personalized_item_embeddings: dict[int, torch.Tensor] = {}

        # Client states are built inside super().__init__, so the template
        # seed must be on the client before that call.
        client = kwargs.get("client_algorithm")
        if client is not None and getattr(client, "init_seed", None) is None:
            client.init_seed = kwargs.get("seed")
        super().__init__(param_ios=default_param_ios(), **kwargs)
        self._check_item_param_io()
        self._check_participation_ratio()

    def _initialize_server_parameters(self) -> dict[str, torch.Tensor]:
        """Draw the initial shared item table from a seeded ``N(0, std)``.

        Returns
        -------
        dict
            ``item_embeddings``: float32 tensor ``(num_items, latent_dim)``
            on ``device`` (CPU when ``device`` is ``None``).

        Raises
        ------
        ValueError
            If ``seed`` is ``None`` or the client's ``latent_dim`` differs.
        """
        client_dim = getattr(self.client_algorithm, "latent_dim", None)
        if client_dim is not None and int(client_dim) != self.latent_dim:
            raise ValueError(
                f"GPFedRec latent_dim={self.latent_dim} does not match client "
                f"latent_dim={client_dim}"
            )
        if self.seed is None:
            raise ValueError(
                "GPFedRec needs seed= for the initial item table and local sampling"
            )
        table = initial_item_table(
            self.interaction_dataset.num_items,
            self.latent_dim,
            self.item_init_std,
            int(self.seed),
        )
        if self.device is not None:
            table = table.to(self.device)
        return {"item_embeddings": table}

    def _build_server_payload(self, client_id: int | None = None) -> dict[str, Any]:
        """Return ``q_global`` and the client's ``r_i``.

        Parameters
        ----------
        client_id : int or None, optional
            Client the payload is for. ``None`` or a client without a
            computed ``r_i`` receives ``q_global`` as ``r_i``.

        Returns
        -------
        dict
            ``item_embeddings`` (``q_global``) and
            ``personalized_item_embeddings`` (``r_i``). Both are the stored
            objects, not copies; the client must not modify them.
        """
        payload = super()._build_server_payload(client_id=client_id)
        q_global = payload["item_embeddings"]
        r_i = q_global
        if client_id is not None:
            r_i = self.personalized_item_embeddings.get(int(client_id), q_global)
        payload["personalized_item_embeddings"] = r_i
        return payload

    def _aggregate_group(
        self,
        param_name: str,
        param_io: ParamIO,
        updates: list[Any],
        participant_ids: list[int],
    ) -> None:
        """Build the relation graph, then aggregate ``R`` into ``q_global``.

        The graph step works on flattened float32 NumPy tables. ``R`` is
        then reshaped to ``(n, num_items, latent_dim)`` and converted with
        ``_convert_to_framework_compatible``, so the aggregator receives
        the same kind of stacked input as in the base class. Its result is
        written through the built-in server apply path. Each participant's
        ``r_i`` is stored as an owned CPU copy.

        Parameters
        ----------
        param_name : str
            Group being aggregated. Groups other than ``item_embeddings``
            use the base implementation.
        param_io : ParamIO
            Declaration of the group.
        updates : list
            Uploaded tables, aligned with ``participant_ids``.
        participant_ids : list of int
            Clients that uploaded this round.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an upload has the wrong shape, is non-finite or has zero
            norm, or the aggregator returns a value of the wrong shape or
            with non-finite entries.
        """
        if param_name != "item_embeddings":
            super()._aggregate_group(param_name, param_io, updates, participant_ids)
            return
        self._assert_updates_match_layout(param_name, param_io, updates)

        previous = self.global_parameters[param_name]
        table_shape = tuple(previous.shape)
        flat = np.stack(
            [
                self._flat_upload(update, table_shape, client_id)
                for update, client_id in zip(updates, participant_ids)
            ]
        )
        adjacency = build_relation_graph(
            cosine_similarity_matrix(flat),
            self.neighbor_rule,
            self.neighborhood_threshold,
        )
        propagated = propagate_tables(adjacency, flat, self.mp_layers).reshape(
            (len(participant_ids),) + table_shape
        )

        aggregated = self.aggregation_strategies[param_name].aggregate(
            self._round_statistics_for_participants(participant_ids),
            previous,
            self._convert_to_framework_compatible(propagated),
            param_io,
        )
        q_global = self._coerce_like(aggregated, previous)
        if tuple(q_global.shape) != table_shape:
            raise ValueError(
                f"Aggregator for '{param_name}' returned shape "
                f"{tuple(q_global.shape)}, expected {table_shape}"
            )
        finite = (
            torch.isfinite(q_global).all()
            if isinstance(q_global, torch.Tensor)
            else np.isfinite(q_global).all()
        )
        if not finite:
            raise ValueError(f"Aggregator for '{param_name}' returned non-finite values")
        # Some aggregators return a view of one update; copying keeps the
        # round's stacked R from staying alive through global_parameters.
        q_global = q_global.clone() if isinstance(q_global, torch.Tensor) else q_global.copy()
        self._apply_aggregated_result(
            param_name=param_name,
            aggregated_result=q_global,
            param_io=param_io,
            selected_client_ids=participant_ids,
        )
        for row, client_id in enumerate(participant_ids):
            self.personalized_item_embeddings[int(client_id)] = torch.from_numpy(
                propagated[row]
            ).clone()

    @staticmethod
    def _flat_upload(update: Any, table_shape: tuple[int, ...], client_id: int) -> np.ndarray:
        """Return one upload as a flat float32 NumPy vector.

        Parameters
        ----------
        update : torch.Tensor or array-like
            Uploaded table.
        table_shape : tuple of int
            Expected ``(num_items, latent_dim)``.
        client_id : int
            Uploading client, used in error messages.

        Returns
        -------
        ndarray
            float32 array of length ``num_items * latent_dim``.

        Raises
        ------
        ValueError
            If the upload's shape is not ``table_shape``.
        """
        if isinstance(update, torch.Tensor):
            array = update.detach().cpu().numpy()
        else:
            array = np.asarray(update)
        if tuple(array.shape) != table_shape:
            raise ValueError(
                f"Client {client_id} uploaded item_embeddings of shape "
                f"{tuple(array.shape)}, expected {table_shape}"
            )
        return array.astype(np.float32, copy=False).reshape(-1)

    def _check_item_param_io(self) -> None:
        """Raise if overrides changed the ``item_embeddings`` declaration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``item_embeddings`` is not ``params`` / ``dense`` /
            ``server``-applied with server residency.
        """
        pio = self.param_ios["item_embeddings"]
        actual = (pio.residency, pio.exchange, pio.apply_to, pio.layout)
        expected = ("server", "params", "server", "dense")
        if actual != expected:
            raise ValueError(
                "GPFedRec requires item_embeddings as (residency, exchange, "
                f"apply_to, layout) = {expected}, got {actual}. The graph step "
                "needs every client's full table, and q_global is written on "
                "the server."
            )

    def _check_participation_ratio(self) -> None:
        """Raise if the client's ``participation_ratio`` disagrees with the selector.

        The ratio is used only for the client's user learning rate. Cohort
        sizes are compared, ``int(num_clients * participation_ratio)``
        against ``number_selected_clients``, because selectors floor
        fractions. The check is skipped when the client sets ``user_lr``
        or has no ``participation_ratio``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the cohort sizes differ.
        """
        client = self.client_algorithm
        if getattr(client, "user_lr", None) is not None:
            return
        ratio = getattr(client, "participation_ratio", None)
        if ratio is None:
            return
        selected = int(self.client_selection_module.number_selected_clients)
        implied = int(self.num_clients * float(ratio))
        if implied == selected:
            return
        selector_ratio = selected / self.num_clients
        selector_lr = (
            f"{client.user_lr_for_ratio(selector_ratio):.6g}" if selected > 0 else "undefined"
        )
        raise ValueError(
            f"Client participation_ratio={ratio} implies {implied} clients per "
            f"round and user_lr={client.user_lr_for_ratio(ratio):.6g}, but the "
            f"selector picks {selected} of {self.num_clients} clients "
            f"(ratio {selector_ratio:.6g}, user_lr={selector_lr}). Set "
            "participation_ratio to match the selector or pass user_lr."
        )
