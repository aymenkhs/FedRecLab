"""Client selection strategies.

A selector returns unique, valid client ids. Length is ``m`` except the
short last cohort of round-robin.

``RandomClientSelection`` draws uniformly, or without replacement with
probability proportional to ``weights``. Size-weighted draws already
put data size into who is selected; averaging those updates again by
``num_training_examples`` (``FedAvg``) counts size twice. Use
``SimpleAvg`` in that case. Uniform draws plus ``FedAvg`` is the
usual FedAvg pairing.

``replace=True`` (with-replacement draws) is not implemented.

Seeded ``RandomClientSelection`` is a pure function of
``(seed, round)``: a fresh instance with the same pair repeats that
round. Round-robin and Power-of-Choice keep a cursor, permutation, or
stored losses, so the next cohort depends on call history.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from src.fedreclab.federated.utils import RoundStatistics

logger = logging.getLogger(__name__)


def _as_probability_vector(
    num_clients: int,
    weights: Mapping[int, float] | Sequence[float] | None,
) -> np.ndarray | None:
    """Normalise client weights to a length-``K`` probability vector.

    Missing mapping keys are treated as zero. A ``None`` input means
    uniform sampling (the caller passes ``p=None`` to ``Generator.choice``).

    Parameters
    ----------
    num_clients : int
        Population size ``K``.
    weights : mapping or sequence or None
        Client id → unnormalised weight, or a length-``K`` sequence.

    Returns
    -------
    ndarray or None
        Probabilities summing to 1, or ``None`` when ``weights`` is
        ``None``.

    Raises
    ------
    ValueError
        If a sequence has the wrong length, a weight is negative or
        non-finite, an id is out of range, or every weight is zero.

    Examples
    --------
    >>> _as_probability_vector(3, {0: 1, 2: 3})
    array([0.25, 0.  , 0.75])
    >>> _as_probability_vector(2, None) is None
    True
    """
    if weights is None:
        return None
    p = np.zeros(num_clients, dtype=np.float64)
    if isinstance(weights, Mapping):
        for key, value in weights.items():
            cid = int(key)
            if cid < 0 or cid >= num_clients:
                raise ValueError(
                    f"weight key {cid} is outside [0, {num_clients})"
                )
            p[cid] = float(value)
    else:
        arr = np.asarray(weights, dtype=np.float64)
        if arr.shape != (num_clients,):
            raise ValueError(
                f"weights sequence must have length {num_clients}, "
                f"got shape {arr.shape}"
            )
        p = arr
    if not np.all(np.isfinite(p)):
        raise ValueError("client weights must be finite")
    if np.any(p < 0):
        raise ValueError("client weights must be non-negative")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("client weights must sum to a positive value")
    return p / total


class BaseClientSelection:
    """Shared constructor for the number of clients drawn per round.

    Subclasses implement ``select``. ``on_round_start`` / ``on_round_end``
    are no-ops unless a selector needs round statistics (rpow-d).

    Parameters
    ----------
    num_clients : int
        Population size ``K``.
    number_selected_clients : int or float
        Count ``m`` when ``selection_value_type`` is ``"num"`` (or an
        int in ``"auto"``). Fraction of ``K`` when the type is
        ``"frac"`` (or a float in ``"auto"``).
    selection_value_type : {"auto", "num", "frac"}, optional
        How to interpret ``number_selected_clients``. Default ``"auto"``
        treats ints as counts and floats as fractions.
    seed : int or None, optional
        Selector seed. ``None`` leaves seeding to a local generator or
        to the server, which writes ``selector.seed = server.seed``
        when this is still ``None``.

    Attributes
    ----------
    number_selected_clients : int
        Resolved ``m``.

    Raises
    ------
    ValueError
        If the resolved ``m`` is outside ``[1, K]``, or a fraction is
        not in ``(0, 1]``.

    Examples
    --------
    >>> BaseClientSelection(10, 0.2).number_selected_clients
    2
    >>> BaseClientSelection(10, 3, "num").number_selected_clients
    3
    """

    def __init__(
        self,
        num_clients,
        number_selected_clients,
        selection_value_type="auto",
        seed=None,
    ):
        self.num_clients = num_clients
        self.seed = seed
        if selection_value_type is None or selection_value_type == "auto":
            if type(number_selected_clients) is float:
                selection_value_type = "frac"
            elif type(number_selected_clients) is int:
                selection_value_type = "num"
            else:
                raise ValueError("Invalid selection method")

        if selection_value_type == "frac":
            if not 0 < number_selected_clients <= 1:
                raise ValueError(
                    f"selection_value_type='frac' expects a fraction in (0, 1], "
                    f"got {number_selected_clients}. Use selection_value_type='num' "
                    f"to select a fixed client count."
                )
            self.number_selected_clients = int(num_clients * number_selected_clients)
        elif selection_value_type == "num":
            self.number_selected_clients = number_selected_clients
        else:
            raise ValueError("Invalid selection method")

        if (
            self.number_selected_clients < 1
            or self.number_selected_clients > self.num_clients
        ):
            raise ValueError(
                f"Cannot select {self.number_selected_clients} clients from a "
                f"population of {self.num_clients} without replacement."
            )

    def select(self, current_round=0):
        """Return the client ids that train this round.

        Parameters
        ----------
        current_round : int, optional
            Round index. Seeded uniform sampling folds this into the
            RNG so the same round is reproducible on a fresh instance.

        Returns
        -------
        list of int
            Unique client ids.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def on_round_start(self, previous_round_statistics: RoundStatistics):
        """Hook before ``select``. Default is a no-op.

        Parameters
        ----------
        previous_round_statistics : RoundStatistics
            Statistics of the previous round, or ``None`` on round 0.

        Returns
        -------
        None
        """
        pass

    def on_round_end(self, current_round_statistics: RoundStatistics):
        """Hook after aggregation. Default is a no-op.

        Parameters
        ----------
        current_round_statistics : RoundStatistics
            Statistics of the round that just finished.

        Returns
        -------
        None
        """
        pass

    def _stored_rng(self) -> np.random.Generator:
        """Return a generator owned by this instance, created on first use.

        Lazy so the server can assign ``seed`` after construction.

        Returns
        -------
        numpy.random.Generator
            ``default_rng(self.seed)``, stored on ``_rng``.
        """
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = np.random.default_rng(self.seed)
            self._rng = rng
        return rng


class FullClientSelection(BaseClientSelection):
    """Every client trains every round.

    Parameters
    ----------
    num_clients : int
        Population size. ``m`` is set equal to this.

    Examples
    --------
    >>> FullClientSelection(4).select()
    [0, 1, 2, 3]
    """

    def __init__(self, num_clients):
        super().__init__(
            num_clients=num_clients,
            number_selected_clients=num_clients,
            selection_value_type="num",
        )

    def select(self, current_round=0):
        """Return ``[0, 1, ..., K-1]``.

        Parameters
        ----------
        current_round : int, optional
            Ignored.

        Returns
        -------
        list of int
            Every client id.
        """
        return list(range(self.num_clients))


class RandomClientSelection(BaseClientSelection):
    """Uniform or size-weighted random sampling without replacement.

    Default (``weights=None``) is uniform sampling without replacement.
    Pair that with ``FedAvg`` to weight the selected updates by
    ``num_training_examples``.

    With ``weights``, each draw picks client ``k`` with probability
    proportional to that weight, without replacement. For ``m > 1`` the
    inclusion probabilities are then not exactly those weights. Pair
    with ``SimpleAvg`` so data size is not applied again at aggregation.

    Seeded draws rebuild ``default_rng([seed, current_round])`` so an
    existing (seed, round) pair stays reproducible. Unseeded draws use a
    stored generator and never touch global ``np.random``.

    Parameters
    ----------
    num_clients : int
        Population size ``K``.
    number_selected_clients : int or float
        ``m``, or a fraction of ``K`` (see the base class).
    selection_value_type : {"auto", "num", "frac"}, optional
        How to interpret ``number_selected_clients``.
    seed : int or None, optional
        Seed folded with the round index when not ``None``.
    weights : mapping or sequence or None, optional
        Unnormalised ``n_k``, typically
        ``ClientPartition.client_train_sizes``. ``None`` is uniform.
    replace : bool, optional
        Must be False. With-replacement sampling is not implemented.

    Attributes
    ----------
    probabilities : ndarray or None
        Normalised ``p``, or ``None`` for uniform.

    Raises
    ------
    NotImplementedError
        If ``replace=True``.
    ValueError
        If weights are invalid, or ``m`` exceeds the number of clients
        with positive weight.

    Examples
    --------
    >>> sel = RandomClientSelection(10, 3, seed=0)
    >>> len(sel.select(0)) == 3
    True
    >>> wsel = RandomClientSelection(
    ...     4, 2, seed=0, weights={0: 1, 1: 0, 2: 1, 3: 2})
    >>> 1 in wsel.select(0)
    False
    """

    def __init__(
        self,
        num_clients,
        number_selected_clients,
        selection_value_type="auto",
        seed=None,
        weights=None,
        replace=False,
    ):
        super().__init__(
            num_clients, number_selected_clients, selection_value_type, seed=seed
        )
        # TODO: with-replacement sampling. Duplicate draws must not be
        # folded into the aggregator (that mixes sampling and
        # aggregation), and training the same client twice would fork
        # client-resident state. Keep replace=False until there is a
        # contract that keeps the two separate.
        if replace:
            raise NotImplementedError(
                "RandomClientSelection(replace=True) is not implemented. "
                "With-replacement sampling needs a way to count duplicate "
                "draws without mixing that into aggregation."
            )
        self.replace = False
        self.weights = weights
        self.probabilities = _as_probability_vector(self.num_clients, weights)
        self._rng = None
        if self.probabilities is not None:
            n_positive = int(np.count_nonzero(self.probabilities))
            if self.number_selected_clients > n_positive:
                raise ValueError(
                    f"Cannot select {self.number_selected_clients} clients "
                    f"from {n_positive} clients with positive weight "
                    f"without replacement."
                )

    def _rng_for_round(self, current_round: int) -> np.random.Generator:
        """Generator for this call.

        Seeded instances rebuild from ``[seed, round]`` so existing
        sequences do not change. Unseeded instances own one generator.

        Parameters
        ----------
        current_round : int
            Round index folded into the seeded generator.

        Returns
        -------
        numpy.random.Generator
        """
        if self.seed is None:
            return self._stored_rng()
        return np.random.default_rng([self.seed, current_round])

    def select(self, current_round=0):
        """Draw ``m`` distinct clients.

        Parameters
        ----------
        current_round : int, optional
            Round index. Used only when ``seed`` is set.

        Returns
        -------
        list of int
            Unique client ids of length ``m``.
        """
        rng = self._rng_for_round(current_round)
        ids = rng.choice(
            self.num_clients,
            size=self.number_selected_clients,
            replace=False,
            p=self.probabilities,
        )
        return [int(i) for i in np.atleast_1d(ids)]


class RoundRobinClientSelection(BaseClientSelection):
    """Cyclic cohorts: each client trains once per cycle by default.

    Without ``groups`` this is RR-CLI / CyCP with the whole cohort
    training: a permutation of ``[0, K)``, take the next ``m`` ids,
    advance the cursor. When ``K`` is not divisible by ``m`` (both
    source papers assume it is):

    * ``remainder="short_last"`` (default) keeps exactly-once coverage
      with a smaller final cohort.
    * ``remainder="wrap"`` always returns ``m`` ids by taking the rest
      from the next cycle, which breaks exactly-once across rounds.
      A mid-cohort reshuffle rotates ids already in the cohort to the
      back of the new permutation so the round still has unique ids.

    With ``groups``, the given disjoint lists are CyCP groups visited in
    order. ``clients_per_group=None`` trains the whole group;
    otherwise ``N`` clients are sampled uniformly without replacement
    from the current group.

    Pair with ``SimpleAvg``. Ignores ``RoundStatistics``, so the
    sequence depends only on the seed. Dropout re-queue is not
    implemented: a dropped client still loses that turn.

    Parameters
    ----------
    num_clients : int
        Population size ``K``.
    number_selected_clients : int or float
        Cohort size ``m`` when ``groups`` is ``None``.
    selection_value_type : {"auto", "num", "frac"}, optional
        How to interpret ``number_selected_clients``.
    seed : int or None, optional
        Seed of the owned generator (permutation and within-group
        sampling).
    reshuffle_each_cycle : bool, optional
        If True, draw a new permutation at the end of each cycle.
        Default False is shuffle-once. Both are covered by RR-CLI.
    groups : sequence of sequences of int or None, optional
        Fixed CyCP groups. ``None`` uses the permutation of all
        clients.
    clients_per_group : int or None, optional
        ``N`` in CyCP. ``None`` means the whole current group.
    remainder : {"short_last", "wrap"}, optional
        How to handle ``K % m != 0`` when ``groups`` is ``None``.

    Raises
    ------
    ValueError
        If ``remainder`` is unknown, ``clients_per_group`` is set
        without ``groups``, groups are empty, overlap, or contain
        invalid ids, or ``clients_per_group`` exceeds a group.

    Examples
    --------
    >>> rr = RoundRobinClientSelection(5, 2, seed=0)
    >>> seen = rr.select(0) + rr.select(1) + rr.select(2)
    >>> sorted(seen)
    [0, 1, 2, 3, 4]
    """

    def __init__(
        self,
        num_clients,
        number_selected_clients,
        selection_value_type="auto",
        seed=None,
        reshuffle_each_cycle=False,
        groups=None,
        clients_per_group=None,
        remainder="short_last",
    ):
        super().__init__(
            num_clients, number_selected_clients, selection_value_type, seed=seed
        )
        if remainder not in ("short_last", "wrap"):
            raise ValueError(
                f"remainder must be 'short_last' or 'wrap', got {remainder!r}"
            )
        if clients_per_group is not None and groups is None:
            raise ValueError(
                "clients_per_group requires groups; without groups the "
                "cohort size is number_selected_clients"
            )
        self.reshuffle_each_cycle = bool(reshuffle_each_cycle)
        self.remainder = remainder
        self.clients_per_group = clients_per_group
        self._rng = None
        self._order: list[int] | None = None
        self._cursor = 0
        self._group_index = 0
        self.groups = None if groups is None else self._validate_groups(groups)

    def _validate_groups(self, groups: Sequence[Sequence[int]]) -> list[list[int]]:
        """Check CyCP groups and return them as lists of ints.

        Parameters
        ----------
        groups : sequence of sequences of int
            Proposed disjoint groups.

        Returns
        -------
        list of list of int
            Copied groups.

        Raises
        ------
        ValueError
            If a group is empty, ids overlap or are out of range, or
            ``clients_per_group`` does not fit in every group.
        """
        parsed: list[list[int]] = []
        seen: set[int] = set()
        for g in groups:
            group = [int(cid) for cid in g]
            if len(group) == 0:
                raise ValueError("CyCP groups must be non-empty")
            for cid in group:
                if cid < 0 or cid >= self.num_clients:
                    raise ValueError(
                        f"group id {cid} is outside [0, {self.num_clients})"
                    )
                if cid in seen:
                    raise ValueError(f"client {cid} appears in more than one group")
                seen.add(cid)
            parsed.append(group)
        if len(parsed) == 0:
            raise ValueError("groups must contain at least one group")
        if self.clients_per_group is not None:
            n = int(self.clients_per_group)
            if n < 1:
                raise ValueError("clients_per_group must be >= 1")
            too_small = [i for i, g in enumerate(parsed) if len(g) < n]
            if too_small:
                raise ValueError(
                    f"clients_per_group={n} exceeds the size of groups {too_small}"
                )
        return parsed

    def _ensure_order(self) -> None:
        """Shuffle the population once, on first use, from the owned RNG."""
        if self._order is None:
            self._order = [int(i) for i in self._stored_rng().permutation(self.num_clients)]
            self._cursor = 0

    def _reshuffle(self) -> None:
        """Replace the permutation at the end of a cycle."""
        self._order = [int(i) for i in self._stored_rng().permutation(self.num_clients)]
        self._cursor = 0

    def select(self, current_round=0):
        """Return the next cohort or CyCP group sample.

        Parameters
        ----------
        current_round : int, optional
            Ignored; the cursor is the only state.

        Returns
        -------
        list of int
            Unique client ids. Length is ``m``, except a short last
            cohort, or the size of the current CyCP group.
        """
        if self.groups is not None:
            return self._select_from_groups()
        return self._select_from_permutation()

    def _select_from_groups(self) -> list[int]:
        """Visit the next CyCP group, optionally subsampling it."""
        group = self.groups[self._group_index]
        self._group_index = (self._group_index + 1) % len(self.groups)
        n = self.clients_per_group
        if n is None or n >= len(group):
            return list(group)
        chosen = self._stored_rng().choice(group, size=int(n), replace=False)
        return [int(i) for i in np.atleast_1d(chosen)]

    def _select_from_permutation(self) -> list[int]:
        """Advance the cursor along the current permutation."""
        self._ensure_order()
        m = self.number_selected_clients
        order = self._order
        if self.remainder == "short_last":
            remaining = len(order) - self._cursor
            take = min(m, remaining)
            chosen = order[self._cursor : self._cursor + take]
            self._cursor += take
            if self._cursor >= len(order):
                if self.reshuffle_each_cycle:
                    self._reshuffle()
                else:
                    self._cursor = 0
            return list(chosen)

        chosen: list[int] = []
        while len(chosen) < m:
            if self._cursor >= len(self._order):
                if self.reshuffle_each_cycle:
                    self._reshuffle()
                    # New permutation can put an already-chosen id first;
                    # those would duplicate in this round. Keep unused ids
                    # at the front; each id still appears once in the cycle.
                    taken = set(chosen)
                    self._order = (
                        [c for c in self._order if c not in taken]
                        + [c for c in self._order if c in taken]
                    )
                else:
                    self._cursor = 0
            chosen.append(self._order[self._cursor])
            self._cursor += 1
        return chosen


class PowerOfChoiceClientSelection(BaseClientSelection):
    """rpow-d: sample ``d`` candidates by ``p_k``, keep the ``m`` highest stored losses.

    Communication-free variant of Cho, Wang and Joshi (AISTATS 2022).
    Candidates are drawn without replacement with probability ``p_k``.
    The server never asks for a fresh loss: selected clients' mean of
    ``training_losses`` (per-epoch batch means in FedRecLab, a slight
    deviation from the paper's per-step mean) is stored in
    ``on_round_end``. Unseen clients stay at ``+∞``, so they are
    preferred until they have trained once.

    Pair with ``SimpleAvg``. ``d = m`` reduces to weighted sampling
    without replacement. ``d_schedule`` implements adapow-d by shrinking
    ``d`` toward ``m``. pow-d / cpow-d need a pre-train eval phase and
    are not implemented.

    Information flow: the selector reads per-client losses
    (``requires_individual_losses = True``). Under a future
    ``aggregate_only`` visibility mode those scalars would travel
    outside secure aggregation.

    Parameters
    ----------
    num_clients : int
        Population size ``K``.
    number_selected_clients : int or float
        ``m``.
    d : int
        Candidate-set size, ``m <= d <= K``.
    selection_value_type : {"auto", "num", "frac"}, optional
        How to interpret ``number_selected_clients``.
    seed : int or None, optional
        Seed of the owned generator.
    weights : mapping or sequence or None, optional
        Unnormalised ``p_k``. ``None`` is uniform.
    d_schedule : callable or None, optional
        ``d_schedule(current_round) -> d_t``, clamped to ``[m, K]``.

    Attributes
    ----------
    last_loss : ndarray
        Length-``K`` proxy losses, initialised to ``+∞``.
    requires_individual_losses : bool
        Always True for this class.

    Raises
    ------
    ValueError
        If ``d`` is outside ``[m, K]``, weights are invalid, or ``m``
        exceeds the number of clients with positive weight.

    Examples
    --------
    >>> poc = PowerOfChoiceClientSelection(5, 2, d=3, seed=0)
    >>> len(poc.select(0))
    2
    """

    requires_individual_losses = True

    def __init__(
        self,
        num_clients,
        number_selected_clients,
        d: int,
        selection_value_type="auto",
        seed=None,
        weights=None,
        d_schedule: Callable[[int], int] | None = None,
    ):
        super().__init__(
            num_clients, number_selected_clients, selection_value_type, seed=seed
        )
        self.d = int(d)
        self.d_schedule = d_schedule
        self.weights = weights
        self.probabilities = _as_probability_vector(self.num_clients, weights)
        self.last_loss = np.full(self.num_clients, np.inf, dtype=np.float64)
        self._rng = None
        m = self.number_selected_clients
        if not m <= self.d <= self.num_clients:
            raise ValueError(
                f"d must satisfy m <= d <= K, got d={self.d}, "
                f"m={m}, K={self.num_clients}"
            )
        n_positive = (
            self.num_clients
            if self.probabilities is None
            else int(np.count_nonzero(self.probabilities))
        )
        if m > n_positive:
            raise ValueError(
                f"Cannot select {m} clients from {n_positive} "
                f"clients with positive weight."
            )

    def _candidate_pool(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Ids that may be drawn, with renormalised probabilities.

        Returns
        -------
        ids : ndarray
            Client ids with positive weight (all ids if uniform).
        p : ndarray or None
            Probabilities on ``ids``, or ``None`` if uniform.
        """
        if self.probabilities is None:
            return np.arange(self.num_clients), None
        ids = np.flatnonzero(self.probabilities)
        p = self.probabilities[ids]
        p = p / p.sum()
        return ids, p

    def select(self, current_round=0):
        """Draw ``d_t`` candidates and return the ``m`` with highest stored loss.

        Parameters
        ----------
        current_round : int, optional
            Passed to ``d_schedule`` if one was set.

        Returns
        -------
        list of int
            ``m`` unique client ids.
        """
        m = self.number_selected_clients
        d_t = self.d if self.d_schedule is None else int(self.d_schedule(current_round))
        d_t = int(np.clip(d_t, m, self.num_clients))
        pool, p = self._candidate_pool()
        d_t = min(d_t, int(pool.size))
        if d_t < m:
            raise ValueError(
                f"Need {m} clients but only {d_t} candidates have positive weight"
            )
        rng = self._stored_rng()
        candidates = rng.choice(pool, size=d_t, replace=False, p=p)
        candidates = np.atleast_1d(candidates).astype(int, copy=False)
        finite_share = float(np.isfinite(self.last_loss[candidates]).mean())
        logger.info(
            "Power-of-Choice round %s: %.3f of %s candidates have a finite stored loss",
            current_round,
            finite_share,
            int(candidates.size),
        )
        tie = rng.random(candidates.size)
        # lexsort: last key is primary, ascending. -loss puts +∞ first.
        order = np.lexsort((-tie, -self.last_loss[candidates]))
        chosen = candidates[order[:m]]
        return [int(i) for i in chosen]

    def on_round_end(self, current_round_statistics: RoundStatistics):
        """Store each responding client's mean of ``training_losses``.

        Parameters
        ----------
        current_round_statistics : RoundStatistics
            Must include ``responded`` and ``training_losses``. Empty
            loss lists are skipped so a silent client does not become
            NaN.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a responding client's mean training loss is not finite.
            NaN would otherwise be sorted last by ``lexsort``.
        """
        if current_round_statistics is None:
            return
        for cid in current_round_statistics.responded:
            losses = current_round_statistics.training_losses.get(cid)
            if losses is None or len(losses) == 0:
                continue
            mean = float(np.mean(losses))
            if not np.isfinite(mean):
                raise ValueError(
                    f"client {cid} mean training loss is not finite: {mean}"
                )
            self.last_loss[int(cid)] = mean


class ActvSampClientSelection(BaseClientSelection):
    """ActvSAMP (Muhammad et al., KDD 2020 Algorithm 2): sample across clusters.

    Clients are partitioned into groups ``G``. Each round draws
    ``m // p`` clients uniformly without replacement from every
    cluster (``p = |G|``, ``m = number_selected_clients``).

    ``m`` must be divisible by ``p`` (both are experiment knobs). When a
    cluster has fewer than ``m // p`` members — which can happen after
    re-clustering — that cluster contributes all of its members and any
    shortfall is filled uniformly without replacement from clients not
    yet chosen. That fill rule is a documented deviation from Algorithm 2.

    Groups must be installed with ``set_groups`` before the first
    ``select``. FedFast refreshes groups after each ActvAGG recluster.

    The paper's prose also mentions round-robin one-per-cluster sampling;
    that variant is not implemented.

    Parameters
    ----------
    num_clients : int
        Population size ``K``.
    number_selected_clients : int or float
        Cohort size ``m``, or a fraction of ``K`` (see the base class).
    selection_value_type : {"auto", "num", "frac"}, optional
        How to interpret ``number_selected_clients``.
    seed : int or None, optional
        Seed of the owned generator used for within-cluster draws and
        fill-ins.

    Attributes
    ----------
    groups : list of list of int or None
        Current partition ``G``, or ``None`` until ``set_groups``.
    last_fill_count : int
        Number of clients added by the undersized-cluster fill on the
        most recent ``select`` (0 when every cluster had size
        ``>= m // p``).

    Raises
    ------
    ValueError
        If ``select`` is called before ``set_groups``, if ``m`` is not
        divisible by ``p``, if groups are invalid, or if fewer than
        ``m`` distinct clients exist in the union of groups.

    Examples
    --------
    >>> sel = ActvSampClientSelection(6, 3, seed=0)
    >>> sel.set_groups([[0, 1], [2, 3], [4, 5]])
    >>> chosen = sel.select(0)
    >>> len(chosen) == 3 and len(set(chosen)) == 3
    True
    """

    def __init__(
        self,
        num_clients,
        number_selected_clients,
        selection_value_type="auto",
        seed=None,
    ):
        super().__init__(
            num_clients, number_selected_clients, selection_value_type, seed=seed
        )
        self.groups: list[list[int]] | None = None
        self.last_fill_count = 0
        self._rng = None

    def set_groups(self, groups: Sequence[Sequence[int]]) -> None:
        """Install the client partition used by the next ``select``.

        Parameters
        ----------
        groups : sequence of sequences of int
            Disjoint non-empty clusters whose union is exactly
            ``{0, ..., K-1}``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a group is empty, ids overlap or fall outside
            ``[0, K)``, the union misses a client, or ``m`` is not
            divisible by the number of groups.

        Examples
        --------
        >>> sel = ActvSampClientSelection(4, 2, seed=0)
        >>> sel.set_groups([[0, 1], [2, 3]])
        >>> len(sel.groups)
        2
        """
        parsed = self._validate_groups(groups)
        p = len(parsed)
        m = self.number_selected_clients
        if m % p != 0:
            raise ValueError(
                f"ActvSAMP requires m divisible by p (number of clusters), "
                f"got m={m}, p={p}. Adjust number_selected_clients or the "
                f"number of groups."
            )
        self.groups = parsed

    def _validate_groups(self, groups: Sequence[Sequence[int]]) -> list[list[int]]:
        """Check ActvSAMP groups and return them as lists of ints.

        Parameters
        ----------
        groups : sequence of sequences of int
            Proposed disjoint clusters covering the population.

        Returns
        -------
        list of list of int
            Copied groups.

        Raises
        ------
        ValueError
            If a group is empty, ids overlap or are out of range, or the
            union is not exactly ``[0, K)``.
        """
        parsed: list[list[int]] = []
        seen: set[int] = set()
        for g in groups:
            group = [int(cid) for cid in g]
            if len(group) == 0:
                raise ValueError("ActvSAMP groups must be non-empty")
            for cid in group:
                if cid < 0 or cid >= self.num_clients:
                    raise ValueError(
                        f"group id {cid} is outside [0, {self.num_clients})"
                    )
                if cid in seen:
                    raise ValueError(f"client {cid} appears in more than one group")
                seen.add(cid)
            parsed.append(group)
        if len(parsed) == 0:
            raise ValueError("groups must contain at least one group")
        if len(seen) != self.num_clients:
            missing = sorted(set(range(self.num_clients)) - seen)
            raise ValueError(
                f"ActvSAMP groups must cover all {self.num_clients} clients; "
                f"missing {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        return parsed

    def select(self, current_round=0):
        """Draw ``m // p`` clients from each cluster, with fill if needed.

        Parameters
        ----------
        current_round : int, optional
            Ignored; the owned RNG is the only randomness source.

        Returns
        -------
        list of int
            ``m`` unique client ids.

        Raises
        ------
        ValueError
            If groups were not set, ``m`` is not divisible by ``p``, or
            fewer than ``m`` clients are available.

        Examples
        --------
        >>> sel = ActvSampClientSelection(6, 3, seed=0)
        >>> sel.set_groups([[0, 1], [2, 3], [4, 5]])
        >>> len(sel.select(0))
        3
        """
        if self.groups is None:
            raise ValueError(
                "ActvSampClientSelection.select requires set_groups(...) "
                "before the first call"
            )
        p = len(self.groups)
        m = self.number_selected_clients
        if m % p != 0:
            raise ValueError(
                f"ActvSAMP requires m divisible by p, got m={m}, p={p}"
            )
        k = m // p
        rng = self._stored_rng()
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for group in self.groups:
            take = min(k, len(group))
            if take == 0:
                continue
            picks = rng.choice(group, size=take, replace=False)
            for cid in np.atleast_1d(picks):
                cid_i = int(cid)
                chosen.append(cid_i)
                chosen_set.add(cid_i)

        fill = 0
        if len(chosen) < m:
            remaining = [c for c in range(self.num_clients) if c not in chosen_set]
            need = m - len(chosen)
            if need > len(remaining):
                raise ValueError(
                    f"Cannot fill ActvSAMP cohort to m={m}: only "
                    f"{len(chosen) + len(remaining)} distinct clients available"
                )
            extras = rng.choice(remaining, size=need, replace=False)
            for cid in np.atleast_1d(extras):
                chosen.append(int(cid))
            fill = need
        self.last_fill_count = fill
        if fill > 0:
            logger.info(
                "ActvSAMP round %s: filled %s clients after undersized clusters "
                "(m=%s, p=%s)",
                current_round,
                fill,
                m,
                p,
            )
        return chosen
