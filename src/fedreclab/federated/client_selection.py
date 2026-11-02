

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from src.fedreclab.federated.utils import RoundStatistics


class BaseClientSelection:
    def __init__(self, num_clients, number_selected_clients, selection_value_type='auto', seed=None):
        self.num_clients = num_clients
        self.seed = seed
        if selection_value_type is None or selection_value_type == 'auto':
            if type(number_selected_clients) is float:
                selection_value_type = 'frac'
            elif type(number_selected_clients) is int:
                selection_value_type = 'num'
            else:
                raise ValueError("Invalid selection method")

        if selection_value_type == 'frac':
            if not 0 < number_selected_clients <= 1:
                raise ValueError(
                    f"selection_value_type='frac' expects a fraction in (0, 1], "
                    f"got {number_selected_clients}. Use selection_value_type='num' "
                    f"to select a fixed client count."
                )
            self.number_selected_clients = int(num_clients * number_selected_clients)
        elif selection_value_type == 'num':
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
        raise NotImplementedError("Subclasses must implement this method")

    def on_round_start(self, previous_round_statistics: RoundStatistics):
        pass

    def on_round_end(self, current_round_statistics: RoundStatistics):
        pass


class FullClientSelection(BaseClientSelection):
    def __init__(self, num_clients):
        super().__init__(
            num_clients=num_clients,
            number_selected_clients=num_clients,
            selection_value_type='num'
        )

    def select(self, current_round=0):
        return list(range(self.num_clients))


class RandomClientSelection(BaseClientSelection):
    def __init__(self, num_clients, number_selected_clients, selection_value_type='auto', seed=None):
        super().__init__(num_clients, number_selected_clients, selection_value_type, seed=seed)

    def select(self, current_round=0):
        if self.seed is None:
            ids = np.random.choice(self.num_clients, size=self.number_selected_clients, replace=False)
        else:
            # TODO: how current_round is folded into the seed should be overridable
            # (e.g. [seed, round] vs a stored advancing Generator).
            rng = np.random.default_rng([self.seed, current_round])
            ids = rng.choice(self.num_clients, size=self.number_selected_clients, replace=False)
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
      from the next cycle, which breaks exactly-once.

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
        If ``remainder`` is unknown, groups are empty, overlap, or
        contain invalid ids, or ``clients_per_group`` exceeds a group.

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
        for _ in range(m):
            if self._cursor >= len(order):
                if self.reshuffle_each_cycle:
                    self._reshuffle()
                    order = self._order
                else:
                    self._cursor = 0
            chosen.append(order[self._cursor])
            self._cursor += 1
        return chosen

