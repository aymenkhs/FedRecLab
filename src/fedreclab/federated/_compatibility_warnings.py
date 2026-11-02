"""Selector/aggregator pairings that are legal but easy to misconfigure.

Sampling and aggregation stay independent. This module only stores
``(selector class, aggregator class) → warning message``. The server
walks the table at construction and logs each match.

Lookup uses exact types (``type(obj) is cls``), not ``isinstance``.
``SimpleAvg`` subclasses ``FedAvg``, so an ``isinstance(..., FedAvg)``
check would also fire for ``SimpleAvg`` — the pairing these rows
recommend, not the one they warn about.

Parameters
----------
None
    This module is a table plus ``iter_selector_aggregator_warnings``.

Examples
--------
>>> from src.fedreclab.federated.aggregation import FedAvg, SimpleAvg
>>> from src.fedreclab.federated.client_selection import RandomClientSelection
>>> sel = RandomClientSelection(4, 2, weights={0: 1, 1: 2, 2: 3, 3: 4})
>>> list(iter_selector_aggregator_warnings(sel, [FedAvg(4)])) != []
True
>>> list(iter_selector_aggregator_warnings(sel, [SimpleAvg(4)]))
[]
"""

from collections.abc import Callable, Iterable, Iterator

from src.fedreclab.federated.aggregation import FedAvg
from src.fedreclab.federated.client_selection import (
    PowerOfChoiceClientSelection,
    RandomClientSelection,
    RoundRobinClientSelection,
)

# Exact (selector class, aggregator class) → message logged at server init.
SELECTOR_AGGREGATOR_WARNINGS: dict[tuple[type, type], str] = {
    (RandomClientSelection, FedAvg): (
        "RandomClientSelection(weights=...) already samples by client "
        "size; FedAvg also weights updates by num_training_examples."
    ),
    (RoundRobinClientSelection, FedAvg): (
        "RoundRobinClientSelection visits clients equally often; "
        "FedAvg then weights their updates by num_training_examples."
    ),
    (PowerOfChoiceClientSelection, FedAvg): (
        "PowerOfChoiceClientSelection already uses client weights when "
        "drawing candidates; FedAvg also weights updates by "
        "num_training_examples."
    ),
}

# Optional extra gate. Missing key means always warn for that pair.
# Uniform RandomClientSelection or PowerOfChoiceClientSelection plus
# FedAvg is the McMahan pairing: size is applied once, at aggregation.
SELECTOR_AGGREGATOR_WARNING_WHEN: dict[
    tuple[type, type], Callable[[object, object], bool]
] = {
    (RandomClientSelection, FedAvg): lambda sel, _agg: sel.weights is not None,
    (PowerOfChoiceClientSelection, FedAvg): lambda sel, _agg: sel.weights is not None,
}


def iter_selector_aggregator_warnings(
    selector: object,
    aggregators: Iterable[object],
) -> Iterator[str]:
    """Yield the configured message for each (selector, aggregator) mismatch.

    Each distinct message is yielded once, even if several parameter
    groups share the same aggregator class.

    Parameters
    ----------
    selector : object
        The experiment's client selector.
    aggregators : iterable
        Aggregators used this run (the values of the server's strategy
        map, or a single aggregator).

    Yields
    ------
    str
        Warning text from ``SELECTOR_AGGREGATOR_WARNINGS``.

    Examples
    --------
    >>> from src.fedreclab.federated.aggregation import FedAvg
    >>> from src.fedreclab.federated.client_selection import RoundRobinClientSelection
    >>> sel = RoundRobinClientSelection(4, 2, seed=0)
    >>> next(iter_selector_aggregator_warnings(sel, [FedAvg(4)]))
    'RoundRobinClientSelection visits clients equally often; FedAvg then weights their updates by num_training_examples.'
    """
    seen: set[str] = set()
    key_sel = type(selector)
    for aggregator in aggregators:
        key = (key_sel, type(aggregator))
        message = SELECTOR_AGGREGATOR_WARNINGS.get(key)
        if message is None:
            continue
        when = SELECTOR_AGGREGATOR_WARNING_WHEN.get(key)
        if when is not None and not when(selector, aggregator):
            continue
        if message in seen:
            continue
        seen.add(message)
        yield message
