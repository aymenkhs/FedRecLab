from typing import List, Dict, Any, Iterable, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn


def module_tensor_names(module: nn.Module) -> set[str]:
    """Return the fully qualified names of a module's parameters and buffers.

    Parameters
    ----------
    module : torch.nn.Module
        Module to inspect (the shared client skeleton, typically).

    Returns
    -------
    set of str
        Union of ``named_parameters()`` and ``named_buffers()`` keys,
        e.g. ``{"item_embeddings.weight", "mlp.0.bias"}``.

    Examples
    --------
    >>> m = nn.Linear(4, 2)
    >>> sorted(module_tensor_names(m))
    ['bias', 'weight']
    """
    names = {n for n, _ in module.named_parameters()}
    names.update(n for n, _ in module.named_buffers())
    return names


def flatten_grouped_state(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten ParamIO groups ``{group: {name: tensor}}`` into ``{name: tensor}``.

    FedNCF stores server weights in three groups (item embeddings, shared
    layers, output weights). Loaders and coverage checks need a single
    name-to-tensor map.

    Parameters
    ----------
    groups : mapping of str to mapping of str to any
        Outer keys are ParamIO group names. Inner keys are parameter names
        and must be unique across groups.

    Returns
    -------
    dict of str to any
        One entry per inner name, values unchanged.

    Raises
    ------
    KeyError
        If the same parameter name appears in more than one group.

    Examples
    --------
    >>> flatten_grouped_state({
    ...     "item_embeddings": {"gmf_item_embeddings.weight": 1},
    ...     "output_weights": {"output_logits.bias": 2},
    ... })
    {'gmf_item_embeddings.weight': 1, 'output_logits.bias': 2}
    """
    flat: dict[str, Any] = {}
    for group_name, group in groups.items():
        for name, value in group.items():
            if name in flat:
                raise KeyError(
                    f"Duplicate parameter name '{name}' in group '{group_name}'"
                )
            flat[name] = value
    return flat


def assert_skeleton_server_coverage(
    skeleton: nn.Module,
    server_state_names: Iterable[str],
) -> None:
    """Check that every skeleton tensor is named in the server state, and vice versa.

    A name present on the module but missing from the server payload would
    keep the previous client's value when the skeleton is reused. A name in
    the payload but not on the module is dropped on load.

    Parameters
    ----------
    skeleton : torch.nn.Module
        Server-only module built by ``init_model_skeleton``.
    server_state_names : iterable of str
        Names returned by ``get_server_state`` / ``get_server_state_groups``
        (flattened).

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the two name sets differ. The message lists names missing from
        the server state and names extra in the server state.

    Examples
    --------
    >>> m = nn.Linear(2, 1)
    >>> assert_skeleton_server_coverage(m, {"weight", "bias"})
    >>> assert_skeleton_server_coverage(m, {"weight"})
    Traceback (most recent call last):
        ...
    RuntimeError: Skeleton coverage mismatch: ...
    """
    skeleton_names = module_tensor_names(skeleton)
    server_names = set(server_state_names)
    if skeleton_names != server_names:
        missing = sorted(skeleton_names - server_names)
        extra = sorted(server_names - skeleton_names)
        raise RuntimeError(
            "Skeleton coverage mismatch: every skeleton parameter and buffer "
            f"must be named in the server state. missing_from_server={missing} "
            f"extra_in_server={extra}"
        )


def assert_client_names_disjoint(
    server_state_names: Iterable[str],
    client_names: Iterable[str],
) -> None:
    """Check that injected client tensors do not reuse server-state names.

    User embeddings are passed in as extra ``nn.Parameter``s for one
    ``local_train`` / ``compute_scores`` call. They must not collide with
    names already on the skeleton.

    Parameters
    ----------
    server_state_names : iterable of str
        Names of tensors on the skeleton / in the server payload.
    client_names : iterable of str
        Names used for the injected client tensors, e.g.
        ``{"gmf_user_embeddings.weight"}``.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the two sets overlap. The message lists the colliding names.

    Examples
    --------
    >>> assert_client_names_disjoint({"item_embeddings.weight"}, {"user_embedding.weight"})
    >>> assert_client_names_disjoint({"user_embedding.weight"}, {"user_embedding.weight"})
    Traceback (most recent call last):
        ...
    RuntimeError: Injected client tensor names must be disjoint ...
    """
    overlap = set(server_state_names) & set(client_names)
    if overlap:
        raise RuntimeError(
            "Injected client tensor names must be disjoint from the skeleton / "
            f"server state. overlap={sorted(overlap)}"
        )


def extract_named_optimizer_state(
    optimizer: torch.optim.Optimizer,
    named_params: Mapping[str, nn.Parameter],
    names_to_keep: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Snapshot optimiser moments keyed by parameter name, not by index.

    ``optimizer.state_dict()`` identifies parameters by registration order.
    Filtering that dict by index breaks if a layer is added or reordered, and
    ``load_state_dict`` also restores stale ``param_groups`` (lr, betas).
    This helper walks ``optimizer.state`` via parameter identity and stores
    ``{name: {exp_avg, exp_avg_sq, step, ...}}``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser after ``step()``. Parameters that never received a gradient
        have empty state and are omitted.
    named_params : mapping of str to torch.nn.Parameter
        Assembled name → parameter map (skeleton parameters plus injected
        client tensors).
    names_to_keep : iterable of str or None, optional
        If given, only those names are copied (``client_resident`` scope).
        ``None`` keeps every assembled name that has state (``all`` scope).

    Returns
    -------
    dict of str to dict
        CPU clones of the per-parameter state. Missing names are absent;
        the next round starts those parameters with empty moments.

    Examples
    --------
    >>> layer = nn.Linear(2, 1)
    >>> opt = torch.optim.Adam(layer.parameters(), lr=0.01)
    >>> layer(torch.ones(1, 2)).sum().backward()
    >>> opt.step()
    >>> named = dict(layer.named_parameters())
    >>> saved = extract_named_optimizer_state(opt, named, names_to_keep=["weight"])
    >>> list(saved)
    ['weight']
    """
    keep = None if names_to_keep is None else set(names_to_keep)
    id_to_name = {id(param): name for name, param in named_params.items()}
    out: dict[str, dict[str, Any]] = {}
    for param, state in optimizer.state.items():
        name = id_to_name.get(id(param))
        if name is None:
            continue
        if keep is not None and name not in keep:
            continue
        if not state:
            continue
        cloned: dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().cpu().clone()
            else:
                cloned[key] = value
        out[name] = cloned
    return out


def restore_named_optimizer_state(
    optimizer: torch.optim.Optimizer,
    named_params: Mapping[str, nn.Parameter],
    saved: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Copy name-keyed moments into a freshly constructed optimiser.

    Does not call ``optimizer.load_state_dict``: ``param_groups`` (lr, betas)
    stay those of the current constructor. Parameters with no stored entry
    keep empty state and start fresh.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        New optimiser over the assembled parameters for this round.
    named_params : mapping of str to torch.nn.Parameter
        Same name → parameter map used when training this round.
    saved : mapping of str to mapping, or None
        Output of :func:`extract_named_optimizer_state`. ``None`` or ``{}``
        is a no-op (first round, or ``optimizer_state_scope="none"``).

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If ``saved`` contains a name that is not in ``named_params``.

    Examples
    --------
    >>> layer = nn.Linear(2, 1)
    >>> opt = torch.optim.Adam(layer.parameters(), lr=0.01)
    >>> restore_named_optimizer_state(opt, dict(layer.named_parameters()), None)
    >>> restore_named_optimizer_state(
    ...     opt, dict(layer.named_parameters()),
    ...     {"not_a_parameter": {"exp_avg": torch.zeros(1)}},
    ... )
    Traceback (most recent call last):
        ...
    KeyError: Stored optimiser state for 'not_a_parameter' ...
    """
    if not saved:
        return
    for name, state in saved.items():
        if name not in named_params:
            raise KeyError(
                f"Stored optimiser state for '{name}' matches no parameter "
                f"in the assembled set {sorted(named_params)}"
            )
        param = named_params[name]
        restored: dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    restored[key] = value.to(device=param.device, dtype=param.dtype)
                else:
                    restored[key] = value.to(device=param.device)
            else:
                restored[key] = value
        optimizer.state[param] = restored


@dataclass
class RoundStatistics:
    current_round: int
    selected: List[int]
    responded: List[int]
    dropped: List[int]
    num_training_examples: Dict[int, int]
    training_losses: Dict[int, List[float]]
    training_durations: Dict[int, float]
    # param_name -> number of rows that fell back from Krum/Bulyan to averaging
    row_aggregator_fallbacks: Dict[str, int] = field(default_factory=dict)


_VALID_LAYOUTS = ("dense", "rows")


class ParamIO:
    def __init__(self, residency, exchange, apply_to, layout="dense"):
        if layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"ParamIO.layout must be one of {_VALID_LAYOUTS}, got {layout!r}"
            )
        self.residency = residency
        self.exchange = exchange
        self.apply_to = apply_to
        self.layout = layout


def is_row_update(value: Any) -> bool:
    """True if ``value`` is a sparse row upload ``{"rows", "values"}``."""
    return isinstance(value, dict) and "rows" in value and "values" in value


@dataclass
class UpdatePackage:
    param_name: str
    param_io: ParamIO
    param_value: Any
    param_metadata: dict
