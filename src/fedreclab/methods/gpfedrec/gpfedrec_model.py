"""GPFedRec model: per-client score function over injected embedding tables.

Source: Zhang, Long, Zhou, Zhang, Yan and Yang, "GPFedRec: Graph-Guided
Personalization for Federated Recommendation", KDD 2024.
https://doi.org/10.1145/3637528.3671702

Every part of the GPFedRec model is held per client: one user vector per
local user, a full item table and a personal MLP score function over
``concat(user, item)``. The reference ``mlp.py`` uses the tower
``64 -> 32 -> 16 -> 8`` with ReLU and a ``Linear(8, 1)`` output followed
by a sigmoid. Here the module returns the logit; BCE-with-logits on the
client is equivalent to the sigmoid plus BCE, and ranking order is
unchanged by the sigmoid.

The user and item tables are not stored on the module. They are passed
in as tensors so one skeleton can serve every client.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class GPFedRecScoreFunction(nn.Module):
    """Personal MLP that maps ``concat(user, item)`` to a logit.

    Parameters
    ----------
    latent_dim : int, optional
        Width of the user and item vectors. Default is 32.
    hidden_layers : sequence of int, optional
        Widths of the MLP tower, starting with its input width. The first
        entry must equal ``2 * latent_dim``. Default is
        ``(64, 32, 16, 8)`` (reference ``train.py``).

    Raises
    ------
    ValueError
        If ``hidden_layers`` has fewer than one entry or its first entry
        is not ``2 * latent_dim``.

    Examples
    --------
    >>> f = GPFedRecScoreFunction(latent_dim=2, hidden_layers=(4, 3))
    >>> f(torch.zeros(5, 2), torch.zeros(5, 2)).shape
    torch.Size([5])
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_layers: Sequence[int] = (64, 32, 16, 8),
    ):
        super().__init__()
        hidden_layers = tuple(int(h) for h in hidden_layers)
        if len(hidden_layers) < 1:
            raise ValueError("hidden_layers must contain at least the input width")
        if hidden_layers[0] != 2 * int(latent_dim):
            raise ValueError(
                f"hidden_layers[0] must equal 2 * latent_dim = {2 * int(latent_dim)} "
                f"(user and item vectors are concatenated), got {hidden_layers[0]}"
            )
        self.latent_dim = int(latent_dim)
        self.hidden_layers = hidden_layers
        self.fc_layers = nn.ModuleList(
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(hidden_layers[:-1], hidden_layers[1:])
        )
        self.affine_output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_vectors: torch.Tensor, item_vectors: torch.Tensor) -> torch.Tensor:
        """Return one logit per (user, item) pair.

        Parameters
        ----------
        user_vectors : torch.Tensor
            Shape ``(batch, latent_dim)``.
        item_vectors : torch.Tensor
            Shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch,)``.
        """
        vector = torch.cat([user_vectors, item_vectors], dim=-1)
        for layer in self.fc_layers:
            vector = F.relu(layer(vector))
        return self.affine_output(vector).view(-1)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Return CPU clones of every parameter, keyed by its qualified name.

        Returns
        -------
        dict of str to torch.Tensor
            ``named_parameters()`` names mapped to detached CPU copies.

        Examples
        --------
        >>> f = GPFedRecScoreFunction(latent_dim=2, hidden_layers=(4,))
        >>> sorted(f.get_state())
        ['affine_output.bias', 'affine_output.weight']
        """
        return {
            name: param.detach().cpu().clone()
            for name, param in self.named_parameters()
        }

    def load_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Copy ``state`` into this module's parameters in place.

        Parameters
        ----------
        state : mapping of str to torch.Tensor
            Output of :meth:`get_state`, possibly from another module with
            the same architecture.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the names in ``state`` differ from this module's parameters.

        Examples
        --------
        >>> a = GPFedRecScoreFunction(latent_dim=2, hidden_layers=(4,))
        >>> b = GPFedRecScoreFunction(latent_dim=2, hidden_layers=(4,))
        >>> b.load_state(a.get_state())
        >>> torch.equal(a.affine_output.bias, b.affine_output.bias)
        True
        """
        named = dict(self.named_parameters())
        missing = set(named) - set(state)
        extra = set(state) - set(named)
        if missing or extra:
            raise KeyError(
                "Score-function state does not match the module: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(torch.as_tensor(state[name], device=param.device))


def predict_logits(
    score_function: GPFedRecScoreFunction,
    local_user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    user_table: torch.Tensor,
    item_table: torch.Tensor,
) -> torch.Tensor:
    """Look up user and item vectors and score them with ``score_function``.

    Parameters
    ----------
    score_function : GPFedRecScoreFunction
        The client's MLP, already loaded with its weights.
    local_user_ids : torch.Tensor
        Row indices into ``user_table``, shape ``(batch,)``.
    item_ids : torch.Tensor
        Row indices into ``item_table``, shape ``(batch,)``.
    user_table : torch.Tensor
        Client user vectors, shape ``(n_local_users, latent_dim)``.
    item_table : torch.Tensor
        Client item table, shape ``(num_items, latent_dim)``.

    Returns
    -------
    torch.Tensor
        Logits of shape ``(batch,)``. Gradients flow into both tables.

    Examples
    --------
    >>> f = GPFedRecScoreFunction(latent_dim=2, hidden_layers=(4,))
    >>> predict_logits(f, torch.tensor([0]), torch.tensor([1]),
    ...                torch.zeros(1, 2), torch.zeros(3, 2)).shape
    torch.Size([1])
    """
    return score_function(
        F.embedding(local_user_ids, user_table),
        F.embedding(item_ids, item_table),
    )


def build_client_template(
    latent_dim: int,
    hidden_layers: Sequence[int],
    init_seed: int,
) -> tuple[torch.Tensor, GPFedRecScoreFunction]:
    """Build the initial user vector and score function shared by all clients.

    The reference copies one freshly initialised model into every client,
    so every client starts from the same MLP and user vector. Values use
    PyTorch's default initialisation (``nn.Linear`` Kaiming-uniform, user
    vector ``N(0, 1)`` as in ``nn.Embedding``), drawn on CPU from
    ``init_seed + 1``. The offset keeps this stream apart from the server's
    item-table generator, which is seeded with the server seed itself.

    Only the CPU default generator is reseeded, inside
    ``torch.random.fork_rng(devices=[])``, so both the CPU and the CUDA
    global RNG states are unchanged afterwards. ``torch.manual_seed`` would
    also reseed every CUDA generator, which ``fork_rng(devices=[])`` does
    not restore.

    Parameters
    ----------
    latent_dim : int
        Width of the user vector.
    hidden_layers : sequence of int
        MLP widths, see :class:`GPFedRecScoreFunction`.
    init_seed : int
        Seed for the template. The stream used is ``init_seed + 1``.

    Returns
    -------
    user_vector : torch.Tensor
        CPU tensor of shape ``(latent_dim,)``.
    score_function : GPFedRecScoreFunction
        CPU module holding the template weights.

    Examples
    --------
    >>> u1, f1 = build_client_template(2, (4,), init_seed=0)
    >>> u2, f2 = build_client_template(2, (4,), init_seed=0)
    >>> torch.equal(u1, u2)
    True
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(init_seed) + 1)
        score_function = GPFedRecScoreFunction(latent_dim, hidden_layers)
        user_vector = torch.randn(int(latent_dim))
    return user_vector, score_function


def initial_item_table(
    num_items: int,
    latent_dim: int,
    std: float,
    seed: int,
) -> torch.Tensor:
    """Draw the initial shared item table from ``N(0, std)``.

    Uses a private ``torch.Generator`` seeded with ``seed``; the global RNG
    is not touched. ``std=1.0`` matches the ``nn.Embedding`` default used
    by the reference.

    Parameters
    ----------
    num_items : int
        Number of item rows.
    latent_dim : int
        Width of each row.
    std : float
        Standard deviation of the normal initialisation.
    seed : int
        Seed of the private generator.

    Returns
    -------
    torch.Tensor
        CPU float32 tensor of shape ``(num_items, latent_dim)``.

    Examples
    --------
    >>> initial_item_table(3, 2, 1.0, seed=0).shape
    torch.Size([3, 2])
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    table = torch.empty(int(num_items), int(latent_dim), dtype=torch.float32)
    table.normal_(0.0, float(std), generator=generator)
    return table
