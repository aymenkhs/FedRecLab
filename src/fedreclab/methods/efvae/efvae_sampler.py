"""Federated Collaborative Importance Sampling (FCIS) proposal samplers.

Source: Zhang et al., "EFVAE: Efficient Federated Variational Autoencoder
for Collaborative Filtering", CIKM 2024, and the authors' FIMS-Uni
reference sampler.

The server rebuilds a product-quantisation codebook from the current
decoder item vectors each round (two half-dimension K-Means). Clients
sample negative item ids from the codebook without materialising a
full-catalogue softmax. This module vendors a seeded K-Means so the
library does not depend on ``fast_pytorch_kmeans``.
"""

from __future__ import annotations

from typing import Any

import torch


def construct_code_index(
    code_ids: torch.Tensor,
    num_cluster: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR-style ``(indices, indptr)`` over product codes.

    Parameters
    ----------
    code_ids : torch.Tensor
        Per-item product code in ``[0, num_cluster**2)``, shape ``(n_items,)``.
    num_cluster : int
        Number of clusters per subspace (``K``). There are ``K**2`` codes.

    Returns
    -------
    indices : torch.Tensor
        Item ids sorted by product code, shape ``(n_items,)``, int64.
    indptr : torch.Tensor
        Prefix counts over ``K**2 + 1`` slots, shape ``(K**2 + 1,)``, int64.

    Examples
    --------
    >>> codes = torch.tensor([1, 0, 1])
    >>> indices, indptr = construct_code_index(codes, num_cluster=2)
    >>> indptr.tolist()
    [0, 1, 3, 3, 3]
    """
    num_code = int(num_cluster) * int(num_cluster)
    sorted_codes, indices = torch.sort(code_ids.to(dtype=torch.long))
    cluster, count = torch.unique_consecutive(sorted_codes, return_counts=True)
    count_all = torch.zeros(num_code + 1, dtype=torch.long, device=code_ids.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr


def _kmeans_pp_init(
    x: torch.Tensor,
    num_clusters: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return ``num_clusters`` K-Means++ centres from rows of ``x``.

    Parameters
    ----------
    x : torch.Tensor
        Points, shape ``(n, d)``.
    num_clusters : int
        Number of centres. Must satisfy ``1 <= num_clusters <= n``.
    generator : torch.Generator or None
        CPU generator for sampling. ``None`` uses the default RNG.

    Returns
    -------
    torch.Tensor
        Centres, shape ``(num_clusters, d)``, same dtype/device as ``x``.
    """
    n = int(x.shape[0])
    device = x.device
    dtype = x.dtype
    centres = torch.empty((num_clusters, x.shape[1]), device=device, dtype=dtype)
    if generator is None:
        first = int(torch.randint(0, n, (1,), device=device).item())
    else:
        first = int(torch.randint(0, n, (1,), generator=generator).item())
    centres[0] = x[first]
    closest = torch.full((n,), float("inf"), device=device, dtype=dtype)
    for k in range(1, num_clusters):
        dist = torch.cdist(x, centres[k - 1 : k]).squeeze(-1)
        closest = torch.minimum(closest, dist)
        weights = closest.clamp(min=0.0).pow(2)
        total = float(weights.sum().item())
        if not (total > 0.0) or not torch.isfinite(weights).all():
            raise ValueError("Non-finite or zero weights in K-Means++ init")
        probs = weights / weights.sum()
        if generator is None:
            idx = int(torch.multinomial(probs, 1).item())
        else:
            idx = int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
        centres[k] = x[idx]
    return centres


def kmeans(
    x: torch.Tensor,
    num_clusters: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-6,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Lloyd K-Means with K-Means++ initialisation.

    Parameters
    ----------
    x : torch.Tensor
        Points, shape ``(n, d)``. ``n`` must be at least ``num_clusters``.
    num_clusters : int
        Number of clusters ``K``.
    max_iter : int, optional
        Maximum Lloyd iterations. Default is 300.
    tol : float, optional
        Relative centre movement stopping tolerance. Default is ``1e-6``.
    generator : torch.Generator or None, optional
        CPU generator for K-Means++. ``None`` uses the default RNG.

    Returns
    -------
    labels : torch.Tensor
        Cluster id per point, shape ``(n,)``, int64.
    centres : torch.Tensor
        Cluster centres, shape ``(K, d)``.

    Raises
    ------
    ValueError
        If ``num_clusters`` is out of range for ``n``, or a centre becomes
        non-finite.

    Examples
    --------
    >>> pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> labels, centres = kmeans(pts, 2, generator=torch.Generator().manual_seed(0))
    >>> labels.shape, centres.shape
    (torch.Size([4]), torch.Size([2, 2]))
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-d, got shape {tuple(x.shape)}")
    n = int(x.shape[0])
    k = int(num_clusters)
    if k < 1 or k > n:
        raise ValueError(f"num_clusters must be in [1, {n}], got {k}")
    centres = _kmeans_pp_init(x, k, generator)
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(int(max_iter)):
        dist = torch.cdist(x, centres)
        labels = torch.argmin(dist, dim=1)
        new_centres = centres.clone()
        for c in range(k):
            mask = labels == c
            if bool(mask.any()):
                new_centres[c] = x[mask].mean(dim=0)
            # empty cluster: keep previous centre
        if not torch.isfinite(new_centres).all():
            raise ValueError("Non-finite K-Means centres")
        shift = torch.linalg.norm(new_centres - centres)
        scale = torch.linalg.norm(centres).clamp(min=1e-12)
        centres = new_centres
        if float((shift / scale).item()) <= float(tol):
            break
    return labels, centres


class FIMSUniSampler:
    """FIMS-Uni proposal: product quantisation over decoder item vectors.

    Parameters
    ----------
    num_items : int
        Catalogue size after remapping.
    num_cluster : int, optional
        Clusters per subspace (``K``). Default is 16.
    num_neg : int, optional
        Negatives drawn per query. Default is 800.
    generator : torch.Generator or None, optional
        CPU generator for K-Means and multinomial draws. ``None`` uses the
        default RNG.

    Raises
    ------
    ValueError
        If sizes are not positive.

    Examples
    --------
    >>> s = FIMSUniSampler(num_items=8, num_cluster=2, num_neg=3)
    >>> s.update(torch.randn(8, 4), generator=torch.Generator().manual_seed(0))
    >>> pos_lp, neg, neg_lp = s(torch.randn(1, 4), torch.tensor([[0, 1]]))
    >>> neg.shape
    torch.Size([1, 3])
    """

    def __init__(
        self,
        num_items: int,
        num_cluster: int = 16,
        num_neg: int = 800,
        generator: torch.Generator | None = None,
    ):
        if not isinstance(num_items, int) or isinstance(num_items, bool) or num_items < 1:
            raise ValueError(f"num_items must be a positive int, got {num_items!r}")
        if not isinstance(num_cluster, int) or isinstance(num_cluster, bool) or num_cluster < 1:
            raise ValueError(f"num_cluster must be a positive int, got {num_cluster!r}")
        if not isinstance(num_neg, int) or isinstance(num_neg, bool) or num_neg < 1:
            raise ValueError(f"num_neg must be a positive int, got {num_neg!r}")
        self.num_items = int(num_items)
        self.num_cluster = int(num_cluster)
        self.num_code = self.num_cluster * self.num_cluster
        self.num_neg = int(num_neg)
        self.generator = generator
        self.i2cd0: torch.Tensor | None = None
        self.i2cd1: torch.Tensor | None = None
        self.i2cd01: torch.Tensor | None = None
        self.cb0: torch.Tensor | None = None
        self.cb1: torch.Tensor | None = None
        self.indices: torch.Tensor | None = None
        self.indptr: torch.Tensor | None = None
        self.cd2weights: torch.Tensor | None = None

    def update(
        self,
        item_embs: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        """Rebuild codebooks from decoder item vectors.

        Parameters
        ----------
        item_embs : torch.Tensor
            Decoder item table, shape ``(num_items, dim)``. ``dim`` must be
            even so the vector splits into two equal subspaces.
        generator : torch.Generator or None, optional
            Overrides the instance generator for this rebuild.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If shapes disagree, ``dim`` is odd, or clustering fails.
        """
        if item_embs.ndim != 2 or int(item_embs.shape[0]) != self.num_items:
            raise ValueError(
                f"item_embs must have shape ({self.num_items}, d), got {tuple(item_embs.shape)}"
            )
        dim = int(item_embs.shape[1])
        if dim % 2 != 0:
            raise ValueError(f"item embedding dim must be even for FIMS, got {dim}")
        # K cannot exceed the number of distinct half-vectors; clamp per subspace.
        k = min(self.num_cluster, self.num_items)
        if k < 1:
            raise ValueError("num_cluster clamps to an empty set")
        gen = self.generator if generator is None else generator
        embs1, embs2 = torch.chunk(item_embs.detach(), 2, dim=-1)
        self.i2cd0, self.cb0 = kmeans(embs1, k, generator=gen)
        self.i2cd1, self.cb1 = kmeans(embs2, k, generator=gen)
        # Remap to a square K_eff x K_eff product even when k < configured K.
        self._active_cluster = k
        self.i2cd01 = self.i2cd0 * k + self.i2cd1
        self.indices, self.indptr = construct_code_index(self.i2cd01, k)
        self.cd2weights = (self.indptr[1:] - self.indptr[:-1]).to(dtype=item_embs.dtype)

    @torch.no_grad()
    def __call__(
        self,
        query: torch.Tensor,
        pos_items: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Sample negatives for ``query`` under the current codebook.

        Parameters
        ----------
        query : torch.Tensor
            Decoder hidden states, shape ``(batch, dim)``.
        pos_items : torch.Tensor or None, optional
            Positive item ids, shape ``(batch, L)``. When given, proposal
            scores for positives are returned as the first output.

        Returns
        -------
        pos_log_probs : torch.Tensor or None
            Proposal scores for ``pos_items``, or ``None``.
        neg_items : torch.Tensor
            Sampled item ids, shape ``(batch, num_neg)``.
        neg_log_probs : torch.Tensor
            Proposal scores for ``neg_items``, same shape.

        Raises
        ------
        RuntimeError
            If ``update`` has not been called.
        ValueError
            If a sampling probability is non-finite.
        """
        self._require_ready()
        k = int(self._active_cluster)
        q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
        logit0 = q0 @ self.cb0.T
        logit1 = q1 @ self.cb1.T
        logit = (
            logit0.unsqueeze(1).permute(0, 2, 1).repeat(1, 1, k)
            + logit1.unsqueeze(1).repeat(1, k, 1)
        ).contiguous()
        logit = logit.view(int(query.shape[0]), -1)
        sample_pro = torch.softmax(logit, dim=-1) * self.cd2weights.to(
            device=logit.device, dtype=logit.dtype
        )
        if not torch.isfinite(sample_pro).all():
            raise ValueError("Non-finite FIMS sampling probabilities")
        # Renormalise: empty codes are zero weight.
        denom = sample_pro.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        sample_pro = sample_pro / denom
        sample_code = self._multinomial(sample_pro, self.num_neg)
        sample_logit = torch.gather(logit, -1, sample_code)
        neg_items, neg_log_probs = self._sample_item(sample_code, sample_logit)
        pos_log_probs = None
        if pos_items is not None:
            pos_log_probs = self._compute_item_p(query, pos_items)
        return pos_log_probs, neg_items, neg_log_probs

    def state_dict(self) -> dict[str, Any]:
        """Return CPU clones of the codebook tensors for payload shipping.

        Returns
        -------
        dict
            Keys consumed by ``load_state_dict``.

        Raises
        ------
        RuntimeError
            If ``update`` has not been called.
        """
        self._require_ready()
        return {
            "num_items": self.num_items,
            "num_cluster": self.num_cluster,
            "num_neg": self.num_neg,
            "active_cluster": int(self._active_cluster),
            "i2cd0": self.i2cd0.detach().cpu().clone(),
            "i2cd1": self.i2cd1.detach().cpu().clone(),
            "i2cd01": self.i2cd01.detach().cpu().clone(),
            "cb0": self.cb0.detach().cpu().clone(),
            "cb1": self.cb1.detach().cpu().clone(),
            "indices": self.indices.detach().cpu().clone(),
            "indptr": self.indptr.detach().cpu().clone(),
            "cd2weights": self.cd2weights.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, Any], device: Any = None) -> None:
        """Restore codebook tensors from ``state_dict``.

        Parameters
        ----------
        state : dict
            Output of ``state_dict``.
        device : torch.device or str or None, optional
            Device for tensors. ``None`` keeps CPU.

        Returns
        -------
        None
        """
        self.num_items = int(state["num_items"])
        self.num_cluster = int(state["num_cluster"])
        self.num_neg = int(state["num_neg"])
        self._active_cluster = int(state["active_cluster"])
        self.num_code = self.num_cluster * self.num_cluster

        def _to(t):
            tensor = torch.as_tensor(t)
            return tensor if device is None else tensor.to(device)

        self.i2cd0 = _to(state["i2cd0"])
        self.i2cd1 = _to(state["i2cd1"])
        self.i2cd01 = _to(state["i2cd01"])
        self.cb0 = _to(state["cb0"])
        self.cb1 = _to(state["cb1"])
        self.indices = _to(state["indices"])
        self.indptr = _to(state["indptr"])
        self.cd2weights = _to(state["cd2weights"])

    def _sample_item(
        self,
        code: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one item uniformly from each selected product code cell."""
        item_cnt = self.indptr[code + 1] - self.indptr[code]
        if bool((item_cnt <= 0).any()):
            raise ValueError("FIMS sampled an empty product code")
        if self.generator is None:
            u = torch.rand(item_cnt.shape, dtype=torch.float32, device=item_cnt.device)
        else:
            u = torch.rand(
                item_cnt.shape, dtype=torch.float32, generator=self.generator
            ).to(device=item_cnt.device)
        item_idx = torch.floor(item_cnt.to(dtype=torch.float32) * u).long()
        neg_items = self.indices[item_idx + self.indptr[code]]
        return neg_items, log_prob

    def _compute_item_p(self, query: torch.Tensor, pos_items: torch.Tensor) -> torch.Tensor:
        """Proposal score for ``pos_items`` under the product codebook."""
        k = int(self._active_cluster)
        cd0 = self.i2cd0[pos_items]
        cd1 = self.i2cd1[pos_items]
        c0 = self.cb0[cd0, :]
        c1 = self.cb1[cd1, :]
        q0, q1 = query.chunk(2, dim=-1)
        if query.dim() == pos_items.dim():
            r = (
                torch.bmm(c0, q0.unsqueeze(-1)) + torch.bmm(c1, q1.unsqueeze(-1))
            ).squeeze(-1)
        else:
            r = torch.sum(c0 * q0.unsqueeze(1), dim=-1) + torch.sum(
                c1 * q1.unsqueeze(1), dim=-1
            )
        del k
        return r

    def _multinomial(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        if self.generator is None:
            return torch.multinomial(probs, num_samples, replacement=True)
        # multinomial with generator requires CPU probs in some builds
        cpu_probs = probs.detach().cpu()
        draws = torch.multinomial(
            cpu_probs, num_samples, replacement=True, generator=self.generator
        )
        return draws.to(device=probs.device)

    def _require_ready(self) -> None:
        if self.cb0 is None or self.indptr is None:
            raise RuntimeError("FIMSUniSampler.update must run before sampling")
