"""Composable client-side privacy mechanisms.

Two mechanisms cover the uploads used by federated recommenders that
hide interaction sets and raw updates:

- :class:`PseudoItemSampling` draws item ids the user has not interacted
  with and synthesises a matching gradient for each one.
- :class:`ClipLaplaceLDP` clips a collection of arrays by one shared L1
  budget and adds isotropic Laplace noise.

Both operate on arrays the caller already computed. They do not read
the dataset and do not apply the server update.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class PseudoItemSampling:
    """Sample pseudo-interacted items and gradients matched to real ones.

    Pseudo items are catalogue ids outside an exclusion set (the user's
    real interactions, plus any id already present in the upload). Each
    pseudo gradient is drawn from a diagonal Gaussian whose per-coordinate
    mean and standard deviation equal those of the real item-embedding
    gradients. A full covariance is not used: it is singular whenever
    there are fewer real items than the embedding width.

    One real gradient has standard deviation 0, so every pseudo gradient
    equals that vector. Laplace noise, when composed afterwards, is what
    then separates the rows.

    Parameters
    ----------
    count : int
        Number of pseudo items to draw. ``0`` disables the mechanism.

    Raises
    ------
    ValueError
        If ``count`` is negative or not an integer.

    Examples
    --------
    >>> sampler = PseudoItemSampling(2)
    >>> grads = sampler.matched_gradients(
    ...     np.array([[1.0, -1.0]]), np.random.default_rng(0))
    >>> np.allclose(grads, [[1.0, -1.0]])
    True
    """

    def __init__(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"count must be a non-negative integer, got {count!r}"
            )
        self.count = count

    def sample_ids(
        self,
        num_items: int,
        excluded: Iterable[int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``count`` item ids from outside ``excluded``.

        Parameters
        ----------
        num_items : int
            Catalogue size. Valid ids are ``0 .. num_items - 1``.
        excluded : iterable of int
            Ids that must not be sampled. Duplicates and ids outside
            the catalogue are ignored.
        rng : numpy.random.Generator
            Source of the draw. Unused when ``count`` is 0.

        Returns
        -------
        ndarray of int64
            Shape ``(count,)``. Empty when ``count`` is 0.

        Raises
        ------
        ValueError
            If ``num_items`` is negative, or fewer than ``count`` ids
            remain after exclusion.

        Examples
        --------
        >>> ids = PseudoItemSampling(2).sample_ids(
        ...     5, excluded=[1], rng=np.random.default_rng(0))
        >>> len(ids) == 2 and set(ids.tolist()).isdisjoint({1})
        True
        """
        if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 0:
            raise ValueError(
                f"num_items must be a non-negative integer, got {num_items!r}"
            )
        if self.count == 0:
            return np.zeros((0,), dtype=np.int64)
        excluded_arr = np.unique(np.asarray(list(excluded), dtype=np.int64))
        pool = np.setdiff1d(
            np.arange(num_items, dtype=np.int64),
            excluded_arr,
            assume_unique=False,
        )
        if self.count > int(pool.size):
            raise ValueError(
                f"Need {self.count} pseudo items but only {int(pool.size)} "
                f"ids remain outside the exclusion set of a catalogue of "
                f"{num_items}."
            )
        picked = rng.choice(pool, size=self.count, replace=False)
        return np.asarray(picked, dtype=np.int64)

    def matched_gradients(
        self,
        real_rows: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``count`` rows from the empirical diagonal Gaussian.

        Parameters
        ----------
        real_rows : ndarray
            Real item gradients, shape ``(n, d)`` with ``n >= 1``.
        rng : numpy.random.Generator
            Source of the normal draws. Unused when ``count`` is 0.

        Returns
        -------
        ndarray of float32
            Shape ``(count, d)``.

        Raises
        ------
        ValueError
            If ``real_rows`` is not a finite 2-d array with at least
            one row.

        Examples
        --------
        >>> real = np.array([[0.0, 2.0], [0.0, 2.0]])
        >>> out = PseudoItemSampling(1).matched_gradients(
        ...     real, np.random.default_rng(0))
        >>> np.allclose(out, [[0.0, 2.0]])
        True
        """
        real = np.asarray(real_rows, dtype=np.float64)
        if real.ndim != 2 or real.shape[0] == 0:
            raise ValueError(
                "real_rows must have shape (n, d) with n >= 1, "
                f"got {getattr(real_rows, 'shape', None)}"
            )
        if not np.isfinite(real).all():
            raise ValueError("real_rows contains non-finite values")
        width = int(real.shape[1])
        if self.count == 0:
            return np.zeros((0, width), dtype=np.float32)
        mean = real.mean(axis=0)
        std = real.std(axis=0, ddof=0)
        draws = rng.normal(loc=mean, scale=std, size=(self.count, width))
        if not np.isfinite(draws).all():
            raise ValueError("Matched Gaussian draw contains non-finite values")
        return draws.astype(np.float32)


class ClipLaplaceLDP:
    """Clip arrays by one shared L1 budget, then add Laplace noise.

    Given arrays ``g_1, ..., g_k`` viewed as one concatenated vector
    ``g``,

        g <- g * min(1, clip / ||g||_1) + Laplace(0, scale)

    when ``clip`` is set. The same factor multiplies every array, which
    is the L1 clip used on a unified gradient. Noise is then independent
    on each element. ``clip is None`` skips clipping. ``scale == 0``
    skips noise. Both together leave the arrays unchanged (copies are
    still returned when either part is enabled).

    Parameters
    ----------
    clip : float or None, optional
        Positive L1 threshold ``δ``. ``None`` (default) disables clipping.
    scale : float, optional
        Laplace scale ``λ``. ``0`` (default) disables noise.

    Raises
    ------
    ValueError
        If ``clip`` is not a positive finite number, or ``scale`` is
        negative or non-finite.

    Examples
    --------
    >>> mech = ClipLaplaceLDP(clip=1.0, scale=0.0)
    >>> out = mech.apply(
    ...     [np.array([3.0, 0.0], dtype=np.float32)],
    ...     np.random.default_rng(0),
    ... )
    >>> np.allclose(out[0], [1.0, 0.0])
    True
    """

    def __init__(self, clip: float | None = None, scale: float = 0.0) -> None:
        if isinstance(clip, bool) or (
            clip is not None and (not np.isfinite(float(clip)) or float(clip) <= 0.0)
        ):
            raise ValueError(
                f"clip must be a positive finite number or None, got {clip!r}"
            )
        if clip is not None:
            clip = float(clip)
        if isinstance(scale, bool) or not np.isfinite(float(scale)) or float(scale) < 0.0:
            raise ValueError(
                f"scale must be a finite number >= 0, got {scale!r}"
            )
        scale_value = float(scale)
        self.clip = clip
        self.scale = scale_value

    @property
    def enabled(self) -> bool:
        """Whether clipping or noise is configured.

        Returns
        -------
        bool
            True when ``clip`` is set or ``scale`` is positive.

        Examples
        --------
        >>> ClipLaplaceLDP().enabled
        False
        >>> ClipLaplaceLDP(scale=0.1).enabled
        True
        """
        return self.clip is not None or self.scale > 0.0

    def apply(
        self,
        arrays: Sequence[np.ndarray],
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Return clipped and noised copies of ``arrays``.

        Parameters
        ----------
        arrays : sequence of ndarray
            Gradients or deltas. Each array is clipped as part of one
            concatenated vector and noised elementwise. Shapes are
            preserved. Empty arrays contribute nothing to the L1 norm.
        rng : numpy.random.Generator
            Source of the Laplace draws. Unused when ``scale`` is 0.
            Draws follow the order of ``arrays``, in C order inside
            each array.

        Returns
        -------
        list of ndarray
            One float32 array per input, same shapes.

        Raises
        ------
        ValueError
            If an input or the noised result contains a non-finite value.

        Examples
        --------
        >>> mech = ClipLaplaceLDP(clip=1.0, scale=0.0)
        >>> parts = mech.apply(
        ...     [np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)],
        ...     np.random.default_rng(0),
        ... )
        >>> float(sum(np.abs(p).sum() for p in parts))
        1.0
        """
        copied = [np.array(array, dtype=np.float32, copy=True) for array in arrays]
        for array in copied:
            if not np.isfinite(array).all():
                raise ValueError("ClipLaplaceLDP received non-finite values")
        if self.clip is not None:
            total = float(sum(np.abs(array, dtype=np.float64).sum() for array in copied))
            if total > self.clip:
                factor = np.float32(self.clip / total)
                for array in copied:
                    array *= factor
        if self.scale > 0.0:
            for array in copied:
                if array.size == 0:
                    continue
                noise = rng.laplace(0.0, self.scale, size=array.shape).astype(np.float32)
                array += noise
                if not np.isfinite(array).all():
                    raise ValueError("Laplace noise produced non-finite values")
        return copied
