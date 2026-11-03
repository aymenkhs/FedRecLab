"""Clip-and-Laplace LDP and pseudo-item sampling."""

import numpy as np
import pytest

from src.fedreclab.federated.mechanisms import ClipLaplaceLDP, PseudoItemSampling


def test_pseudo_ids_avoid_the_exclusion_set_and_are_reproducible():
    sampler = PseudoItemSampling(3)
    first = sampler.sample_ids(10, excluded=[0, 1, 1], rng=np.random.default_rng(0))
    second = sampler.sample_ids(10, excluded=[0, 1], rng=np.random.default_rng(0))
    assert first.shape == (3,)
    assert first.dtype == np.int64
    assert set(first.tolist()).isdisjoint({0, 1})
    np.testing.assert_array_equal(first, second)


def test_pseudo_sampling_rejects_a_pool_that_is_too_small():
    with pytest.raises(ValueError, match="only 1"):
        PseudoItemSampling(2).sample_ids(3, excluded=[0, 2], rng=np.random.default_rng(0))


def test_zero_pseudo_items_does_not_draw():
    ids = PseudoItemSampling(0).sample_ids(4, excluded=[], rng=np.random.default_rng(0))
    grads = PseudoItemSampling(0).matched_gradients(
        np.ones((2, 3)), np.random.default_rng(0)
    )
    assert ids.shape == (0,)
    assert grads.shape == (0, 3)


def test_matched_gradients_copy_a_single_real_row():
    real = np.array([[0.25, -0.5, 1.0]])
    out = PseudoItemSampling(4).matched_gradients(real, np.random.default_rng(1))
    np.testing.assert_allclose(out, np.repeat(real, 4, axis=0))


def test_matched_gradients_follow_the_empirical_mean_and_std():
    real = np.array([[0.0, 0.0], [2.0, 4.0]])
    out = PseudoItemSampling(20_000).matched_gradients(real, np.random.default_rng(0))
    np.testing.assert_allclose(out.mean(axis=0), real.mean(axis=0), atol=0.05)
    np.testing.assert_allclose(out.std(axis=0), real.std(axis=0), atol=0.05)


def test_matched_gradients_reject_non_finite_rows():
    with pytest.raises(ValueError, match="non-finite"):
        PseudoItemSampling(1).matched_gradients(
            np.array([[np.nan, 0.0]]), np.random.default_rng(0)
        )


def test_l1_clip_is_shared_across_arrays_and_does_not_mutate_inputs():
    source = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([1.0, 0.0], dtype=np.float32)
    out = ClipLaplaceLDP(clip=1.0, scale=0.0).apply(
        [source, other], np.random.default_rng(0)
    )
    np.testing.assert_allclose(source, [1.0, 0.0])
    total = float(sum(np.abs(part).sum() for part in out))
    assert total == pytest.approx(1.0)
    np.testing.assert_allclose(out[0], out[1])


def test_laplace_noise_is_reproducible_and_disabled_at_scale_zero():
    source = np.zeros(5, dtype=np.float32)
    noisy = ClipLaplaceLDP(clip=None, scale=0.3)
    first = noisy.apply([source], np.random.default_rng(2))[0]
    second = noisy.apply([source], np.random.default_rng(2))[0]
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, 0.0)
    unchanged = ClipLaplaceLDP(clip=None, scale=0.0).apply(
        [source], np.random.default_rng(2)
    )[0]
    np.testing.assert_array_equal(unchanged, source)


def test_clip_laplace_rejects_non_finite_input():
    with pytest.raises(ValueError, match="non-finite"):
        ClipLaplaceLDP(clip=1.0).apply(
            [np.array([np.inf], dtype=np.float32)], np.random.default_rng(0)
        )


def test_mechanism_constructors_reject_bad_settings():
    with pytest.raises(ValueError, match="count"):
        PseudoItemSampling(-1)
    with pytest.raises(ValueError, match="count"):
        PseudoItemSampling(True)
    with pytest.raises(ValueError, match="clip"):
        ClipLaplaceLDP(clip=0.0)
    with pytest.raises(ValueError, match="clip"):
        ClipLaplaceLDP(clip=True)
    with pytest.raises(ValueError, match="scale"):
        ClipLaplaceLDP(scale=-0.1)
