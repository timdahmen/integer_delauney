"""Tests for Delaunay.reset(): restore just-constructed state without
freeing/reallocating device buffers, so a caller can reuse one object across
many same-canvas seed sets instead of constructing a fresh one each time.

Correctness strategy
---------------------
The bar is byte-for-byte agreement between "reset(), then insert set B" and
"a freshly constructed Delaunay, insert set B" -- reset must leave no trace
of what came before. Separately, the generation counter (already exercised
for insert/finalise/finalise_device in test_finalise_device.py) must treat
reset() as a mutation too: a view taken before a reset must raise, the same
way it does after any other mutating call.
"""
import numpy as np
import pytest


def _cuda_available() -> bool:
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="_delauney_cuda extension not built or no CUDA device",
)

if _cuda_available():
    from delauney._delauney_cuda import Delaunay

W, H, MAX_SEEDS = 64, 48, 512


def _random_seeds(rng, n, w=W, h=H):
    xs = rng.integers(0, w, size=n).astype(np.int32)
    ys = rng.integers(0, h, size=n).astype(np.int32)
    xy = np.unique(np.stack([xs, ys], axis=1), axis=0)
    return [(int(x), int(y)) for x, y in xy]


def _finalise_grid(d, seeds):
    d.insert_deferred(seeds, None)
    verts, tgrid = d.finalise(as_arrays=True)
    return np.asarray(verts), np.asarray(tgrid)


@pytest.mark.parametrize("n_seeds", [10, 40, 150])
def test_reset_then_insert_matches_fresh_construction(n_seeds):
    rng = np.random.default_rng(n_seeds)
    seeds = _random_seeds(rng, n_seeds)

    reused = Delaunay(W, H, MAX_SEEDS, -1)
    # Seed it with something else first, so reset has actual state to discard.
    reused.insert_deferred(_random_seeds(rng, 20), None)
    reused.finalise(False)
    reused.reset()
    verts_reused, tgrid_reused = _finalise_grid(reused, seeds)

    fresh = Delaunay(W, H, MAX_SEEDS, -1)
    verts_fresh, tgrid_fresh = _finalise_grid(fresh, seeds)

    np.testing.assert_array_equal(verts_reused, verts_fresh)
    np.testing.assert_array_equal(tgrid_reused, tgrid_fresh)


def test_reset_clears_seed_count():
    rng = np.random.default_rng(0)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(_random_seeds(rng, 30), None)
    assert d.seed_count > 0
    d.reset()
    assert d.seed_count == 0


def test_reset_preserves_capacity_and_geometry():
    d = Delaunay(W, H, MAX_SEEDS, 5)
    before = (d.width, d.height, d.max_seeds, d.border_padding)
    d.insert_deferred(_random_seeds(np.random.default_rng(1), 30), None)
    d.reset()
    after = (d.width, d.height, d.max_seeds, d.border_padding)
    assert before == after


def test_reset_on_never_inserted_object_is_a_no_op():
    """reset() before any insert must not crash, and the object must stay
    usable afterward -- construction and reset both claim to reach the same
    state, so nothing should distinguish them here."""
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.reset()
    assert d.seed_count == 0
    seeds = _random_seeds(np.random.default_rng(2), 20)
    d.insert_deferred(seeds, None)
    assert d.seed_count == len(seeds)


def test_reset_clears_pending_flag():
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(_random_seeds(np.random.default_rng(3), 10), None)
    assert d.has_pending
    d.reset()
    assert not d.has_pending


def test_stale_view_raises_after_reset():
    """The generation counter, exercised for reset() the same way
    test_finalise_device.py exercises it for insert/finalise -- reset() must
    invalidate an outstanding finalise_device() view exactly like any other
    mutating call."""
    rng = np.random.default_rng(4)
    seeds = _random_seeds(rng, 20)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    _, pixel_tids, pixel_seed_ids, outside_mask = d.finalise_device()

    d.reset()

    for view in (pixel_tids, pixel_seed_ids, outside_mask):
        with pytest.raises(RuntimeError):
            view.__cuda_array_interface__


def test_reset_then_smaller_seed_set_matches_fresh():
    """The object was seeded generously, then reset and reused for a much
    smaller set -- capacity headroom left over from the larger set must not
    leak into the smaller triangulation."""
    rng = np.random.default_rng(5)
    big = _random_seeds(rng, 200)
    small = _random_seeds(rng, 8)

    reused = Delaunay(W, H, MAX_SEEDS, -1)
    reused.insert_deferred(big, None)
    reused.finalise(False)
    reused.reset()
    verts_reused, tgrid_reused = _finalise_grid(reused, small)

    fresh = Delaunay(W, H, MAX_SEEDS, -1)
    verts_fresh, tgrid_fresh = _finalise_grid(fresh, small)

    np.testing.assert_array_equal(verts_reused, verts_fresh)
    np.testing.assert_array_equal(tgrid_reused, tgrid_fresh)
