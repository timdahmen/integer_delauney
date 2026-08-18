"""Tests for Delaunay.finalise_device(): the device-resident counterpart of
finalise(as_arrays=True) that hands pixel_tids/pixel_seed_ids/outside_hull_mask
to the caller as __cuda_array_interface__ views instead of downloading and
cropping them on the host.

Correctness strategy
---------------------
finalise_device()'s crop kernel replaces build_outputs_()'s host crop/remap
loop, so the bar is byte-for-byte agreement with finalise()'s (H,W,3) grid on
an identically-seeded mesh -- read back from the device with a plain
cudaMemcpy via ctypes, since this project has no cupy/torch dependency.

Separately, the generation counter that guards against reading a view after
the mesh has moved on is tested by mutating the mesh between
finalise_device() and the read and checking the resulting RuntimeError --
that failure mode has no other test coverage anywhere in the suite.
"""
import ctypes
import ctypes.util

import numpy as np
import pytest


def _cuda_available() -> bool:
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


def _load_cudart():
    for name in ("cudart64_13.dll", "cudart64_12.dll", "libcudart.so"):
        path = ctypes.util.find_library(name) or name
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="_delauney_cuda extension not built or no CUDA device",
)

if _cuda_available():
    from delauney._delauney_cuda import Delaunay

_CUDART = _load_cudart() if _cuda_available() else None
_CUDA_MEMCPY_DEVICE_TO_HOST = 2

W, H, MAX_SEEDS = 64, 48, 512


def _read_device(cai, dtype):
    """Copy a __cuda_array_interface__ view back to a host numpy array."""
    if _CUDART is None:
        pytest.skip("no cudart available to read device memory back for assertion")
    ptr, count = cai["data"][0], cai["shape"][0]
    out = np.empty(count, dtype=dtype)
    nbytes = count * np.dtype(dtype).itemsize
    rc = _CUDART.cudaMemcpy(
        out.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
        ctypes.c_size_t(nbytes), ctypes.c_int(_CUDA_MEMCPY_DEVICE_TO_HOST))
    assert rc == 0, f"cudaMemcpy D2H failed, rc={rc}"
    return out


def _random_seeds(rng, n, w=W, h=H):
    xs = rng.integers(0, w, size=n).astype(np.int32)
    ys = rng.integers(0, h, size=n).astype(np.int32)
    xy = np.unique(np.stack([xs, ys], axis=1), axis=0)
    return [(int(x), int(y)) for x, y in xy]


def _host_reference(seeds):
    """finalise()'s (H,W,3) grid, split/masked the way TriangulationAdapter
    does, as the ground truth the device path must match exactly."""
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    verts, tgrid = d.finalise(as_arrays=True)
    tgrid = np.asarray(tgrid)
    seed_ids = tgrid[:, :, 0].reshape(-1).astype(np.int32)
    raw_tids = tgrid[:, :, 2].reshape(-1).astype(np.int32)
    outside = (raw_tids == -1).astype(np.uint8)
    tids = np.where(outside, 0, raw_tids).astype(np.int32)
    return np.asarray(verts), seed_ids, outside, tids


@pytest.mark.parametrize("n_seeds", [10, 40, 150])
def test_finalise_device_matches_finalise(n_seeds):
    rng = np.random.default_rng(n_seeds)
    seeds = _random_seeds(rng, n_seeds)

    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    verts, pixel_tids, pixel_seed_ids, outside_mask = d.finalise_device()

    tids_dev = _read_device(pixel_tids.__cuda_array_interface__, np.int32)
    sids_dev = _read_device(pixel_seed_ids.__cuda_array_interface__, np.int32)
    mask_dev = _read_device(outside_mask.__cuda_array_interface__, np.uint8)

    verts_ref, sids_ref, mask_ref, tids_ref = _host_reference(seeds)

    np.testing.assert_array_equal(np.asarray(verts), verts_ref)
    np.testing.assert_array_equal(sids_dev, sids_ref)
    np.testing.assert_array_equal(mask_dev, mask_ref)
    np.testing.assert_array_equal(tids_dev, tids_ref)


def test_finalise_device_cuda_array_interface_shape():
    rng = np.random.default_rng(0)
    seeds = _random_seeds(rng, 30)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    _, pixel_tids, pixel_seed_ids, outside_mask = d.finalise_device()

    for view, typestr in ((pixel_tids, "<i4"), (pixel_seed_ids, "<i4"),
                          (outside_mask, "|u1")):
        cai = view.__cuda_array_interface__
        assert cai["shape"] == (H * W,)
        assert cai["typestr"] == typestr
        assert cai["data"][1] is True  # read-only
        assert cai["version"] == 3


def test_fresh_view_reads_without_error():
    rng = np.random.default_rng(1)
    seeds = _random_seeds(rng, 20)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    _, pixel_tids, _, _ = d.finalise_device()
    pixel_tids.__cuda_array_interface__  # must not raise


def test_stale_view_raises_after_mutation():
    """The generation counter, exercised intentionally.

    A view read after the mesh it came from is mutated must raise rather than
    silently return memory that has moved on -- this is the one new failure
    surface the device handoff adds, and "it didn't crash" is not enough
    coverage for it.
    """
    rng = np.random.default_rng(2)
    seeds = _random_seeds(rng, 20)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    _, pixel_tids, pixel_seed_ids, outside_mask = d.finalise_device()

    more = _random_seeds(rng, 5)
    more = [s for s in more if s not in set(seeds)]
    if not more:
        pytest.skip("random seeds collided with the original set")
    d.insert_deferred(more, None)  # mutates -> must invalidate the views above

    for view in (pixel_tids, pixel_seed_ids, outside_mask):
        with pytest.raises(RuntimeError):
            view.__cuda_array_interface__


@pytest.mark.parametrize("n_seeds", [10, 40, 150])
def test_to_host_matches_finalise(n_seeds):
    """to_host()'s own D2H copy against the same host reference as
    __cuda_array_interface__ above -- the two must agree since they read the
    same buffers, just via a different route (no cudart ctypes call needed
    here, to_host() does its own memcpy)."""
    rng = np.random.default_rng(n_seeds)
    seeds = _random_seeds(rng, n_seeds)

    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    verts, pixel_tids, pixel_seed_ids, outside_mask = d.finalise_device()

    tids_host = pixel_tids.to_host()
    sids_host = pixel_seed_ids.to_host()
    mask_host = outside_mask.to_host()

    verts_ref, sids_ref, mask_ref, tids_ref = _host_reference(seeds)

    assert tids_host.dtype == np.int32
    assert sids_host.dtype == np.int32
    assert mask_host.dtype == np.uint8
    np.testing.assert_array_equal(tids_host, tids_ref)
    np.testing.assert_array_equal(sids_host, sids_ref)
    np.testing.assert_array_equal(mask_host, mask_ref)


def test_stale_view_to_host_raises_after_mutation():
    """to_host()'s own staleness check, exercised the same way as the
    __cuda_array_interface__ property's above -- it is a separate code path
    (own generation comparison) and needs its own coverage."""
    rng = np.random.default_rng(4)
    seeds = _random_seeds(rng, 20)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    _, pixel_tids, pixel_seed_ids, outside_mask = d.finalise_device()

    more = _random_seeds(rng, 5)
    more = [s for s in more if s not in set(seeds)]
    if not more:
        pytest.skip("random seeds collided with the original set")
    d.insert_deferred(more, None)  # mutates -> must invalidate the views above

    for view in (pixel_tids, pixel_seed_ids, outside_mask):
        with pytest.raises(RuntimeError):
            view.to_host()


def test_second_finalise_device_invalidates_the_first():
    """Two finalise_device() calls in a row on the same mesh, with nothing
    inserted between them, still bump the generation -- the second call's
    crop kernel re-launches unconditionally (see finalise_device()'s
    docstring), so the first call's views are no longer current."""
    rng = np.random.default_rng(3)
    seeds = _random_seeds(rng, 20)
    d = Delaunay(W, H, MAX_SEEDS, -1)
    d.insert_deferred(seeds, None)
    _, pixel_tids_1, _, _ = d.finalise_device()
    _, pixel_tids_2, _, _ = d.finalise_device()

    with pytest.raises(RuntimeError):
        pixel_tids_1.__cuda_array_interface__
    pixel_tids_2.__cuda_array_interface__  # still current, must not raise
