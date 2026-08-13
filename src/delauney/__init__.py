import math

_CUDA_EXPORTS = {"RegularDelaunay", "GridTriangulation"}

__all__ = list(_CUDA_EXPORTS) + ["auto_border_padding"]


def auto_border_padding(width: int, height: int, n_seeds: int) -> int:
    """A border_padding that scales with seed density.

    A triangle is only detected where three Voronoi regions meet -- its
    circumcenter.  Border triangles frequently have circumcenters outside the
    image, so at padding 0 they are never registered and the pixels they cover
    degrade to nearest-seed with no error raised.  Padding reruns the Voronoi
    on a larger canvas to expose them.

    Circumcenter excursion scales with seed spacing, whose mean is
    ``0.5 * sqrt(area / n)``; returning ``sqrt(area / n)`` covers it with
    margin without paying for padded area that recovers nothing.  A closed-form
    density estimate rather than a measured nearest-neighbour distance, because
    for clustered seeds a measured mean is dominated by the clusters while the
    triangles needing padding are the sparse ones spanning the gaps.

    Padding reduces but never eliminates missing triangles -- sub-pixel slivers
    are missed at any padding.
    """
    if n_seeds <= 0:
        return 0
    return int(round(math.sqrt(float(width) * float(height) / float(n_seeds))))


def __getattr__(name):
    if name in _CUDA_EXPORTS:
        from delauney._delauney_cuda import RegularDelaunay, GridTriangulation  # noqa: F401
        globals().update({k: v for k, v in locals().items() if k in _CUDA_EXPORTS})
        return globals()[name]
    raise AttributeError(f"module 'delauney' has no attribute {name!r}")
