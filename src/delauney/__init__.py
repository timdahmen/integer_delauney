import math

_CUDA_EXPORTS = {"RegularDelaunay", "GridTriangulation"}

__all__ = list(_CUDA_EXPORTS) + ["auto_border_padding"]


def auto_border_padding(width: int, height: int, n_seeds: int) -> int:
    """A border_padding that scales with seed density.

    A triangle is only detected where three Voronoi regions meet, and that point
    is its *circumcenter*.  Boundary triangles frequently have circumcenters
    outside the image, so without padding they are never registered at all and
    the pixels they should cover silently degrade to nearest-seed.  Padding
    re-runs the Voronoi on a larger canvas to expose them.

    The scale that governs circumcenter excursion is the seed spacing.  For a
    2D point process of density ``n / (width * height)`` the mean
    nearest-neighbour distance is ``0.5 * sqrt(area / n)``; empirically the
    recovery curve flattens at about twice that, while cost grows with the
    padded area — so this returns ``sqrt(area / n)``.

    Deliberately a closed-form density estimate rather than a measured
    nearest-neighbour distance.  For *clustered* seeds a measured mean is
    dominated by the dense clusters and comes out far too small, yet the
    triangles that actually need padding are the sparse ones spanning the gaps
    between clusters, which have the largest circumcircles.  The density
    estimate stays conservative in exactly that case, and agrees with a measured
    mean to within a pixel or two for uniform seeds.

    Note this reduces but never eliminates missing triangles: cocircular
    (4-region) meetings and sub-pixel slivers are missed at any padding.
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
