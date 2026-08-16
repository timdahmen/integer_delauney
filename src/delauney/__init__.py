_CUDA_EXPORTS = {"Voronoi", "GridTriangulation"}

__all__ = list(_CUDA_EXPORTS) + ["DEFAULT_BORDER_PADDING", "max_boundary_spacing"]


#: Pixels of Voronoi canvas added on each side when a caller does not choose.
#:
#: A triangle is only detected where three Voronoi regions meet -- its
#: circumcentre -- and boundary triangles frequently have circumcentres outside
#: the image, so at padding 0 they are never registered and the pixels they
#: cover degrade to nearest-seed with no error raised.
#:
#: This used to be estimated per call as ``sqrt(area / n_seeds)``, on the
#: argument that circumcentre excursion scales with mean seed spacing. That is a
#: global density argument, and the quantity it estimates is not the one that
#: matters: circumradius depends on triangle *shape*, and grows without limit as
#: three points approach collinearity, at any seed count. Measured on a real
#: 1510x1018 frame with 30694 seeds, for which the formula returned 7, the
#: largest circumradius was 24.6 -- out by 3.5x, and worse under clustered
#: sampling, where the density estimate is dominated by the dense bands while
#: the triangles needing padding span the sparse gaps.
#:
#: A fixed default is not a better estimate; it is an honest one. The padding a
#: seed set actually needs is bounded only if the caller controls how densely
#: the convex hull boundary is sampled, and then the bound is exact. See
#: BORDER_PADDING_BOUND.md, and max_boundary_spacing() below.
#:
#: 16 admits a boundary spacing of 11 px, which costs about 5% in padded grid
#: area. 32 would admit 16 px spacing for 11%.
DEFAULT_BORDER_PADDING = 16


def max_boundary_spacing(border_padding: int = DEFAULT_BORDER_PADDING) -> int:
    """Largest boundary seed spacing for which no triangle can be missed.

    A caller that samples the image boundary at spacing ``s`` is guaranteed
    every Delaunay triangle has its circumcentre within ``s^2/8`` pixels of the
    image, so ``border_padding >= s^2/8`` suffices. Inverting that gives the
    spacing a given padding supports.

    The bound is derived and verified in BORDER_PADDING_BOUND.md. It is exact
    rather than conservative: the worst case is attained by a triangle whose
    third vertex sits one pixel inside, directly below the midpoint of a
    boundary edge, and the integer pixel grid is what makes it finite at all.
    """
    if border_padding <= 0:
        return 0
    return int((8 * border_padding) ** 0.5)


def __getattr__(name):
    if name in _CUDA_EXPORTS:
        from delauney._delauney_cuda import Voronoi, GridTriangulation  # noqa: F401
        globals().update({k: v for k, v in locals().items() if k in _CUDA_EXPORTS})
        return globals()[name]
    raise AttributeError(f"module 'delauney' has no attribute {name!r}")
