_CUDA_EXPORTS = {"Voronoi", "GridTriangulation", "Delaunay", "triangulate"}

__all__ = list(_CUDA_EXPORTS) + ["DEFAULT_BORDER_PADDING", "max_boundary_spacing"]


#: Pixels of Voronoi canvas added on each side when a caller does not choose.
#:
#: A triangle is only detected where three Voronoi regions meet -- its
#: circumcentre -- and boundary triangles frequently have circumcentres outside
#: the image, so at padding 0 they are never registered and the pixels they
#: cover degrade to nearest-seed with no error raised.
#:
#: A density-based estimate (e.g. sqrt(area / n_seeds)) is the wrong quantity:
#: circumradius depends on triangle *shape*, and grows without limit as three
#: points approach collinearity, at any seed count -- worst under clustered
#: sampling, where density is dominated by the dense bands while the triangles
#: needing padding span the sparse gaps.
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
        from delauney import _delauney_cuda as _cu
        globals().update({k: getattr(_cu, k) for k in _CUDA_EXPORTS})
        return globals()[name]
    raise AttributeError(f"module 'delauney' has no attribute {name!r}")
