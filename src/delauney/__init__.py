_CUDA_EXPORTS = {"Voronoi", "GridTriangulation", "Delaunay", "triangulate"}

__all__ = list(_CUDA_EXPORTS) + ["DEFAULT_BORDER_PADDING", "max_boundary_spacing"]


#: Pixels of Voronoi canvas added on each side when a caller does not choose.
#:
#: A triangle is only detected where three Voronoi regions meet - its
#: circumcentre - and boundary triangles frequently have circumcentres outside
#: the image, so at padding 0 they are never registered and the pixels they
#: cover degrade to nearest-seed with no error raised.
#:
#: Depending on how densely the boundary is samples, a specific padding guarantuees
#: no triangles are missed. 
#: See BORDER_PADDING_BOUND.md, and max_boundary_spacing() below.
#:
DEFAULT_BORDER_PADDING = 16

def max_boundary_spacing(border_padding: int = DEFAULT_BORDER_PADDING) -> int:
    """Largest boundary seed spacing for which no triangle can be missed.
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
