try:
    from delauney._delauney_cuda import RegularDelaunay, GridTriangulation
except ImportError:
    from delauney.reference.voronoi import RegularDelaunay
    from delauney.reference.triangulation import GridTriangulation

__all__ = ["RegularDelaunay", "GridTriangulation"]
