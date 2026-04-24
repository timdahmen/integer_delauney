_CUDA_EXPORTS = {"RegularDelaunay", "GridTriangulation"}

__all__ = list(_CUDA_EXPORTS)


def __getattr__(name):
    if name in _CUDA_EXPORTS:
        from delauney._delauney_cuda import RegularDelaunay, GridTriangulation  # noqa: F401
        globals().update({k: v for k, v in locals().items() if k in _CUDA_EXPORTS})
        return globals()[name]
    raise AttributeError(f"module 'delauney' has no attribute {name!r}")
