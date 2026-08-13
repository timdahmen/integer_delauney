# integer_delauney

A GPU implementation of a Delaunay triangulation for pixel coordinates.

Seeds live on an integer grid. The library computes an L2 (Euclidean) Voronoi
diagram on the GPU, then extracts an approximate Delaunay triangulation from it
and assigns every pixel to its containing triangle.

See [API_REFERENCE.md](API_REFERENCE.md) for the Python API.

## Build

The CUDA extension needs the MSVC environment loaded before `pip install`.
`build.bat` does both — edit the `vcvars64.bat` path in it if your Visual
Studio install differs:

```
build.bat
```

To do it by hand, load the MSVC environment into your PowerShell session first:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && set' |
    ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
pip install -e . --no-cache-dir
```

For VS2019 Professional the script is at
`C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat`
and takes an architecture argument (`vcvarsall.bat x64`).

`WITH_CUDA` is on by default. The extension is imported lazily, so
`import delauney` works without a GPU; only the CUDA classes need one. The pure
NumPy fallbacks in `delauney.reference` always work.

## Test

```
pytest
```

CUDA tests skip automatically when the extension is not built.
