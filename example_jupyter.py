# %% [markdown]
# ## Voronoi + Delaunay example — 256 × 256 grid, 256 random seeds

# %% — imports and computation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from delauney import RegularDelaunay, GridTriangulation

W, H, N = 256, 256, 256

# Generate N unique seed positions
rng = np.random.default_rng(42)
coords: set[tuple[int, int]] = set()
while len(coords) < N:
    coords.add((int(rng.integers(0, W)), int(rng.integers(0, H))))
seeds = list(coords)

# Compute Voronoi diagram
vgrid = RegularDelaunay().compute(W, H, seeds)   # int32 (H, W, 2)

# Compute Delaunay triangulation
tri_map, tgrid = GridTriangulation().compute(vgrid, seeds)  # int32 (H, W, 3)

# Sorted seed positions: index == seed_id assigned by the library
sorted_seeds = sorted(seeds, key=lambda s: (s[0], s[1]))
sx = np.array([s[0] for s in sorted_seeds])
sy = np.array([s[1] for s in sorted_seeds])

print(f"Seeds: {N}   Triangles: {len(tri_map)}")

# %% — visualise
# Perceptually-spread colormap for N discrete regions
_hues = np.linspace(0, 1, N, endpoint=False)
_rng2 = np.random.default_rng(7)          # shuffle so neighbours differ visually
_hues = _hues[_rng2.permutation(N)]
region_colors = plt.cm.hsv(_hues)
region_cmap = mcolors.ListedColormap(region_colors)

# Separate (shuffled) colormap for the N triangles
_hues_t = np.linspace(0, 1, len(tri_map), endpoint=False)
_hues_t = _hues_t[_rng2.permutation(len(tri_map))]
tri_colors = plt.cm.hsv(_hues_t)
tri_cmap = mcolors.ListedColormap(tri_colors)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#1a1a2e')

# ── left: Voronoi diagram ──────────────────────────────────────────────────
ax = axes[0]
ax.imshow(
    vgrid[:, :, 0],
    cmap=region_cmap, vmin=0, vmax=N - 1,
    origin='upper', interpolation='nearest', aspect='equal',
    alpha=0.85,
)
ax.scatter(sx, sy, s=6, c='white', linewidths=0, zorder=5)
ax.set_title('Voronoi diagram', color='white', fontsize=13, pad=8)
ax.axis('off')

# ── right: Delaunay triangulation ─────────────────────────────────────────
ax = axes[1]

# Background: colour each pixel by its triangle_id
ax.imshow(
    tgrid[:, :, 2],
    cmap=tri_cmap, vmin=0, vmax=len(tri_map) - 1,
    origin='upper', interpolation='nearest', aspect='equal',
    alpha=0.55,
)

# Deduplicated triangle edges as a LineCollection
edge_set: set[tuple[int, int]] = set()
segments: list[list[tuple[float, float]]] = []
for _, (_, _, a, b, c) in tri_map.items():
    for u, v in ((a, b), (b, c), (a, c)):
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            segments.append([
                (sorted_seeds[u][0], sorted_seeds[u][1]),
                (sorted_seeds[v][0], sorted_seeds[v][1]),
            ])

ax.add_collection(LineCollection(segments, colors='white', linewidths=0.5, alpha=0.7))
ax.scatter(sx, sy, s=8, c='white', linewidths=0, zorder=5)
ax.set_xlim(0, W)
ax.set_ylim(H, 0)   # keep y-down convention to match imshow
ax.set_aspect('equal')
ax.set_title(f'Delaunay triangulation  ({len(tri_map)} triangles)',
             color='white', fontsize=13, pad=8)
ax.axis('off')

plt.tight_layout(pad=1.5)
plt.show()
