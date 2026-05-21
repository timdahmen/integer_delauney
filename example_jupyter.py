# %% [markdown]
# ## Voronoi + Delaunay — parameterized reproduction script
#
# Edit the PARAMETERS cell below to switch between scenarios:
#
#   SCENARIO 1 — baseline (works correctly)
#       W, H, N = 256, 256, 256   ALLOW_DUPLICATES = False
#
#   SCENARIO 2 — sparse seeds (previously triggered WINDOW_CAP bug)
#       W, H, N = 1024, 512, 32   ALLOW_DUPLICATES = False
#
#   SCENARIO 3 — duplicate seeds (triggers silent-overwrite bug)
#       W, H, N = 256, 256, 256   ALLOW_DUPLICATES = True
#       Expected: some seeds missing from Voronoi; unexpected -1 regions.

# %% — PARAMETERS
W, H = 512, 512           # grid width, height in pixels
N    = 512                # number of seeds requested
ALLOW_DUPLICATES = False  # True → skip deduplication to expose overwrite bug
RNG_SEED = 42

# %% — imports and computation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from delauney import RegularDelaunay, GridTriangulation

rng = np.random.default_rng(RNG_SEED)

if ALLOW_DUPLICATES:
    seeds = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(N)]
    n_unique = len(set(seeds))
    n_dupes  = N - n_unique
else:
    coords: set[tuple[int, int]] = set()
    while len(coords) < N:
        coords.add((int(rng.integers(0, W)), int(rng.integers(0, H))))
    seeds    = list(coords)
    n_unique = N
    n_dupes  = 0

# Compute Voronoi diagram
vgrid = RegularDelaunay().compute(W, H, seeds)   # int32 (H, W, 2)

# Compute Delaunay triangulation
tri_map, tgrid = GridTriangulation().compute(vgrid, seeds)  # int32 (H, W, 3)

# Sorted seed positions: index == seed_id assigned by the library
sorted_seeds = sorted(seeds, key=lambda s: (s[0], s[1]))
sx = np.array([s[0] for s in sorted_seeds])
sy = np.array([s[1] for s in sorted_seeds])

# --- diagnostics ---
max_voronoi_dist = int(vgrid[:, :, 1].max())
n_outside = int((tgrid[:, :, 2] == -1).sum())
pct_outside = 100.0 * n_outside / (W * H)

print(f"Grid: {W}x{H}   Seeds: {n_unique}   Duplicates: {n_dupes}")
print(f"Triangles found: {len(tri_map)}")
print(f"Max Voronoi distance: {max_voronoi_dist}")
print(f"Pixels outside triangulation (-1): {n_outside} ({pct_outside:.1f}%)")
if n_dupes:
    print(f"WARNING: {n_dupes} duplicate seed position(s)")

# %% — visualise
_hues = np.linspace(0, 1, N, endpoint=False)
_rng2 = np.random.default_rng(7)
_hues = _hues[_rng2.permutation(N)]
region_colors = plt.cm.hsv(_hues)
region_cmap = mcolors.ListedColormap(region_colors)

_n_tri = max(len(tri_map), 1)
_hues_t = np.linspace(0, 1, _n_tri + 1, endpoint=False)
_hues_t = _hues_t[_rng2.permutation(_n_tri)]
tri_colors = plt.cm.hsv(_hues_t)
tri_colors[0, 3] = 0          # make index 0 transparent (maps to -1 via vmin=-1)
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
ax.imshow(
    tgrid[:, :, 2],
    cmap=tri_cmap, vmin=-1, vmax=max(_n_tri - 1, 0),
    origin='upper', interpolation='nearest', aspect='equal',
    alpha=0.55,
)

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
ax.set_ylim(H, 0)
ax.set_aspect('equal')
ax.set_title(
    f'Delaunay triangulation  ({len(tri_map)} triangles)\n'
    f'max dist={max_voronoi_dist}  outside={pct_outside:.1f}%',
    color='white', fontsize=11, pad=8,
)
ax.axis('off')

plt.tight_layout(pad=1.5)
plt.savefig("triangulation.png", bbox_inches='tight')
plt.show()
