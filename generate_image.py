"""Generate presentation-quality Voronoi / Delaunay images.

Outputs (all 2048×2048):
  presentation_voronoy.png          — analytical Voronoi, smooth cell boundaries
  presentation_delauney.png         — analytical Voronoi + triangulation overlay
  presentation_pixels.png           — discretized pixels + analytical boundaries
  presentation_pixels_delauney.png  — discretized pixels + triangulation overlay
  presentation_padded.png           — the full padded detection canvas
  step_NN.png                       — BFS growth animation, one frame per
                                      iteration (count depends on seed spacing)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi
from delauney import RegularDelaunay, GridTriangulation

# ── Parameters ────────────────────────────────────────────────────────────────
W, H     = 64, 64
DENSITY  = 0.025
OUT_PX   = 2048
RNG_SEED = 42
DPI      = 200
SCALE    = OUT_PX // W    # pixels per Voronoi cell in the upscaled raster
BORDER_PADDING = 16      # padding added around the image for border-triangle detection

# ── Seeds ─────────────────────────────────────────────────────────────────────
N_target = int(W * H * DENSITY)
rng = np.random.default_rng(RNG_SEED)

coords: set[tuple[int, int]] = set()
while len(coords) < N_target:
    coords.add((int(rng.integers(0, W)), int(rng.integers(1, H))))

# ── Enforce a clean 4-cell meeting point ──────────────────────────────────────
# Place 4 seeds at the corners of a large square whose centre lies on a pixel
# corner (odd side length → centre at *.5), so their Voronoi cells meet exactly
# at that centre: a 2×2 block of pixels where all four cells touch.
#
# For the four corners to actually own that centre they must be the *closest*
# seeds to it.  Clearing only the square's interior is NOT enough — a stray seed
# just outside an edge midpoint is closer to the centre (side/2) than the corners
# are (side/2·√2).  So we clear every other seed inside the square's
# circumscribed circle (plus a small margin) instead.
SQ_LO, SQ_HI = 21, 42                        # side = 21 (odd) → centre (31.5, 31.5)
square_corners = [(SQ_LO, SQ_LO), (SQ_HI, SQ_LO),
                  (SQ_LO, SQ_HI), (SQ_HI, SQ_HI)]
cx = cy = (SQ_LO + SQ_HI) / 2
half_diag = ((SQ_HI - cx) ** 2 + (SQ_HI - cy) ** 2) ** 0.5
clear_r2  = (half_diag + 2.0) ** 2           # circumscribed circle + 2px margin

coords = {(x, y) for (x, y) in coords
          if (x - cx) ** 2 + (y - cy) ** 2 > clear_r2}
coords.update(square_corners)

seeds = list(coords)
N = len(seeds)
print(f"Grid: {W}×{H}   Seeds: {N}   ({100*N/(W*H):.1f}%)")

# Library sorts seeds by (x asc, y asc) before assigning IDs
sorted_seeds = sorted(seeds, key=lambda s: (s[0], s[1]))
points = np.array(sorted_seeds, dtype=float)   # (N, 2)  columns: x, y

# ── Color palette ─────────────────────────────────────────────────────────────
rng2 = np.random.default_rng(13)
hues = np.linspace(0, 1, N, endpoint=False)
hues = hues[rng2.permutation(N)]
hsv  = np.stack([hues, np.full(N, 0.55), np.full(N, 0.90)], axis=1)
rgb_palette = hsv_to_rgb(hsv)   # (N, 3)

# ── Discretized Voronoi: 64×64 → 2048×2048 nearest-neighbour upscale ─────────
vgrid        = RegularDelaunay().compute(W, H, seeds)
seed_id_grid = vgrid[:, :, 0]                     # (H, W)
rgb_small    = rgb_palette[seed_id_grid]           # (H, W, 3)
rgb_large    = np.repeat(np.repeat(rgb_small, SCALE, axis=0), SCALE, axis=1)

# ── Delaunay triangulation (triangle edges between seed pairs) ────────────────
tri_map, _, padded_vgrid = GridTriangulation().compute_debug(
    vgrid, seeds, border_padding=BORDER_PADDING)

edge_set      = set()
tri_segments  = []
for _, (_, _, a, b, c) in tri_map.items():
    for u, v in ((a, b), (b, c), (a, c)):
        key = (min(u, v), max(u, v))
        if key not in edge_set:
            edge_set.add(key)
            tri_segments.append([
                (sorted_seeds[u][0], sorted_seeds[u][1]),
                (sorted_seeds[v][0], sorted_seeds[v][1]),
            ])
print(f"Triangles: {len(tri_map)}   Edges: {len(tri_segments)}")

# ── Analytical Voronoi (scipy: exact perpendicular bisectors) ─────────────────
vor        = Voronoi(points)
center     = points.mean(axis=0)
far_radius = max(W, H) * 4    # large enough to push infinite rays off-screen

def build_finite_regions(vor, center, far_radius):
    """Return (regions, verts): every region polygon has only finite vertices.

    Infinite cells (touching the convex hull) are closed by extending their
    open rays to far_radius along the outward-facing perpendicular bisector.
    """
    adj = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        adj.setdefault(p1, []).append((p2, v1, v2))
        adj.setdefault(p2, []).append((p1, v1, v2))

    verts   = vor.vertices.tolist()
    regions = []

    for p_idx, reg_idx in enumerate(vor.point_region):
        vidx = vor.regions[reg_idx]
        if all(v >= 0 for v in vidx):
            regions.append(vidx)
            continue

        new_vidx = [v for v in vidx if v >= 0]
        for p2, v1, v2 in adj.get(p_idx, []):
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p_idx]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            mid = vor.points[[p_idx, p2]].mean(axis=0)
            direction = np.sign(np.dot(mid - center, n)) * n
            new_vidx.append(len(verts))
            verts.append((vor.vertices[v2] + direction * far_radius).tolist())

        vs    = np.array([verts[v] for v in new_vidx])
        c     = vs.mean(axis=0)
        order = np.argsort(np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0]))
        regions.append([new_vidx[i] for i in order])

    return regions, np.array(verts)

regions, verts = build_finite_regions(vor, center, far_radius)

# ── Drawing helpers ───────────────────────────────────────────────────────────
BORDER_COLOR = '#0d0d0d'
BORDER_LW    = 1.4
BG_COLOR     = '#111111'

def new_fig():
    fig, ax = plt.subplots(figsize=(OUT_PX / DPI, OUT_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)    # y increases downward (image convention)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax

def draw_analytical_fills(ax):
    for i, region in enumerate(regions):
        ax.add_patch(Polygon(verts[region], closed=True,
                             facecolor=rgb_palette[i], edgecolor='none'))

def draw_analytical_boundaries(ax, color=BORDER_COLOR, lw=BORDER_LW):
    for (p1, p2), simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if all(s >= 0 for s in simplex):
            ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1],
                    color=color, lw=lw, solid_capstyle='round', zorder=3)
        else:
            i_fin = simplex[simplex >= 0][0]
            t   = points[p2] - points[p1]
            t  /= np.linalg.norm(t)
            n   = np.array([-t[1], t[0]])
            mid = points[[p1, p2]].mean(axis=0)
            d   = np.sign(np.dot(mid - center, n)) * n
            far = vor.vertices[i_fin] + d * far_radius
            ax.plot([vor.vertices[i_fin, 0], far[0]],
                    [vor.vertices[i_fin, 1], far[1]],
                    color=color, lw=lw, solid_capstyle='round', zorder=3)

def draw_triangulation(ax):
    lc = LineCollection(tri_segments, colors='black',
                        linewidths=2.2, alpha=0.95, zorder=4)
    ax.add_collection(lc)

def draw_seeds(ax):
    ax.scatter(points[:, 0], points[:, 1], s=130, c='black',
               linewidths=0, zorder=6, alpha=0.55)
    ax.scatter(points[:, 0], points[:, 1], s=50,  c='white',
               linewidths=0, zorder=7)

def draw_discretized_bg(ax):
    # Place the 2048×2048 upscaled image in the [0,W]×[0,H] data space.
    # extent = (left, right, bottom, top) in data coordinates.
    ax.imshow(rgb_large, extent=[0, W, H, 0],
              origin='upper', interpolation='nearest',
              aspect='equal', zorder=0)

def save(fig, name):
    fig.savefig(name, dpi=DPI, bbox_inches='tight', pad_inches=0)
    print(f"Saved {name}")
    plt.close(fig)

# ── Image 1: analytical Voronoi with smooth boundaries ───────────────────────
fig, ax = new_fig()
draw_analytical_fills(ax)
draw_analytical_boundaries(ax)
draw_seeds(ax)
save(fig, 'presentation_voronoy.png')

# ── Image 2: Voronoi + Delaunay triangulation ─────────────────────────────────
fig, ax = new_fig()
draw_analytical_fills(ax)
# draw_analytical_boundaries(ax)
draw_triangulation(ax)
draw_seeds(ax)
save(fig, 'presentation_delauney.png')

# ── Image 3: discretized pixel colors + analytical boundaries ─────────────────
fig, ax = new_fig()
draw_discretized_bg(ax)
draw_analytical_boundaries(ax, color='#111111', lw=1.6)
draw_seeds(ax)
save(fig, 'presentation_pixels.png')

# ── Image 4: discretized pixel colors + Delauney triangulation ─────────────────
fig, ax = new_fig()
draw_discretized_bg(ax)
draw_triangulation(ax)
draw_seeds(ax)
save(fig, 'presentation_pixels_delauney.png')

# ── Image 5: full padded canvas (pixels + triangulation incl. border padding) ─
# Every image above is clipped to the original [0,W]×[0,H] region.  Border-triangle
# detection, however, runs on a (W+2P)×(H+2P) padded Voronoi canvas.  This image
# renders that *entire* padded canvas so the padding margin — and the border
# triangles that only exist because of it — is visible.  The original image
# extent is outlined in yellow.
P = BORDER_PADDING
rgb_padded = rgb_palette[padded_vgrid[:, :, 0]]        # (H+2P, W+2P, 3)

fig, ax = plt.subplots(figsize=(OUT_PX / DPI, OUT_PX / DPI), dpi=DPI)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
# padded_vgrid[0,0] corresponds to original coordinate (-P, -P)
ax.imshow(rgb_padded, extent=[-P, W + P, H + P, -P],
          origin='upper', interpolation='nearest', aspect='equal', zorder=0)
draw_triangulation(ax)
draw_seeds(ax)
ax.add_patch(Rectangle((0, 0), W, H, linewidth=1.4,
                       edgecolor='yellow', facecolor='none', zorder=5))
ax.set_xlim(-P, W + P)
ax.set_ylim(H + P, -P)          # y increases downward (image convention)
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
save(fig, 'presentation_padded.png')

# ── Animation: BFS growth, one frame per iteration (step_01.png …) ───────────
# Frame 01 = initialisation (only seed pixels lit); frame N = after N-1
# iterations.  The wavefront expands one cardinal step per iteration, so the
# pixels revealed at iteration k are those with Manhattan distance <= k to
# their seed — even though cell *ownership* is decided by L2.  Colours come
# from the final L2 assignment, to match the other images.

# Manhattan distance from each pixel to its L2-assigned seed
ys_grid, xs_grid = np.mgrid[0:H, 0:W]
s_xs = np.array([s[0] for s in sorted_seeds], dtype=np.int32)
s_ys = np.array([s[1] for s in sorted_seeds], dtype=np.int32)

manhattan_dist = (np.abs(xs_grid - s_xs[seed_id_grid]) +
                  np.abs(ys_grid - s_ys[seed_id_grid]))   # (H, W)

max_manhattan = int(manhattan_dist.max())
print(f"Manhattan distance range: 0 to {max_manhattan}  ({max_manhattan + 1} frames)")

BG_DARK = np.array([0.067, 0.067, 0.067])

for step in range(max_manhattan + 1):            # one frame per BFS iteration
    covered   = manhattan_dist <= step

    rgb_frame = np.where(
        covered[:, :, np.newaxis],
        rgb_palette[seed_id_grid],
        BG_DARK,
    ).astype(np.float32)

    frame_large = np.repeat(np.repeat(rgb_frame, SCALE, axis=0), SCALE, axis=1)

    fig, ax = new_fig()
    ax.imshow(frame_large, extent=[0, W, H, 0],
              origin='upper', interpolation='nearest',
              aspect='equal', zorder=0)
    draw_seeds(ax)
    save(fig, f'step_{step + 1:02d}.png')
