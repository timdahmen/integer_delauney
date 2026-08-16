# A sufficient border padding, given boundary seed spacing

## The problem

`GridTriangulation` witnesses a triangle at the pixel where three Voronoi
regions meet — its circumcentre. A triangle whose circumcentre falls outside the
detection canvas is never registered, and the pixels it should cover degrade to
nearest-seed with no error raised.

`border_padding` enlarges the canvas to catch them. The question this note
answers is how large it has to be, and the answer is only bounded if the caller
controls how densely the *convex hull boundary* is sampled.

## Why the density estimate is not a bound

The former `auto_border_padding` returned `round(sqrt(area / n))`, reasoning
that circumcentre excursion scales with mean seed spacing. That is a global
density argument, and the quantity it estimates is not the one that matters.

Circumradius is a function of triangle *shape*, not of density: as three points
approach collinearity, R grows without limit at any n. Boundary regions are
exactly where slivers occur, because points just inside the hull form thin
triangles with the hull chord.

Measured on a real frame, 30 694 seeds on 1510×1018, for which the formula
returns 7:

    circumradius:  median 4.32   mean 4.77   p95 9.66   max 24.6

The formula is out by 3.5x at the tail. Under clustered sampling it is worse,
because `n/area` is dominated by the dense bands while the triangles needing
padding span the sparse gaps between them.

## The bound

Sample the image boundary deterministically at spacing `s` (integer pixels).
Then for every Delaunay triangle, the circumcentre lies within

    excursion  <=  s^2/8 - 1/2

pixels of the image, so `border_padding >= s^2/8` is sufficient.

### Lemma 1 — a boundary triangle uses *adjacent* boundary seeds

Let `A` and `B` be seeds on the top edge (`y = 0`) with a third boundary seed
`M` strictly between them. The circumcircle of any triangle on `A`,`B` meets the
line `y = 0` exactly at `A` and `B`, so `M`, lying on the chord between them, is
strictly inside that circle. The triangle is therefore not Delaunay.

Hence the base of any Delaunay triangle with two boundary vertices is exactly
`s`.

### Lemma 2 — the integer grid floors the third vertex

Take `A = (x, 0)`, `B = (x + s, 0)`, and a third vertex `C = (x + s/2 + u, c)`
with `c > 0` (inside the image). The circumcentre lies on the perpendicular
bisector of `AB`, the vertical line through `x + s/2`, at height `k` given by

    k = ( u^2 + c^2 - (s/2)^2 ) / (2c)

`k < 0` means the circumcentre is outside the image, at excursion `|k|`. The
excursion is largest when `u = 0`:

    |k| = ( (s/2)^2 - c^2 ) / (2c)

which is unbounded as `c -> 0`. Continuous geometry therefore gives no
guarantee at all. But seeds are integer pixels and the boundary row is `y = 0`,
so `c >= 1`, giving

    |k| <= ( (s/2)^2 - 1 ) / 2  =  s^2/8 - 1/2

### Verification

Delaunay triangulations (scipy) of a boundary ring at spacing `s` plus 3000
random interior seeds on 400x300, worst case over five seeds:

    s     bound s^2/8     measured max excursion
    4          2.0                        1.5
    6          4.5                        4.0
    8          8.0                        7.5
    12        18.0                       17.5
    16        32.0                       31.5

The bound is attained, not merely respected: the measured value is exactly
`s^2/8 - 1/2` at every spacing, which is the worst case the derivation predicts
(third vertex one pixel inside, directly below the midpoint of the base).

## Cost

Two costs pull in opposite directions. Boundary seeds are acquired
measurements; padding is grid work, scaling as `(W+2P)(H+2P) / (WH)`.

For 1510x1018:

    s     P = s^2/8    boundary seeds    padded grid overhead
    4          2            ~2530                0.8%
    8          8             ~632                2.6%
    11        15             ~460                4.9%
    16        32             ~316               11.1%

## Open cases

The derivation covers Delaunay triangles with **two** vertices on the boundary.
Two families are verified empirically by the table above but not derived:

  * triangles with a single boundary vertex and two interior ones
  * the four corners, where two boundary rows meet at right angles

An adversarial test — seeds crowded one pixel inside a corner — would be the
place to start if these are to be closed properly.

## What this does and does not guarantee

It bounds the *ideal* Delaunay triangulation's circumcentre excursion. It does
not make the raster pipeline exact: the CUDA Voronoi is an approximation whose
BFS can settle a cell on a non-nearest seed, and sub-pixel slivers are missed at
any padding. What it removes is the **unbounded** failure — a triangle lost
because its circumcentre was 25 px outside a canvas padded by 7 — and it makes
the padding requirement a chosen quantity rather than an estimated one, so
clustered sampling can no longer defeat it.
