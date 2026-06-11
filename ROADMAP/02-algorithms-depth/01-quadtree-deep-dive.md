# QuadTree Deep Dive

## Learning Objective

Understand your existing QuadTree at the level where you could prove its
performance characteristics and implement it from memory.

## Refresher: What a QuadTree Is

A QuadTree recursively subdivides 2D space into four equal quadrants.
Each node stores points and has a capacity. When capacity is exceeded,
the node subdivides into four children and redistributes its points.

Your implementation in `game.py` is a **point-region QuadTree (PR-Quadtree)**:
points live only in leaf nodes.

## Formal Analysis

### Data Structure

```
Node:
  - boundary: Rect (center x, y, half-width w, half-height h)
  - capacity: int (max points before split, constant)
  - points: list[Point]
  - children: NW, NE, SW, SE (optional, exist only if subdivided)
```

**Invariant:** A node is either a leaf (has points, no children) or an
internal node (has children, no points of its own).

### Insert Complexity

**Best case: O(1)** — the root has room, point is appended.

**Average case: O(log₄ n)** — for uniformly distributed points, the tree
depth is ~log₄(n/capacity). Each level has one boundary check (O(1)).
Total: O(log n).

**Worst case: O(n)** — all points cluster in one corner. The tree
degenerates: one child gets all the points, subdivides, repeats. Depth
= n/capacity. Each point insertion traverses this spine.

*Proof of worst-case:* Consider points at (0.1, 0.1), (0.11, 0.11),
(0.101, 0.101), ... All fall in the NW quadrant of every subdivision.
The tree becomes a chain of length n/capacity.

### Query Complexity (Range Search)

**Best case: O(1)** — query range doesn't intersect root.

**Average case: O(k + log n)** — k = number of returned points. Query
visits nodes whose boundaries intersect the search range. For uniform
distributions, this is O(log n) nodes plus O(k) for collecting points.

**Worst case: O(n + k) = O(n)** — query range covers the entire space.
Every node and every point is visited.

*Space complexity:* O(n) for stored points + O(m) for internal nodes,
where m ≤ 4n/capacity. So O(n).

### Comparison with Your Implementation

Your `game.py` QuadTree:

```python
class QuadTree:
    def __init__(self, boundary, n):
        self.boundary = boundary   # Rect
        self.capacity = n           # int
        self.points = []           # list
        self.divided = False
```

This is a **canonical PR-Quadtree**. No bugs, no optimizations.
It's textbook-correct.

## Checking the Implementation

Your `Rect` class uses half-widths:

```python
class Rect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
```

Where `x, y` is center and `w, h` is half-width/height. The `contains`
method:

```python
def contains(self, p):
    return (self.x - self.w <= p.x < self.x + self.w) and \
           (self.y - self.h <= p.y < self.y + self.h)
```

Note: uses `<=` on the left and `<` on the right. This means points on
the right/bottom boundary are **excluded** from the current quadrant and
will fall to the adjacent quadrant. This is intentional — it avoids
double-counting on boundaries. However, it means a particle exactly at
x = +4000 is outside any cell. The world boundary checks in the particle
movement code (`if self.x < -limit or self.x > limit: self.vx *= -1`)
use `>` which excludes +4000, so a particle at exactly +4000 is bounced
back. Consistent, but worth noting.

**Edge case to test:** What happens to a particle at exactly (-4000, -4000)?

## Implementation Task

### Part 1: Visualize the QuadTree Structure

Add a method to `QuadTree` that prints the tree depth and node count:

```python
def stats(self, depth=0):
    """Return (depth, num_nodes, num_points, max_depth)"""
    if not self.divided:
        return (depth, 1, len(self.points), depth)
    else:
        children = [self.nw, self.ne, self.sw, self.se]
        child_stats = [c.stats(depth+1) for c in children]
        max_d = max(cs[3] for cs in child_stats)
        total_nodes = 1 + sum(cs[1] for cs in child_stats)
        total_pts = sum(cs[2] for cs in child_stats)
        return (depth, total_nodes, total_pts, max_d)
```

### Part 2: Worst-Case Input

Write a script that:
1. Creates 10,000 particles all clustered in a 1m × 1m corner
2. Inserts them into a QuadTree
3. Measures insert time and tree depth
4. Compares with 10,000 uniformly distributed particles

Confirm the depth ~ n/capacity in the worst case.

### Part 3: Capacity Tuning

Vary the capacity parameter from 1 to 64. For each:
- Insert 10,000 uniform particles
- Run 1,000 random-range queries
- Plot insert time, query time, and tree depth vs capacity

What capacity minimizes total (insert + query) time? Why?

## Check-In Questions

1. Your QuadTree stores points only in leaf nodes. What would change if
   internal nodes also stored points (a "loose QuadTree")?

2. In the worst-case clustered input, what is the tree depth as a function
   of n and capacity?

3. Why does the QuadTree use a `<` (strict less) on the right boundary
   check? What would break if it used `<=`?

4. The QuadTree `insert` returns `True` if the point was inserted and
   `False` if it was outside the boundary. When would this return value
   be useful?

## Complexity Benchmark

```python
import time
import numpy as np
from game import QuadTree, Rect, Particle, WORLD_SIZE

boundary = Rect(0, 0, 4000, 4000)

for n in [100, 1000, 10000, 100000]:
    particles = [Particle(np.random.uniform(-4000, 4000),
                          np.random.uniform(-4000, 4000))
                 for _ in range(n)]
    
    t0 = time.perf_counter()
    qt = QuadTree(boundary, 4)
    for p in particles:
        qt.insert(p)
    t1 = time.perf_counter()
    
    found = []
    t2 = time.perf_counter()
    for _ in range(1000):
        found.clear()
        qt.query(Rect(0, 0, 500, 500), found)
    t3 = time.perf_counter()
    
    _, total_nodes, total_pts, max_depth = qt.stats()
    print(f"n={n:6d} | insert={(t1-t0)*1000:8.2f}ms | query={(t3-t2)*1000:8.2f}ms | "
          f"nodes={total_nodes:6d} | depth={max_depth:3d}")
```

The insert time should scale roughly as O(log n) — each 10x increase adds
~1.15x to log₄ time. Query time for a fixed-size range should also grow
slowly.

## Connection Back

Your `game.py` QuadTree is the foundation. It's used for the mouse radar
query every frame. Understanding its performance profile tells you at
what particle count the frame rate will drop (hint: it's higher than 400).

**Next:** `02-uniform-grid.md`
