# KD-Tree

## Learning Objective

Build a KD-Tree — the spatial data structure of choice for nearest-neighbor
search in low dimensions. Understand its construction, query algorithms,
and complexity.

## Motivation

QuadTrees partition space by area (all four quadrants at once). KD-Trees
partition space by alternating dimensions (split on x, then y, then x, ...).
This produces a binary tree with several advantages:

1. Better nearest-neighbor search (O(log n) average vs QuadTree's
   potential O(n) for certain queries)
2. No capacity parameter to tune
3. Naturally balanced if built all at once (bulk loading)
4. Extends cleanly to k-dimensions (hence the name)

## Formal Analysis

### Data Structure

```
KDNode:
  - point: (x, y)  (or arbitrary k-dimensional vector)
  - axis: int (0 = x, 1 = y, ...)
  - left: KDNode (points with coord[axis] < point[axis])
  - right: KDNode (points with coord[axis] >= point[axis])
```

### Construction (Bulk Loading)

To build a balanced KD-Tree from n points:

1. Sort points by axis 0 (x). Pick median as root.
2. Recurse left half (axis = 1) and right half (axis = 1).
3. Alternate axis each level.

**Complexity: O(n log n).** Sorting at each level costs O(n log n) in
naive implementation. Median-finding in O(n) per level gives O(n log n)
total. The optimal construction is O(n log n).

**Space: O(n).** One node per point, plus the tree structure overhead.

### Nearest-Neighbor Search

Given a query point q, find the closest point in the tree.

Algorithm:
1. Traverse to leaf along the splitting axis, maintaining best-so-far.
2. On unwind, check if the other side of each split could contain a
   closer point (using the split distance).
3. If yes, recurse into the other side.

**Complexity:**
- **Best case: O(log n)** — query point is close to a point in the tree,
  and the pruning works perfectly.
- **Average case: O(log n)** — for uniformly distributed points and
  low dimensions (k ≤ 3).
- **Worst case: O(n)** — query is far from all points, and the pruning
  never eliminates subtrees. In practice, this is rare for k ≤ 3 but
  common for k ≥ 10 (the "curse of dimensionality").

*Proof sketch:* For a balanced KD-Tree in 2D, the pruning condition uses
the distance along the splitting axis. If this distance exceeds the current
best, the entire subtree is pruned. The probability of pruning increases
with tree depth and decreases with dimensionality.

### Range Query

Same as QuadTree: visit nodes whose bounding box intersects the range.

**O(k + log n)** average, **O(n)** worst case.

## Implementation Task

### Build `kdtree.py`

```python
class KDNode:
    def __init__(self, point, axis):
        self.point = point          # (x, y) tuple or numpy array
        self.axis = axis            # 0 for x, 1 for y
        self.left = None
        self.right = None
        self.boundary = None        # Optional: bounding box for range queries

class KDTree:
    def __init__(self, points):
        """Bulk-load points into a balanced KD-Tree."""
        self.root = self._build(points, depth=0)

    def _build(self, points, depth):
        if not points:
            return None
        axis = depth % 2  # 0 = x, 1 = y
        points.sort(key=lambda p: (p.x, p.y)[axis])
        median_idx = len(points) // 2
        node = KDNode(points[median_idx], axis)
        node.left = self._build(points[:median_idx], depth + 1)
        node.right = self._build(points[median_idx + 1:], depth + 1)
        return node

    def nearest(self, qx, qy):
        """Return (point, distance²) of nearest neighbor to (qx, qy)."""
        def _search(node, best, best_dist):
            if node is None:
                return best, best_dist
            
            # Current point
            dx = node.point.x - qx
            dy = node.point.y - qy
            d2 = dx*dx + dy*dy
            if d2 < best_dist:
                best = node.point
                best_dist = d2
            
            # Decide which side to explore first
            axis = node.axis
            diff = (qx - node.point.x) if axis == 0 else (qy - node.point.y)
            first = node.left if diff < 0 else node.right
            second = node.right if diff < 0 else node.left
            
            best, best_dist = _search(first, best, best_dist)
            
            # Check if we need to explore the other side
            if diff*diff < best_dist:
                best, best_dist = _search(second, best, best_dist)
            
            return best, best_dist

        best, d2 = _search(self.root, None, float('inf'))
        return best, math.sqrt(d2)

    def range_query(self, cx, cy, radius):
        """Return all points within radius of (cx, cy)."""
        found = []
        r2 = radius * radius
        
        def _query(node):
            if node is None:
                return
            dx = node.point.x - cx
            dy = node.point.y - cy
            if dx*dx + dy*dy <= r2:
                found.append(node.point)
            
            axis = node.axis
            diff = (cx - node.point.x) if axis == 0 else (cy - node.point.y)
            
            # Always check the primary side
            first = node.left if diff < 0 else node.right
            _query(first)
            
            # Check the other side if the split plane is within range
            if abs(diff) <= radius:
                second = node.right if diff < 0 else node.left
                _query(second)
        
        _query(self.root)
        return found
```

### Test Against QuadTree

Generate 10,000 random points. Compare:
1. Construction time (KD-Tree vs QuadTree)
2. Single nearest-neighbor query time (1,000 queries)
3. Range query time (1,000 queries, radius 500m)
4. Memory usage

### Visualize the KD-Tree Splits

Write a script that:
1. Generates 20 random points
2. Builds a KD-Tree
3. Draws the splitting planes (alternating x/y) on a matplotlib plot
4. Marks the points

Compare visually with the QuadTree grid from `game.py`.

## Check-In Questions

1. The KD-Tree above always splits on the median. What happens to the
   query time if you split at a random point instead?

2. Nearest-neighbor search uses `diff*diff < best_dist` to decide
   whether to explore the other side. Why is the threshold squared?

3. For a 2D KD-Tree, what is the maximum number of nodes visited in a
   nearest-neighbor search in the worst case? (This is a known result.)

4. The KD-Tree range query above checks both sides if `abs(diff) <=
   radius`. Could this be tightened? What condition would make it exact?

## Complexity Benchmark

```python
import time
import numpy as np
import random
from kdtree import KDTree
from game import QuadTree, Rect, Particle, WORLD_SIZE

for n in [100, 1000, 10000]:
    particles = [Particle(random.uniform(-4000, 4000),
                          random.uniform(-4000, 4000))
                 for _ in range(n)]
    
    # KD-Tree
    t0 = time.perf_counter()
    kd = KDTree(particles)
    t1 = time.perf_counter()
    
    # Nearest neighbor queries
    qx, qy = random.uniform(-4000, 4000), random.uniform(-4000, 4000)
    t2 = time.perf_counter()
    for _ in range(1000):
        kd.nearest(qx + random.uniform(-500, 500),
                   qy + random.uniform(-500, 500))
    t3 = time.perf_counter()
    
    print(f"KD-Tree n={n:6d} | build={(t1-t0)*1000:8.2f}ms | "
          f"NN queries={(t3-t2)*1000:8.2f}ms")
```

Compare build time with QuadTree insert time. The KD-Tree build is
O(n log n) once. QuadTree insert is O(n log n) total for n inserts.
Which is faster in practice?

## Connection Back

Your `game.py` QuadTree does range queries (mouse radar). The KD-Tree
does both range queries AND nearest-neighbor search. Nearest-neighbor
is useful for pathfinding (find the closest obstacle), clustering
(group nearby drones), and sensor fusion (match radar blips to known
tracks).

**Note on incremental inserts:** The KD-Tree above uses bulk loading
(all-at-once construction). If you need incremental inserts (drones
enter/leave dynamically), the tree becomes unbalanced. A solution is
to rebuild periodically, or use a more complex structure (scapegoat
tree, kd-tree with rebuilding).

**Next:** `04-comparison-spatial.md`
