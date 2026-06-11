# Uniform Grid Spatial Hash

## Learning Objective

Build an alternative spatial indexing structure — the uniform grid — and
understand the engineering tradeoffs between grid and QuadTree.

## Motivation

QuadTrees are elegant but have overhead: pointer chasing, unpredictable
memory access, worst-case degeneration. A uniform grid trades memory for
simplicity and cache friendliness.

The core idea: partition space into a fixed grid of cells. Each cell
contains a list of points. To query, compute which cells overlap the
search range and iterate their contents.

## Formal Analysis

### Data Structure

```
Grid:
  - cell_size: float (side length of one cell)
  - num_cells_x, num_cells_y: int
  - cells: list[list[list[Point]]]  (or dict of cell coord -> list)
  - world_bounds: (min_x, min_y, max_x, max_y)
```

For a world of size W (8000m) and cell size c, the grid has
(W/c) × (W/c) cells.

### Cell Index Computation

```python
def _cell_coord(self, x, y):
    ix = int((x - self.min_x) // self.cell_size)
    iy = int((y - self.min_y) // self.cell_size)
    return (ix, iy)
```

O(1) — two subtractions, two divisions, two integer casts.

### Insert Complexity

**O(1).** Compute cell coordinates, append to the cell's list. No
subdivision, no tree traversal, no recursion.

### Query Complexity

**Best case: O(k)** — k = number of returned points. Query range falls
entirely within one cell. Compute cell coordinates, iterate its list.

**Average case: O(c²ρ + k)** — query range overlaps (r/c + 1)² cells
where r = query range radius, c = cell size, ρ = average density per cell
(= n * c² / W²). For a well-chosen cell size, this is close to O(k).

**Worst case: O(n)** — query range covers the entire world. Every cell
is visited and every point checked.

### Space Complexity

**O(n + (W/c)²)** — n points stored once each, plus (W/c)² cell buckets.
For c = 100m and W = 8000m, that's 80×80 = 6400 cells — negligible.

The grid trades memory for speed: more cells = less overlap, but more
memory and more cells to check for large queries.

## Tradeoff: Grid vs QuadTree

| Property | Grid | QuadTree |
|---|---|---|
| Insert | O(1) | O(log n) avg, O(n) worst |
| Query (point) | O(1) | O(log n) |
| Query (range) | O(cells × density) | O(log n + k) |
| Memory | O(n + fixed grid) | O(n) |
| Cache locality | Good (array of lists) | Poor (pointer-heavy) |
| Adaptive | No (fixed resolution) | Yes (subdivides dense areas) |
| Implementation | ~20 lines | ~80 lines |

**When Grid wins:**
- Dense, uniform distributions
- Fixed cell size matches query size well
- Hardware-friendly (predictable memory access)
- Simple to parallelize (cells are independent)

**When QuadTree wins:**
- Highly non-uniform distributions
- Varying query sizes
- Sparse data (no wasted cells in empty regions)

## Implementation Task

### Build `spatial_grid.py`

Implement a `SpatialGrid` class:

```python
class SpatialGrid:
    def __init__(self, cell_size, world_size=8000.0):
        self.cell_size = cell_size
        self.half_world = world_size / 2
        self.num_cells = int(world_size // cell_size) + 1
        self.cells = [[[] for _ in range(self.num_cells)]
                      for _ in range(self.num_cells)]

    def _cell(self, x, y):
        ix = int((x + self.half_world) // self.cell_size)
        iy = int((y + self.half_world) // self.cell_size)
        if 0 <= ix < self.num_cells and 0 <= iy < self.num_cells:
            return (ix, iy)
        return None

    def insert(self, particle):
        cell = self._cell(particle.x, particle.y)
        if cell:
            self.cells[cell[0]][cell[1]].append(particle)

    def query(self, cx, cy, radius):
        """Return all particles within radius of (cx, cy)"""
        # Determine cell range
        min_ix = max(0, int((cx - radius + self.half_world) // self.cell_size))
        max_ix = min(self.num_cells - 1,
                     int((cx + radius + self.half_world) // self.cell_size))
        min_iy = max(0, int((cy - radius + self.half_world) // self.cell_size))
        max_iy = min(self.num_cells - 1,
                     int((cy + radius + self.half_world) // self.cell_size))

        found = []
        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                for p in self.cells[ix][iy]:
                    # Optional: precise distance filter
                    dx = p.x - cx
                    dy = p.y - cy
                    if dx*dx + dy*dy <= radius*radius:
                        found.append(p)
        return found
```

### Integration: Replace QuadTree with Grid in `game.py`

Copy `game.py` to `game_grid.py` and replace the QuadTree with your
`SpatialGrid`. Compare the frame rates.

### Cell Size Tuning

Vary cell_size from 10m to 2000m and measure:
- Insert time (10,000 particles)
- Query time (1000 random 500m-radius queries)
- Memory usage

Plot the results. Where is the sweet spot? (Hint: think about the
cell_size relative to your typical query radius of 500m.)

## Check-In Questions

1. If the query radius is 500m and cell_size is 100m, how many cells
   must be checked in the worst case? (Assume the query is not near
   the world boundary.)

2. A cell_size of 1m would make queries very precise but slow. Why?
   Quantify: for W = 8000m, how many cells does a 500m-radius query
   touch?

3. The grid uses a 2D list of lists. How would you implement it with
   a dict instead? When would that be better?

4. The `query` method above applies a precise distance filter inside
   each cell. Is this necessary? Under what conditions could you skip
   it?

## Complexity Benchmark

Write `bench_grid_vs_quadtree.py`. For each N in [100, 1000, 10000]:

1. Generate N uniformly random particles
2. Insert all into both Grid and QuadTree
3. Run 1000 queries of radius 500m at random positions
4. Print: N, Grid insert time, QuadTree insert time, Grid query time,
   QuadTree query time, Grid memory (approx), QuadTree depth

## Connection Back

Your `game.py` QuadTree is adaptive — it naturally allocates more
resolution where points cluster. A uniform grid does not adapt. For
the particle simulation with uniform random positions, the grid will
likely be faster. For a simulation with dense clusters (e.g., drones
gathered at a waypoint), the QuadTree may win.

**Next:** `03-kd-tree.md`
