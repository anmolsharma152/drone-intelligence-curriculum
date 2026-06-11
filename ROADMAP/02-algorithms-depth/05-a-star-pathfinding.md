# A* Pathfinding

## Learning Objective

Implement the A* search algorithm on a 2D grid. Understand optimality
conditions, heuristic design, and when A* is the right tool.

## Motivation

Spatial indexing answers "what is near me?". Pathfinding answers "how
do I get from here to there?".

Your drones currently fly diagonally in a straight line or bounce around
randomly. Real drones navigate around obstacles, follow waypoints, and
find efficient paths. A* is the workhorse algorithm for grid-based
pathfinding.

## The Algorithm

A* is a best-first search that uses a cost function:

```
f(n) = g(n) + h(n)
```

Where:
- `g(n)` = actual cost from start to node n
- `h(n)` = estimated cost from node n to goal (heuristic)
- `f(n)` = total estimated cost through node n

### Pseudocode

```
function A*(start, goal, h):
    open_set = PriorityQueue()
    open_set.add(start, h(start))
    came_from = {}
    g_score = {start: 0}

    while not open_set.empty():
        current = open_set.pop_min()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(current):
            tentative_g = g_score[current] + cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, ∞):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + h(neighbor, goal)
                if neighbor in open_set:
                    open_set.decrease_key(neighbor, f_score)
                else:
                    open_set.add(neighbor, f_score)

    return failure  # No path exists
```

### Optimality Condition

A* is **admissible** (guarantees the shortest path) if the heuristic
h(n) is **admissible** — it never overestimates the true cost to the goal.

A* is **consistent** (monotone) if h(n) ≤ cost(n, n') + h(n') for all
neighbors n' of n. Consistent heuristics guarantee the first time A*
expands a node, it has found the optimal path to that node.

**Key insight:** The heuristic guides the search. A perfect heuristic
(h = true cost) makes A* expand only nodes on the optimal path. A
trivial heuristic (h = 0) makes A* identical to Dijkstra's algorithm.

### Heuristics for Grids

| Heuristic | Formula | Admissible? | Notes |
|---|---|---|---|
| Manhattan | `|dx| + |dy|` | Yes (4-dir movement) | Overestimates for diagonal |
| Diagonal | `max(|dx|, |dy|)` | Yes (8-dir movement) | Chebyshev distance |
| Euclidean | `sqrt(dx² + dy²)` | Yes (any direction) | More accurate, slower |
| Octile | `max(|dx|,|dy|) + (√2-1)*min(|dx|,|dy|)` | Yes (8-dir movement) | Best for 8-directional |

### Complexity

**Time:**
- **Best case:** O(b^d) where b = branching factor, d = solution depth.
  In practice, a good heuristic reduces this drastically.
- **Worst case:** Same as Dijkstra: O(E + V log V) on a graph with
  V vertices and E edges.
- With a perfect heuristic: O(d) — expands only the optimal path nodes.

**Space:**
- **O(V)** — stores g_score, came_from, and the open set for all
  visited nodes. This is A*'s main weakness for large grids.

## Implementation Task

### Build `astar.py`

```python
import heapq
import math

class Grid:
    """2D grid with obstacles."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = set()

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x, y):
        return (x, y) not in self.obstacles

    def neighbors(self, x, y):
        """8-directional neighbors."""
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and self.passable(nx, ny):
                    cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                    yield (nx, ny), cost

def heuristic(a, b, method="octile"):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if method == "manhattan":
        return dx + dy
    elif method == "euclidean":
        return math.sqrt(dx*dx + dy*dy)
    elif method == "octile":
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
    elif method == "chebyshev":
        return max(dx, dy)

def astar(grid, start, goal, heuristic_method="octile"):
    """Returns (path, explored_count, path_cost)."""
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, heuristic_method)}
    explored = 0

    while open_set:
        current = heapq.heappop(open_set)[1]
        explored += 1

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, explored, g_score[goal]

        for neighbor, cost in grid.neighbors(*current):
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal, heuristic_method)
                heapq.heappush(open_set, (f, neighbor))

    return None, explored, None  # No path
```

### Integration: Obstacle Map from QuadTree

Connect your QuadTree/grid to A*:

1. Place 20 random circular obstacles (radius 100m) in the 8km world
2. Convert the world to a 2D grid (cell size = 50m → 160×160 grid)
3. Mark cells as obstacles if they intersect any obstacle circle
4. Place a drone at (-3000, -3000) and a goal at (3000, 3000)
5. Run A* and visualize the path

### Visualization

Create `visualize_path.py` using matplotlib:

1. Draw the grid
2. Mark obstacles in black
3. Draw the A* path in red
4. Mark start (green) and goal (blue)
5. Overlay the explored nodes (faded red)

## Analysis Tasks

### Heuristic Comparison

Run A* with all four heuristics on the same 10 random obstacle maps.
Record for each:

| Heuristic | Path Length | Nodes Explored | Time (ms) |
|---|---|---|---|
| Manhattan | _____ | _____ | _____ |
| Chebyshev | _____ | _____ | _____ |
| Euclidean | _____ | _____ | _____ |
| Octile | _____ | _____ | _____ |

- Which heuristic explores the fewest nodes?
- Are any paths non-optimal? Why?

### Grid Resolution

Vary the grid cell size from 10m to 500m. Record:

| Cell Size | Grid Dims | Path Cost | Time (ms) | Memory |
|---|---|---|---|---|
| 10m | 800×800 | _____ | _____ | _____ |
| 50m | 160×160 | _____ | _____ | _____ |
| 100m | 80×80 | _____ | _____ | _____ |
| 500m | 16×16 | _____ | _____ | _____ |

- What happens to path quality at coarse resolution?
- What happens to search time at fine resolution?

## Check-In Questions

1. Is the Manhattan heuristic admissible for 8-directional movement?
   Prove or disprove.

2. The open set uses a binary heap (heapq). What is the complexity of
   each heap operation? Does `decrease_key` exist in heapq? (Hint: it
   doesn't — what does our implementation do instead, and what's the
   cost?)

3. For a 160×160 grid (25,600 cells), what is the worst-case memory
   usage of A* in bytes? (Assume Python dict overhead of ~72 bytes
   per entry plus two floats per storage.)

4. If you have a grid with no obstacles, what path does A* return?
   How many nodes does it explore with the Octile heuristic?

## Connection Back

Your drone simulation currently has no pathfinding — drones fly in
straight lines. After this module, you can place obstacles in the
world and have drones navigate around them using A*.

**Next:** `06-rrt-path-planning.md`
