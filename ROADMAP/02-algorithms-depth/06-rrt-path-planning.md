# RRT and RRT* Path Planning

## Learning Objective

Implement Rapidly-exploring Random Trees (RRT) and RRT*, the standard
algorithms for continuous-space path planning with non-holonomic
constraints.

## Motivation

A* discretizes space into a grid. This works well when:
- The grid resolution matches the problem scale
- Movement is axis-aligned or 8-directional
- Obstacles align with grid cells

For drone flight, these assumptions fail:
- Drones move in continuous space
- Drones have turning radius, acceleration limits (non-holonomic)
- Obstacles are arbitrary shapes

RRT handles continuous space natively and can incorporate complex
constraints.

## The Algorithm

### RRT (Basic Version)

```
function RRT(start, goal, obstacle_map, max_iterations):
    tree = Tree(start)
    
    for i in range(max_iterations):
        sample = random_sample()          // Random point in space
        nearest = tree.nearest(sample)    // Find closest node
        new_node = steer(nearest, sample) // Move toward sample
        if not collision(nearest, new_node, obstacle_map):
            tree.add_node(new_node)
            tree.add_edge(nearest, new_node)
            if distance(new_node, goal) < goal_threshold:
                return extract_path(tree, new_node)
    
    return failure
```

### RRT* (Optimal Version)

RRT* adds two key improvements over RRT:

1. **Rewiring:** After adding a new node, check nearby nodes to see if
   the new node offers a cheaper path. If so, reparent them.

2. **Nearby selection:** Instead of connecting to the nearest node,
   find all nodes within a radius r and connect to the one that
   minimizes path cost.

```
function RRT*(start, goal, obstacle_map, max_iterations):
    tree = Tree(start)
    
    for i in range(max_iterations):
        sample = random_sample()
        nearest = tree.nearest(sample)
        new_node = steer(nearest, sample)
        if not collision(nearest, new_node, obstacle_map):
            near_nodes = tree.near(new_node, radius(i))
            // Choose best parent
            for node in near_nodes:
                if not collision(node, new_node, obstacle_map):
                    cost = tree.cost(node) + distance(node, new_node)
                    if cost < tree.cost(new_node):
                        tree.reparent(new_node, node)
            tree.add_node(new_node)
            
            // Rewire existing nodes
            for node in near_nodes:
                if node != new_node.parent:
                    cost = tree.cost(new_node) + distance(new_node, node)
                    if cost < tree.cost(node) and 
                       not collision(new_node, node, obstacle_map):
                        tree.reparent(node, new_node)
            
            if distance(new_node, goal) < goal_threshold:
                return extract_path(tree, new_node)
    
    return failure
```

### The `steer` Function

The steering function moves from one point toward another by a fixed
step size:

```python
def steer(from_point, to_point, step_size=50.0):
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < step_size:
        return to_point
    ratio = step_size / dist
    return (from_point[0] + dx * ratio,
            from_point[1] + dy * ratio)
```

### The `near` Radius (RRT*)

The search radius for RRT* should decrease with iteration count:

```
r(i) = γ * (log(i+1) / (i+1))^(1/d)
```

Where γ is a constant (typically 2-5x the step size) and d is the
dimension (2 for our case).

### Complexity

**RRT:**
- **Per iteration:** O(n) where n = number of tree nodes (nearest
  neighbor search with KD-Tree: O(log n))
- **Total:** O(m · n) for m iterations with linear search, or
  O(m log n) with KD-Tree acceleration
- **No optimality guarantee** — RRT finds a path, not necessarily
  the shortest one

**RRT*:**
- **Per iteration:** O(n log n) due to near-neighbor search and
  rewiring
- **Total:** O(m · n log n)
- **Probabilistically optimal** — as m → ∞, RRT* converges to the
  optimal path

**Space: O(n)** for the tree.

## Implementation Task

### Build `rrt_planner.py`

```python
import math
import random
import numpy as np
from kdtree import KDTree  # For nearest-neighbor acceleration

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRT:
    def __init__(self, start, goal, obstacle_map,
                 step_size=50.0, goal_threshold=30.0,
                 max_iterations=10000):
        self.start = RRTNode(*start)
        self.goal = RRTNode(*goal)
        self.obstacle_map = obstacle_map
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations

        self.nodes = [self.start]
        self.kdtree = None  # Will store KD-Tree for fast NN queries

    def _rebuild_kdtree(self):
        """Rebuild KD-Tree from all nodes. Called periodically."""
        # Build from self.nodes
        pass

    def random_sample(self):
        """Sample with goal bias."""
        if random.random() < 0.05:  # 5% goal bias
            return (self.goal.x, self.goal.y)
        return (random.uniform(-4000, 4000),
                random.uniform(-4000, 4000))

    def nearest(self, point):
        """Find nearest node to point using KD-Tree or linear scan."""
        best = None
        best_dist = float('inf')
        for node in self.nodes:
            dx = node.x - point[0]
            dy = node.y - point[1]
            d2 = dx*dx + dy*dy
            if d2 < best_dist:
                best_dist = d2
                best = node
        return best

    def steer(self, from_node, to_point):
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < self.step_size:
            return RRTNode(to_point[0], to_point[1])
        ratio = self.step_size / dist
        return RRTNode(from_node.x + dx * ratio,
                       from_node.y + dy * ratio)

    def collision_free(self, from_node, to_node):
        """Check if the straight-line path is obstacle-free."""
        # Sample along the line
        steps = max(2, int(math.sqrt(
            (to_node.x - from_node.x)**2 +
            (to_node.y - from_node.y)**2) / 10))
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + (to_node.x - from_node.x) * t
            y = from_node.y + (to_node.y - from_node.y) * t
            if self.obstacle_map(x, y):
                return False
        return True

    def plan(self):
        for _ in range(self.max_iterations):
            sample = self.random_sample()
            nearest = self.nearest(sample)
            new_node = self.steer(nearest, sample)

            if not self._contains(new_node) and \
               self.collision_free(nearest, new_node):
                new_node.parent = nearest
                new_node.cost = nearest.cost + math.sqrt(
                    (new_node.x - nearest.x)**2 +
                    (new_node.y - nearest.y)**2)
                self.nodes.append(new_node)

                # Check for goal
                dx = new_node.x - self.goal.x
                dy = new_node.y - self.goal.y
                if math.sqrt(dx*dx + dy*dy) < self.goal_threshold:
                    self.goal.parent = new_node
                    return self.extract_path()

        return None  # Failed to find path

    def extract_path(self):
        path = []
        node = self.goal
        while node:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()
        return path

    def _contains(self, node, tolerance=1.0):
        for n in self.nodes:
            if abs(n.x - node.x) < tolerance and \
               abs(n.y - node.y) < tolerance:
                return True
        return False
```

### Integration: Drone Path Planning

1. Define an obstacle map function that returns True if (x, y) is in
   an obstacle
2. Place 10-20 circular obstacles in the 8000m world
3. Run RRT from (-3500, -3500) to (3500, 3500)
4. Visualize the tree and resulting path using matplotlib

### Upgrade to RRT*

Implement the full RRT* with rewiring. Compare RRT vs RRT*:

| Metric | RRT | RRT* |
|---|---|---|
| Path length | _____ | _____ |
| Time to find first path | _____ | _____ |
| Time to converge | N/A | _____ |
| Nodes explored | _____ | _____ |

## Parameter Tuning

Vary the step_size parameter: 10, 50, 100, 200.

| Step Size | Path Length | Iterations to Goal | Success Rate |
|---|---|---|---|
| 10 | _____ | _____ | _____ |
| 50 | _____ | _____ | _____ |
| 100 | _____ | _____ | _____ |
| 200 | _____ | _____ | _____ |

What's the tradeoff?

## Check-In Questions

1. Goal bias (biasing samples toward the goal at 5% probability) helps
   RRT converge faster. What's the danger of setting it too high (e.g.,
   50%)?

2. RRT* requires nearest-neighbor search in a radius. If implemented
   with a linear scan, what is the complexity per iteration? At what
   N does this become slower than using a KD-Tree?

3. The collision check samples points along the line. What happens if
   the step size between samples is too large? Too small?

4. RRT is probabilistically complete — as iterations → ∞, the
   probability of finding a path (if one exists) → 1. Why is it only
   "probabilistically" complete and not guaranteed?

## Connection Back

You now have:
- Spatial indexing (QuadTree, Grid, KD-Tree) for fast queries
- Pathfinding (A*) for discrete grids
- Path planning (RRT) for continuous space

Your drone simulation can now navigate obstacle-filled worlds. In the
next phase, you'll give the drone realistic physics and control systems.

**Next:** `GATE-01.md`
