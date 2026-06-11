# Vectorization with Numba

## Learning Objective

Use Numba JIT compilation and NumPy vectorization to accelerate the
hot loops identified by profiling.

## Numba Basics

Numba compiles Python functions to machine code via LLVM:

```python
from numba import jit, njit, prange
import numpy as np

# Compile once, run at C speed
@njit
def physics_step(pos, vel, forces, dt):
    """Update positions and velocities for ALL entities."""
    for i in range(len(pos)):
        vel[i] += forces[i] * dt
        pos[i] += vel[i] * dt
    return pos, vel
```

### What Numba Can Accelerate

| Pattern | Numba? | Example |
|---|---|---|
| Loop over array | ✓ Excellent | Physics update |
| Math operations | ✓ Excellent | Distance, rotation |
| String/objects | ✗ No | Python objects |
| Dynamic lists | ⚠️ Limited | Use arrays instead |
| PyTorch tensors | ✗ No | Use PyTorch directly |

## Vectorizing the Physics

### Before (Python loop):

```python
def update_all(drones, dt):
    for drone in drones:
        drone.pos[0] += drone.vel[0] * dt
        drone.pos[1] += drone.vel[1] * dt
        drone.vel[0] += drone.acc[0] * dt
        drone.vel[1] += drone.acc[1] * dt
```

### After (Numba + Array of Structs → Struct of Arrays):

```python
# Use arrays instead of object list
pos_x = np.zeros(N, dtype=np.float32)
pos_y = np.zeros(N, dtype=np.float32)
vel_x = np.zeros(N, dtype=np.float32)
vel_y = np.zeros(N, dtype=np.float32)
acc_x = np.zeros(N, dtype=np.float32)
acc_y = np.zeros(N, dtype=np.float32)

@njit
def update_all_numba(pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, dt):
    for i in range(len(pos_x)):
        vel_x[i] += acc_x[i] * dt
        vel_y[i] += acc_y[i] * dt
        pos_x[i] += vel_x[i] * dt
        pos_y[i] += vel_y[i] * dt
```

## Spatial Index with Numba

The QuadTree is harder to Numba-ify because of dynamic allocation.
The Grid (uniform grid) is much easier:

```python
@njit
def grid_insert(points, cell_size, grid_size):
    """Insert N points into a fixed grid. Returns cell assignments."""
    cells_x = points[:, 0] // cell_size + grid_size // 2
    cells_y = points[:, 1] // cell_size + grid_size // 2
    # Clip to bounds
    cells_x = np.clip(cells_x, 0, grid_size - 1)
    cells_y = np.clip(cells_y, 0, grid_size - 1)
    return cells_x, cells_y

@njit
def grid_query(cell_x, cell_y, query_x, query_y, radius,
               cells_x, cells_y, points):
    """Query points within radius using grid."""
    results = []
    cell_radius = int(radius // cell_size) + 1
    for dx in range(-cell_radius, cell_radius + 1):
        for dy in range(-cell_radius, cell_radius + 1):
            cx = cell_x + dx
            cy = cell_y + dy
            if 0 <= cx < grid_size and 0 <= cy < grid_size:
                # Check all points in this cell
                mask = (cells_x == cx) & (cells_y == cy)
                # ... (in practice, use parallel arrays with cell start indices)
    return results
```

## Benchmark

| Implementation | N=1000 | N=10000 | N=100000 |
|---|---|---|---|
| Python loop | 2.1ms | 21ms | 210ms |
| NumPy vectorized | 0.3ms | 2.8ms | 28ms |
| Numba @njit | 0.08ms | 0.7ms | 7ms |
| Speedup (Numba vs Python) | 26× | 30× | 30× |

## Implementation Task

### 1. Build `numba_physics.py`

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def physics_step_parallel(pos, vel, acc, dt):
    """Parallel Numba loop using all CPU cores."""
    for i in prange(len(pos)):
        vel[i] += acc[i] * dt
        pos[i] += vel[i] * dt
    return pos, vel

@njit
def squared_distances(x1, y1, x2, y2):
    """Compute all-pairs squared distances between two sets of points."""
    n1, n2 = len(x1), len(x2)
    dists = np.empty((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dists[i, j] = dx*dx + dy*dy
    return dists
```

### 2. Migrate from Object Lists to Arrays

Refactor the simulation to use parallel arrays instead of lists of
drone objects:

```
Before: drones = [Drone(), Drone(), ...]
After:  pos = np.zeros((N, 3), dtype=f32)
        vel = np.zeros((N, 3), dtype=f32)
        euler = np.zeros((N, 3), dtype=f32)
        ...
```

### 3. Benchmark Every Step

```
Step                 | Before (ms) | After (ms) | Speedup
---------------------+-------------+------------+--------
Physics (per frame)  |    ___      |    ___     |  ___
Spatial insert       |    ___      |    ___     |  ___
Spatial query        |    ___      |    ___     |  ___
Collision check      |    ___      |    ___     |  ___
```

## Check-In Questions

1. Numba's `@njit` compiles at call time. What triggers recompilation?
   How can you avoid paying the compilation cost in every run?

2. `prange` is Numba's parallel loop. When does it help? When does it
   hurt (due to overhead)?

3. The physics update `pos += vel * dt` is a trivially parallelizable
   operation. Why doesn't the Python version use multiple cores?

4. The QuadTree uses dynamic allocation (subdividing nodes) which
   Numba can't compile. The Grid uses fixed allocation. What memory
   tradeoff does the Grid make?

## 🤖 Connection

Numba accelerates the simulation, which directly accelerates RL
training. A 30× speedup means the same training runs in 1/30th the
time — or you can train with 30× more data.

For distributed RL (file 05), each parallel environment runs the
Numba-accelerated sim, giving 30× more experience per second.

**Next:** `03-parallel-sim-multiprocessing.md`
