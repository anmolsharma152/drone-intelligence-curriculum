# Optimizing for 10,000 Entities

## Learning Objective

Apply all optimizations from this phase to achieve 60fps with 10,000
simultaneous drones.

## The Target

```
10,000 entities
60 fps (16.67ms per frame)
Components: physics, spatial, collision, render, optional AI
```

## Optimization Checklist

### 1. Data Layout (Phase 06-02)

- [ ] Struct of Arrays (SoA) not Array of Structs (AoS)
- [ ] NumPy float32 arrays for all physics state
- [ ] Numba @njit for hot loops

### 2. Spatial Index (Phase 02 + 06-02)

- [ ] Uniform Grid (Numba-compatible) preferred over QuadTree
- [ ] Grid cell size tuned: cell_size = avg_query_radius × 2
- [ ] Pre-allocate grid arrays, no dynamic allocation

### 3. Collision (Phase 03)

- [ ] Coarse phase: grid cell check
- [ ] Fine phase: exact distance check
- [ ] Use spatial index, not O(N²)

### 4. Rendering (Phase 05)

- [ ] OpenGL instanced rendering (1 draw call for N drones)
- [ ] Or: reduce render frequency (only render every 3rd frame)
- [ ] Or: disable render during training

### 5. AI Inference (Phase 04)

- [ ] Batch inference (N observations → one tensor)
- [ ] Or: reduce frequency (AI update every 3 frames)
- [ ] Or: CPU inference with ONNX Runtime

### 6. Logging (Phase 04)

- [ ] Reduce log rate (every 10th frame during production)
- [ ] Binary format (already done in Phase 04-04)
- [ ] Async writer (I/O on separate thread)

## Benchmark Progression

Track improvement at each step:

```
Optimization Level                       | N=1000  | N=5000  | N=10000
-----------------------------------------+---------+---------+--------
Baseline (no optimization)               | ___ms   | ___ms   | ___ms
+ NumPy SoA data layout                  | ___ms   | ___ms   | ___ms
+ Numba @njit physics                    | ___ms   | ___ms   | ___ms
+ Uniform Grid (replaces QuadTree)       | ___ms   | ___ms   | ___ms
+ Parallel physics (prange)              | ___ms   | ___ms   | ___ms
+ Batch AI inference                     | ___ms   | ___ms   | ___ms
+ Reduce render to 20fps                 | ___ms   | ___ms   | ___ms
+ Final (all optimizations)              | ___ms   | ___ms   | ___ms
Goal: <16.67ms (60fps)                   | ✓/✗    | ✓/✗    | ✓/✗
```

## Profiling the Optimized Version

Run the profiler again after all optimizations:

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
  10000    1.2ms  ...      ...     ...     numba_physics.py:20(physics)
  10000    2.1ms  ...      ...     ...     grid.py:40(query)
  10000    0.5ms  ...      ...     ...     collision.py:30(check)
        (remaining time: rendering, logging, overhead)
```

## When Optimization Isn't Enough

If you still can't hit 60fps:

1. **C libraries:** Rewrite physics in C with pybind11
2. **GPU physics:** Use PyTorch or CUDA for physics stepping
3. **Distributed simulation:** Split entities across machines
4. **Simpler model:** Reduce 6-DOF to 3-DOF physics
5. **Lower precision:** Use float16 (half precision)

```python
# GPU-accelerated physics (PyTorch)
import torch

class GPUPhysics:
    def __init__(self, n):
        self.pos = torch.zeros((n, 3), dtype=torch.float32, device='cuda')
        self.vel = torch.zeros((n, 3), dtype=torch.float32, device='cuda')

    def step(self, forces, dt):
        self.vel += forces * dt
        self.pos += self.vel * dt
        # All operations are GPU-parallel
```

## Implementation Task

### 1. Create `run_10k.py`

A script that:

1. Creates 10,000 drones with random positions and velocities
2. Runs them with all optimizations applied
3. Reports achieved FPS every second
4. Visualizes with simple 2D projection (no heavy render)

### 2. Find Your Wall

Increase N until FPS drops below 30. What's your maximum entity
count? Report it.

```
Optimized max N at 30fps: ________
Optimized max N at 60fps: ________
```

## Check-In Questions

1. Amdahl's Law says the serial portion limits speedup. After
   optimization, what is the serial portion of your simulation?
   What limits further speedup?

2. Uniform Grid requires a fixed cell size. If the query radius
   varies (e.g., different sensor ranges), what grid cell size
   do you choose and what's the tradeoff?

3. Reducing render frequency to 20fps means the visual is 20fps
   while physics runs at 60fps. How would you interpolate between
   rendered frames for smooth visuals?

## 🤖 Connection

10,000 entities is beyond single-drone RL. It enables:
- **Swarm RL:** Train cooperative/competitive policies for 1000 drones
- **Environment diversity:** 10,000 different obstacle configurations
  running in parallel = extremely diverse training data
- **Population-based training:** Train 1000 policies simultaneously,
  each slightly different, evolving the best ones

**Next:** `05-distributed-rl-training.md`
