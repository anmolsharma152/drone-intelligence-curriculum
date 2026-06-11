# Gate 05: Performance, Scale & Sim-to-Real

Complete all of the following to finish the roadmap.

## 1. Profiling

- [ ] Profiled at N = 100, 1000, 5000, 10000
- [ ] Flamegraph generated (`flamegraph.svg`)
- [ ] Top 3 bottlenecks identified by name

Bottlenecks identified:
```
1. _________________ taking ___% of frame time
2. _________________ taking ___% of frame time
3. _________________ taking ___% of frame time
```

## 2. Vectorization

- [ ] `numba_physics.py` — Numba-accelerated physics
- [ ] Struct-of-Arrays data layout replaces object lists
- [ ] `prange` used for parallel loops
- [ ] Benchmark recorded:

```
Step           | Before (ms) | After (ms) | Speedup
Physics        |    ___      |    ___     |   ___×
Spatial insert |    ___      |    ___     |   ___×
Collision      |    ___      |    ___     |   ___×
```

## 3. Parallel Simulation

- [ ] `parallel_env.py` — N worker processes running sims
- [ ] Sync mode working
- [ ] Async mode working (bonus)

```
N Workers | Steps/sec | Speedup vs single
    1     |    ___    |    1.0×
    2     |    ___    |    ___×
    4     |    ___    |    ___×
    8     |    ___    |    ___×
```

## 4. 10,000 Entities

- [ ] `run_10k.py` achieves 60fps with 10,000 entities
- [ ] Or: max entity count at 30fps recorded

```
Max N at 60fps: ________
Max N at 30fps: ________
```

Optimization progression table filled out.

## 5. 🤖 Distributed RL Training

- [ ] `distributed_ppo.py` — Shared-memory distributed PPO
- [ ] At least 2 workers running in parallel
- [ ] Training throughput improves with more workers
- [ ] Policy converges to comparable reward as single-env training

```
N Workers | Steps/sec | Wall time to 1M steps
    1     |    ___    |        ___
    2     |    ___    |        ___
    4     |    ___    |        ___
    8     |    ___    |        ___
```

## 6. 🤖 Sim-to-Real

- [ ] Domain randomization implemented (mass, noise, wind, obstacles)
- [ ] Policy trained with randomization generalizes better
  (test on unseen parameter combinations)
- [ ] Policy distilled to smaller model
- [ ] Model exported to ONNX or TorchScript

```
Teacher network: ____ params, ____ KB
Student network:  ____ params, ____ KB
Performance drop: ____% reward change
```

## 7. Final Understanding Check

Answer these without looking at the files:

1. Draw the full system architecture from Phase 01 to Phase 06. Show
   how data flows from physics simulation → spatial index → feature
   pipeline → RL agent → action → physics.

2. What's the bottleneck in your current simulation? How would you
   further optimize it beyond what you've done?

3. A policy trained in simulation with no domain randomization fails
   on real hardware. Name 5 things that are different between sim
   and real, and for each, describe how domain randomization or
   another technique handles it.

4. You have 10,000 drones. Each runs a neural network policy at 60Hz.
   The network is small (32 hidden units, 4 outputs). How many CPU
   cores are needed? How would the answer change if you used a GPU?

5. What's the hardest problem you haven't solved yet? (Be honest.)

## Reflect: What You've Built

```
Phase 01: Fixed bugs, understand existing code's complexity
Phase 02: QuadTree, Grid, KD-Tree, A*, RRT with benchmarks
Phase 03: 6-DOF physics, PID, sensors, Kalman, collision,
          Gym env, reward shaping, PPO from scratch ✓
Phase 04: Multi-drone, ECS, UDP protocol, telemetry logging,
          feature pipeline, inference server, online learning ✓
Phase 05: 3D rendering, live dashboard, policy visualization ✓
Phase 06: Profiling, Numba, parallel sims, 10k entities,
          distributed PPO, sim-to-real transfer ✓
```

---

**Gate status:** `[COMPLETED]` / `[NOT YET]`

Congratulations on reaching the end.

No next — you've completed the roadmap.
