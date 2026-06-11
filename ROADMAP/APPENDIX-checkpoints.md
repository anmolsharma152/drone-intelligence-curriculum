# Checkpoints & Gates

Quick reference for verifying your progress through the curriculum.

## Phase 01: Foundations Review

**Gate: GATE-00.md**

Must complete:
- [ ] Read and understand all existing code (main.py, main2.py, game.py)
- [ ] Big-O analysis written for each component
- [ ] Both bugs identified:
  - Out-of-bounds freeze (position silently goes stale)
  - Diagonal speed (141 m/s vs intended 100 m/s)
- [ ] Fixes applied and verified

**Pass criteria:** Can explain the complexity of every loop in game.py.

---

## Phase 02: Algorithms & Data Structures

**Gate: GATE-01.md**

Must complete:
- [ ] QuadTree implemented and benchmarked (from game.py)
- [ ] Uniform Grid implemented
- [ ] KD-Tree implemented with NN + range query
- [ ] All three produce identical query results
- [ ] `bench_spatial.py` produces log-log plots matching Big-O
- [ ] A* implemented with 3+ heuristics
- [ ] RRT planner implemented

**Pass criteria:** Results table filled out, log-log plot saved to `plots/`.

---

## Phase 03: Physics & RL Environment

**Gate: GATE-02.md**

Must complete:
- [ ] 6-DOF rigid body with Euler + RK4
- [ ] Cascaded PID with Ziegler-Nichols tuning
- [ ] Kalman filter (1D minimum, 12D EKF bonus)
- [ ] Collision detection + avoidance
- [ ] Wind field + obstacle course generator
- [ ] Gym-compatible drone environment
- [ ] Reward shaping (3+ variants)
- [ ] PPO training from scratch — policy beats random baseline

**Pass criteria:** Trained policy achieves > 60% success rate,
mean reward > 0, PID vs RL comparison table filled.

---

## Phase 04: Systems Engineering & ML Pipeline

**Gate: GATE-03.md**

Must complete:
- [ ] Multi-drone architecture (N processes, ground station)
- [ ] ECS refactor with 4+ components, 3+ systems
- [ ] UDP binary protocol with checksums
- [ ] Telemetry logging with replay
- [ ] Feature engineering pipeline (15+ derived features)
- [ ] Inference server with batching + versioning
- [ ] Online learning loop (experience buffer + PPO update)

**Pass criteria:** Multi-drone swarm runs, inference benchmark done,
online learning shows improvement over static policy.

---

## Phase 05: Visualization & Policy Insight

**Gate: GATE-04.md**

Must complete:
- [ ] 3D rendering with ModernGL (drone + obstacle meshes, camera)
- [ ] Live telemetry dashboard (altitude, speed, attitude plots)
- [ ] Policy visualizations:
  - Value heatmap
  - Action distribution
  - Saliency map
  - Failure mode analysis
  - Rollout video

**Pass criteria:** All visualizations saved to `plots/`, analysis report
written.

---

## Phase 06: Performance, Scale & Sim-to-Real

**Gate: GATE-05.md**

Must complete:
- [ ] Profiling with cProfile + flamegraph
- [ ] Numba-accelerated physics (SoA data layout)
- [ ] Parallel simulation (N workers)
- [ ] 10,000 entities at 60fps (or max N documented)
- [ ] Distributed PPO (shared-memory, N workers)
- [ ] Domain randomization (mass, noise, wind, obstacles)
- [ ] Policy distillation + ONNX export

**Pass criteria:** Optimization progression table filled, distributed
training measured, sim-to-real checklist completed.

---

## Full Curriculum Checklist

```
Phase 01: Foundations Review    [PASSED] / [NOT YET]
Phase 02: Algorithms            [PASSED] / [NOT YET]
Phase 03: Physics + RL          [PASSED] / [NOT YET]
Phase 04: Systems + ML Pipeline [PASSED] / [NOT YET]
Phase 05: Visualization         [PASSED] / [NOT YET]
Phase 06: Performance + Deploy  [PASSED] / [NOT YET]
```
