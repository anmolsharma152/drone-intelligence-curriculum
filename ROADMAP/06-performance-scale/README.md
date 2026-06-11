# Phase 06: Performance, Scale & Sim-to-Real

## Overview

The final phase. Take everything you've built and make it fast, then
make it real.

```
Performance:            Scale:
  Profile bottlenecks    Multi-process sim
  Vectorize with Numba   10,000 entities
  GPU acceleration       Distributed RL

Sim-to-Real (🤖):
  Domain randomization
  Policy distillation for edge
  Real-world deployment path
```

## Files

| # | File | Description | AI? |
|---|---|---|---|
| 01 | `01-profiling-bottlenecks.md` | Profile, identify bottlenecks | |
| 02 | `02-vectorization-numba.md` | Numba JIT, SIMD, GPU | |
| 03 | `03-parallel-sim-multiprocessing.md` | Multi-process parallel sim | |
| 04 | `04-10k-entities-optimization.md` | Optimize for 10,000 entities | |
| 05 | `05-distributed-rl-training.md` | 🤖 Parallel envs, distributed PPO | ✓ |
| 06 | `06-sim-to-real-transfer.md` | 🤖 Domain randomization, deployment | ✓ |
| — | `GATE-05.md` | Final verification gate | |

## Learning Objectives

1. Profile Python code and identify bottlenecks systematically
2. Accelerate hot loops with Numba JIT compilation
3. Parallelize simulation across CPU cores
4. Achieve 60fps with 10,000 entities
5. **Scale RL training to 100+ parallel environments** 🤖
6. **Bridge sim-to-real gap for real hardware deployment** 🤖

## Prerequisites

- All prior phases completed (gates passed)
- numba (`pip install numba`)
- multiprocessing (stdlib)
- Optional: CUDA toolkit for GPU acceleration

**Next:** `01-profiling-bottlenecks.md`
