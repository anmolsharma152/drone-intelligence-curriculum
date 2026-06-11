# Phase 03: Physics Simulation & RL Environment

## Overview

This phase has two parallel tracks that feed into each other:

```
Physics Track:           RL Track (🤖):
  6-DOF kinematics  ──►  State space definition
  PID controller    ──►  Action space definition
  Sensor noise      ──►  Observation space
  Collision/Avoid   ──►  Reward shaping
  Wind & env        ──►  Environment dynamics
       │                        │
       └────────┬───────────────┘
                ▼
         Gym Environment
                │
                ▼
         Train RL Agent (PPO/policy gradient)
```

Every physics component you build is immediately wrapped into the RL
observation/action space. By the end of this phase, you will have a
drone that learns to fly via reinforcement learning.

## Files

| # | File | Description | AI? |
|---|---|---|---|
| 01 | `01-6dof-kinematics.md` | 6-DOF rigid body kinematics, Euler integration | |
| 02 | `02-pid-control.md` | PID altitude and velocity controllers | |
| 03 | `03-sensor-noise-filtering.md` | GPS/IMU noise, Kalman filter intro | |
| 04 | `04-collision-avoidance-geometric.md` | Geometric collision detection | |
| 05 | `05-environment-obstacles-wind.md` | Wind model, obstacle field generator | |
| 06 | `06-drone-as-rl-environment.md` | 🤖 Gym wrapper, state/action spaces | ✓ |
| 07 | `07-reward-shaping-for-flight.md` | 🤖 Reward design for navigation | ✓ |
| 08 | `08-rl-agents-from-scratch.md` | 🤖 PPO implementation, training loop | ✓ |
| — | `GATE-02.md` | Verification gate | |

## Learning Objectives

By the end of this phase:

1. Model a drone as a 6-DOF rigid body with realistic dynamics
2. Implement PID control for altitude, velocity, and position hold
3. Model sensor noise and implement a simple Kalman filter
4. Detect collisions and navigate around obstacles
5. **Wrap the entire simulation as a Gym environment** 🤖
6. **Design reward functions that produce desired flight behaviors** 🤖
7. **Train a neural network policy from scratch using PPO** 🤖

## Prerequisites

- Phase 01 and Phase 02 completed (gates passed)
- numpy, matplotlib installed
- PyTorch installed (`pip install torch`)

## Embedded Notes

Throughout this phase, each concept includes a note about how it
translates to real embedded drone hardware:
- 6-DOF on an IMU (BNO055, MPU6050)
- Sensor fusion with complementary filter vs Kalman on a microcontroller
- RL inference on edge (TensorFlow Lite, ONNX runtime on Jetson Nano)

**Next:** `01-6dof-kinematics.md`
