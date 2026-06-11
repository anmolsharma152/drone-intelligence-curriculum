# Drone Intelligence Curriculum

A self-guided, project-based journey from 2D physics simulation to
reinforcement learning for autonomous drone flight.

**This is not a tutorial.** There are no step-by-step instructions,
no copy-paste solutions, no "type this and run it." Each concept is a
task you implement from scratch, prove to yourself, and verify before
advancing.

## Why This Exists

Most drone resources teach you to *fly* a drone (PID tuning, waypoint
navigation). This curriculum teaches you to build a drone that *learns
to fly itself* — combining classical robotics (6-DOF physics, sensor
fusion, control theory) with modern reinforcement learning (PPO from
scratch, distributed training, sim-to-real transfer).

The arc:

```
Classical stack:    algorithms → physics → control → sensors
AI stack:           simulation env → state/action spaces →
                    reward shaping → PPO → trained policy
End result:         A drone that learns to navigate obstacle fields
                    through trial and error
```

## Structure

The curriculum is organized into **6 gated phases, 46 files**:

| Phase | Topic | Files | AI/ML Content |
|---|---|---|---|
| 01 | Foundations Review | 6 | — |
| 02 | Algorithms & Data Structures | 8 | — |
| 03 | Physics Simulation & RL Environment | 10 | Gym wrapper, PPO from scratch |
| 04 | Systems Engineering & ML Pipeline | 8 | Feature engineering, inference server, online learning |
| 05 | Visualization & Policy Insight | 5 | Policy visualization, saliency, value heatmaps |
| 06 | Performance, Scale & Sim-to-Real | 7 | Distributed PPO, domain randomization, edge deployment |

Each file follows the same structure:
- **Learning Objective** — what you will understand
- **Formal Analysis** — Big-O proofs, space complexity, tradeoffs
- **Implementation Task** — what to build, with code templates
- **Check-In Questions** — verify understanding without peeking at answers
- **Complexity Benchmark** — empirically confirm the theory
- **Connection Back** — how this ties to the drone sim

Every phase ends with a **hard gate** — concrete milestones you must
pass before proceeding.

## What Makes This Unique

- **AI is not a capstone.** RL is woven in from Phase 03, not tacked
  on at the end. Every physics feature you build immediately expands
  the RL observation space.

- **Gated progression.** You cannot skip. Each gate requires verified
  benchmarks, filled tables, and self-assessed understanding.

- **Everything from scratch.** No Gym environment libraries, no RL
  framework abstractions. You write the QuadTree, the A*, the 6-DOF
  physics, the Kalman filter, the PPO — all from first principles.

- **Empirical complexity.** Every Big-O claim is validated with a
  benchmark that produces log-log plots, not hand-waving.

- **Sim-to-real path.** Domain randomization, policy distillation,
  ONNX export, and hardware considerations for real drone deployment.

## Prerequisites

- Python 3.10+, numpy, PyTorch, pygame
- A terminal and a text editor
- Curiosity and patience

## Quick Start

```
git clone https://github.com/anmolsharma152/drone-intelligence-curriculum
cd drone-intelligence-curriculum
# Start at ROADMAP/01-foundations-review/01-code-audit.md
# Complete each phase in order. Skip nothing.
```

## The Starting Point

The existing code (`main.py`, `main2.py`, `game.py`) is a minimal 2D
drone simulation with two known bugs:
- Out-of-bounds drones freeze silently (no crash)
- Diagonal speed is 141 m/s, not 100 m/s

Phase 01 asks you to find and fix these before moving on.

## License

MIT
