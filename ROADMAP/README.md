# DroneSim Roadmap: AI-First Flight Intelligence

A self-guided journey through algorithms, physics simulation, systems
engineering, and **reinforcement learning for drone intelligence** —
all built around a drone simulation you write from scratch.

## The Vision

Most drone tutorials teach you to fly a drone. This roadmap teaches you
to build a drone that learns to fly itself.

```
Classical stack (Phases 1-3):
  algorithms → physics → control → sensors

AI stack (Phases 3-7):
  simulation env → state/action spaces → reward shaping → RL → policy

End-to-end:
  You build the world, the drone, and its brain.
```

## Core Philosophy: RL Throughout

AI/ML/RL is not a separate "capstone" tacked on at the end. It is woven
into every phase from the moment you have a working physics sim:

- **Phase 02:** Spatial structures enable fast RL environment stepping
- **Phase 03:** You write a Gym wrapper before the physics engine is
  even finished — every feature you add (wind, sensors, obstacles)
  immediately becomes part of the RL observation space
- **Phase 04:** Systems engineering includes model serving, feature
  pipelines, and online learning loops
- **Phase 05:** Visualization includes policy heatmaps, attention maps,
  and training reward curves
- **Phase 06:** Performance work targets distributed RL training and
  sim-to-real considerations

## How This Roadmap Works

This is a **progressive, gated curriculum**. Each phase builds on the
previous one. You cannot skip ahead. Each phase ends with a **hard gate**
— a concrete milestone you must complete and verify.

```
01. Foundations Review (fix, audit, understand existing code)
         │
         ▼
02. Algorithms & Data Structures (spatial indexing, pathfinding)
         │
         ▼
03. Physics Simulation & RL Environment (6-DOF, PID, Gym wrapper,
    state/action spaces, reward shaping, PPO from scratch)
         │
         ▼
04. Systems Engineering & ML Pipeline (multi-drone, UDP, ECS,
    feature engineering, model serving, online learning)
         │
         ▼
05. Visualization, UI & Policy Insight (3D rendering, live dashboards,
    policy heatmaps, training visualization)
         │
         ▼
06. Performance, Scale & Sim-to-Real (profiling, vectorization,
    distributed RL, 10k entities, real-world deployment path)
```

## Prerequisites

- Python 3.10+
- numpy, pygame, matplotlib (install as needed)
- PyTorch (for RL agents, added in Phase 03)
- A terminal and a text editor
- Curiosity and patience

## How to Use This Roadmap

1. **Start at Phase 01.** Read the code-audit, understand the complexity
   of what you already wrote, and fix the bugs.
2. **Complete the Gate.** Each phase has a `GATE-NN.md` with verification
   steps. Do not proceed until you pass it.
3. **Log your results.** Each gate asks you to record benchmark numbers.
   Keep them — you'll compare against later optimizations.
4. **AI/ML tasks are not optional.** They are marked with an 🤖 icon
   in each phase. If you skip them, the later RL phases won't make sense.
5. **Skip nothing.** The phases are deliberately ordered. Pathfinding
   needs spatial indexing. Physics needs timestep knowledge from Phase 01.
   RL needs a working physics environment from Phase 03.

## File Naming Convention

```
NN-concept-name.md       — A single concept file
GATE-NN.md               — The gate at the end of a phase
```

Each concept file has the same structure:
- Learning Objective
- Formal Analysis (Big-O proofs, space complexity)
- Implementation Task
- Check-In Questions
- Complexity Benchmark
- Connection Back to prior work
- Next pointer

Files marked with 🤖 contain AI/ML/RL content.

## The End Goal

By the end of this roadmap you will have built:

### Simulation
- A 10,000-entity drone simulation running at 60Hz
- 3 spatial indexing structures with empirical complexity verification
- A 6-DOF physics engine with PID control and realistic sensor noise
- A multi-drone UDP command protocol
- A 3D visualizer with live telemetry
- A benchmark suite that automatically validates Big-O claims

### AI / Drone Intelligence
- A Gym-compatible RL environment built on the physics sim
- PPO and policy gradient agents trained from scratch
- A feature engineering pipeline feeding live agent observations
- Policy visualization (attention maps, value heatmaps)
- An online learning loop that adapts mid-flight
- A sim-to-real gap analysis for real drone deployment

### Embedded Systems Path
Each phase includes optional "embedded notes" that reframe the concept
for resource-constrained deployment (Pi Pico, ESP32, Jetson Nano).

---

**Start here:** `01-foundations-review/README.md`
