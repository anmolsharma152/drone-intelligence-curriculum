# Phase 05: Visualization, UI & Policy Insight

## Overview

You've built the simulation and the intelligence. Now make it visible.

This phase has three goals:
1. **3D rendering** of the drone world (replace pygame 2D circles)
2. **Live dashboard** with real-time telemetry plots
3. **Policy visualization** — see what the neural network is "thinking"

## Files

| # | File | Description | AI? |
|---|---|---|---|
| 01 | `01-3d-rendering-opengl.md` | 3D rendering with ModernGL/Pyglet | |
| 02 | `02-live-dashboard-telemetry.md` | Real-time telemetry dashboard | |
| 03 | `03-policy-visualization.md` | 🤖 Attention maps, value heatmaps, policy rollout viz | ✓ |
| — | `GATE-04.md` | Verification gate | |

## Learning Objectives

1. Render the drone simulation in 3D with basic lighting and camera
2. Build a real-time dashboard with altitude, speed, attitude plots
3. **Visualize policy internals: value function, action distributions,
   attention, failure modes** 🤖

## Prerequisites

- Phase 04 completed (gate passed)
- ModernGL (`pip install moderngl`) or pyglet
- matplotlib

**Next:** `01-3d-rendering-opengl.md`
