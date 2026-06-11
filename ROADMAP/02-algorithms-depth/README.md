# Phase 02: Algorithms & Data Structures

## Goal

Deepen your understanding of spatial data structures and pathfinding
algorithms. You already implemented a QuadTree in `game.py`. Now we
analyze it formally, build alternatives, compare them, and then use
them for drone path planning.

## Why Spatial Data Structures?

A naive approach to "find all objects within 500m of the mouse" is to
loop over all N particles — O(N). With 400 particles, that's fast.
With 10,000, it starts to hurt. With 1,000,000, it's impossible at
60Hz.

Spatial data structures reduce this to O(log N) or O(1).

## Files in This Phase

| File | What You'll Build / Learn |
|---|---|
| `01-quadtree-deep-dive.md` | Formal analysis of your existing QuadTree |
| `02-uniform-grid.md` | Build a spatial hash grid; O(1) queries |
| `03-kd-tree.md` | Build a KD-Tree; nearest-neighbor search |
| `04-comparison-spatial.md` | Benchmark framework comparing all three |
| `05-a-star-pathfinding.md` | A* on a grid; optimality proof |
| `06-rrt-path-planning.md` | RRT/RRT* for continuous path planning |
| `GATE-01.md` | **Gate:** Working benchmark + planner |

## Learning Objectives

By the end of this phase, you will:

- Prove the average-case O(log n) of QuadTree operations
- Build a uniform grid spatial hash and explain when it beats QuadTree
- Build a KD-Tree with k-nearest-neighbor search
- Empirically verify Big-O claims with a benchmark framework
- Implement A* on a 2D grid with an admissible heuristic
- Implement RRT for continuous-space path planning
- Compare RRT vs A*: when to use each

## Prerequisites

- Phase 01 complete (all gate items checked)
- numpy, matplotlib installed
- `game.py` working

## Progression

Read the files in order. Each builds on the previous.

**Start:** `01-quadtree-deep-dive.md`
