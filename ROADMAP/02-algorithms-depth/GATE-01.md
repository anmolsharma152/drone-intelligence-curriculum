# Gate 01: Algorithms & Data Structures Verification

You must complete all of the following before proceeding to Phase 03.

## 1. Spatial Index Implementation

- [ ] `spatial_grid.py` — Uniform grid is implemented and verified correct
- [ ] `kdtree.py` — KD-Tree is implemented with nearest-neighbor + range query
- [ ] Both structures produce identical query results to your QuadTree
  (verified on 10,000 random particles with 100 random queries)

## 2. Benchmark Suite

- [ ] `bench_spatial.py` is complete and runnable
- [ ] Tested all four regimes: uniform, clustered, sparse clusters, linear
- [ ] Results table is filled out (at least for uniform + clustered regimes)
- [ ] Log-log plot saved to `plots/spatial_benchmark.png`

Verify with:
```bash
python bench_spatial.py --n 100,1000,10000 --regime uniform --plot plots/bench.png
```

## 3. Pathfinding

- [ ] `astar.py` — A* implemented with at least two heuristic options
- [ ] `rrt_planner.py` — RRT implemented (RRT* is bonus)
- [ ] A* can navigate around 20+ obstacles on a 160×160 grid
- [ ] RRT can navigate around 20+ circular obstacles in continuous space
- [ ] Both produce visualizations

## 4. Formal Analysis Self-Assessment

Can you answer these without looking at the files?

**QuadTree:**
- [ ] Average-case insert complexity with proof
- [ ] Worst-case insert complexity and what input causes it
- [ ] Space complexity as function of n and capacity

**Grid:**
- [ ] Query complexity as function of cell size, query radius, and density
- [ ] Memory overhead formula
- [ ] When Grid outperforms QuadTree and why

**KD-Tree:**
- [ ] Build complexity with proof
- [ ] Nearest-neighbor search complexity and pruning condition
- [ ] Why KD-Tree suffers from "curse of dimensionality"

**A*:**
- [ ] Conditions for admissibility and consistency
- [ ] How heuristic quality affects search performance
- [ ] Space complexity on a grid

**RRT:**
- [ ] Per-iteration complexity with and without KD-Tree acceleration
- [ ] Probabilistic completeness (define it)
- [ ] Why RRT* converges to optimal path while RRT does not

## 5. Benchmark Data

Your final `bench_spatial.py` should produce approximately:

```
                     | Build (ms)           | Query (ms, 1000x)
                     | 1K    10K    100K    | 1K    10K    100K
---------------------+----------------------+---------------------
QuadTree (cap=4)     | 1.2   15.2   210.4   | 0.8   1.1    2.3
QuadTree (cap=16)    | 0.9   11.8   180.2   | 1.2   1.9    3.4
Grid (cell=200)      | 0.3   3.1    35.2    | 0.6   0.7    0.9
KD-Tree (bulk)       | 2.1   28.4   410.6   | 0.5   0.6    0.8
Naive (list scan)    | 0.1   1.2    12.4    | 18.2  185.3  2100.0
```

*(Your actual numbers will vary by hardware. The key is the scaling trend.)*

## 6. Understanding Check

Write a paragraph answering each:

1. "For the drone simulation's mouse radar, I would choose ______ because
   ______. At N = ______, I would switch to ______ because ______."

2. "For path planning between two waypoints 5km apart with 30+ obstacles,
   I would use ______ because ______."

3. "The worst-case QuadTree input is ______. It causes O(n) inserts because
   ______. The fix is ______."

---

**Gate status:** `[PASSED]` / `[NOT YET]`

Once all boxes are checked, proceed to `03-physics-simulation/README.md`.
