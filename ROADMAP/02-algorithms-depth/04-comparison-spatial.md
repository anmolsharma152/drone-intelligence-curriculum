# Spatial Index Comparison: QuadTree vs Grid vs KD-Tree

## Learning Objective

Build a unified benchmark framework that empirically validates the
Big-O claims of all three spatial data structures. Understand the
engineering factors that determine which structure to use in practice.

## The Benchmark Framework

Create `bench_spatial.py` that:

1. **Generates test data** at multiple scales: N = 100, 1K, 10K, 100K
2. **Tests each structure** on the same data
3. **Measures** build/insert time, query time, memory
4. **Verifies correctness** — all structures must return the same results
5. **Plots** the results with log-log axes to confirm scaling behavior

### Test Regimes

```
Regime A: Uniform random (particles spread evenly across 8km)
Regime B: Clustered (all points in a 100m × 100m corner)
Regime C: Sparse clusters (10 clusters of 1000 points each)
Regime D: Linear strip (points along a diagonal line)
```

### Verification

```python
def verify_match(particles, cx, cy, radius):
    """All structures must return the same particles for the same query."""
    # QuadTree
    qt = build_quadtree(particles)
    found_qt = set(qt.query(Rect(cx, cy, radius, radius), []))

    # Grid
    grid = build_grid(particles, cell_size=200)
    found_grid = set(grid.query(cx, cy, radius))

    # KD-Tree
    kd = build_kdtree(particles)
    found_kd = set(kd.range_query(cx, cy, radius))

    assert found_qt == found_grid == found_kd, "Results differ!"
```

## Expected Results

| Structure | Insert (uniform) | Insert (clustered) | Query (uniform) | Query (clustered) |
|---|---|---|---|---|
| QuadTree | O(n log n) | O(n²) worst-case | O(k + log n) | O(k + n) worst |
| Grid | O(n) | O(n) | O(k + cells) | O(k + cells) |
| KD-Tree | O(n log n) | O(n log n)* | O(k + log n) | O(k + log n)* |

\* KD-Tree is bulk-loaded and always balanced.

## Log-Log Plot Analysis

For a structure with complexity O(n^c), a log-log plot of runtime vs n
has slope c. For O(log n), the slope decreases with n.

```python
import matplotlib.pyplot as plt
import numpy as np

# After collecting times[n] for each n:
log_n = np.log(list(times.keys()))
log_t = np.log(list(times.values()))
slope = np.polyfit(log_n, log_t, 1)[0]
print(f"Empirical scaling exponent: {slope:.2f}")
# Should be ~1.0 for O(n), ~0.0 for O(1), decreasing for O(log n)
```

## Results Table Template

Copy this and fill in your actual numbers:

```
                  | Build Time (ms)         | Query Time (ms, 1000 iters)  | Memory (MB)
                  | 100  1K   10K  100K     | 100  1K   10K  100K          | 100  1K  10K  100K
------------------+-------------------------+------------------------------+-------------------
QuadTree (cap=4)  | ...  ...  ...  ...      | ...  ...  ...  ...           | ...  ...  ...  ...
QuadTree (cap=16) | ...  ...  ...  ...      | ...  ...  ...  ...           | ...  ...  ...  ...
Grid (cell=100)   | ...  ...  ...  ...      | ...  ...  ...  ...           | ...  ...  ...  ...
Grid (cell=500)   | ...  ...  ...  ...      | ...  ...  ...  ...           | ...  ...  ...  ...
KD-Tree            | ...  ...  ...  ...      | ...  ...  ...  ...           | ...  ...  ...  ...
Naive (list scan)  | ...  ...  ...  ...      | ...  ...  ...  ...           | ...  ...  ...  ...
```

## Decision Matrix

Based on your benchmarks, fill in which structure wins for each scenario:

| Scenario | Best Structure | Why |
|---|---|---|
| 400 particles, uniform, 60Hz game | _____ | _____ |
| 100K particles, uniform, radar queries | _____ | _____ |
| 10K particles in 10 clusters, NN queries | _____ | _____ |
| Dynamic: particles constantly moving | _____ | _____ |
| Memory-constrained embedded system | _____ | _____ |

## Implementation Task

### Build `bench_spatial.py`

Complete the benchmark framework. It must:

1. Accept command-line arguments for N range, structure selection, regime
2. Print a markdown-formatted results table
3. Save a log-log plot to `plots/spatial_benchmark.png`
4. Verify correctness across structures

### Run and Analyze

Run the benchmark for all four regimes. Write a short analysis answering:

- Which structure has the fastest build/insert?
- Which has the fastest query?
- Which degrades most gracefully under clustering?
- At what N does the naive O(n) scan become unacceptable for 60Hz?
- Which structure would you use for the drone simulation and why?

## Check-In Questions

1. On a log-log plot, what slope indicates O(n²)? O(√n)? O(n log n)?

2. The naive approach (list scan) is O(n) per query. If your game runs
   at 60Hz and you need one query per frame, at what N does the naive
   approach exceed 16.67ms?

3. The Grid has a fixed memory overhead regardless of N. Compute this
   overhead for cell_size = 200m in an 8000m world. How does it compare
   to the QuadTree overhead for 10K points?

4. Why does the KD-Tree not suffer from worst-case clustering like the
   QuadTree does? What is the tradeoff?

## Connection Back

In `game.py`, the mouse radar uses QuadTree query. At 400 particles,
any structure would work. At 10,000 particles, the difference becomes
apparent. At 100,000, only the right choice keeps you at 60fps.

**Next:** `05-a-star-pathfinding.md`
