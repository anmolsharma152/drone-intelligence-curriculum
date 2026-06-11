# Gate 00: Foundations Verification

You must complete all of the following before proceeding to Phase 02.

## 1. Code Fixes

- [ ] Bug 1 fixed: out-of-bounds drone does not crash or silently freeze
- [ ] Bug 2 fixed: speed calculation matches stated speed
- [ ] `main.py` runs without errors for 120+ simulated seconds
- [ ] `main2.py` runs without errors for 120+ simulated seconds
- [ ] `flight_log.csv` shows correct, continuous telemetry

To verify:
```bash
# Run for 120 seconds (simulated) — use timeout or let it run
timeout 130 python main.py
timeout 130 python main2.py
```

During the run, let it pass the boundary (~80 seconds). Confirm:
- `main.py` prints "WARNING" messages instead of crashing
- `main2.py` produces a complete CSV

## 2. Complexity Self-Assessment

Can you answer these without looking at the files?

- [ ] What is the Big-O of `SpatialMap.update_object`? Worst and average case.
- [ ] What is the Big-O of `QuadTree.insert`? Distinguish best/average/worst.
- [ ] What is the Big-O of `QuadTree.query`? Under what conditions does it
      degenerate to O(n)?
- [ ] What is the space complexity of `DataLogger` as a function of queue depth?
- [ ] The main loop runs at 20Hz. What is the complexity of one iteration?

Write your answers in `notes/complexity-answers.txt` (create the directory
if needed).

## 3. Performance Baseline

Record baseline benchmarks for your current code:

```python
# bench_baseline.py
import time
import numpy as np

# SpatialMap benchmark
from main import SpatialMap, BOUNDS_METERS
sm = SpatialMap()
t0 = time.perf_counter()
for i in range(100000):
    sm.update_object(f"obj_{i}", np.random.uniform(-4000, 4000),
                     np.random.uniform(-4000, 4000), 150.0)
t1 = time.perf_counter()
print(f"SpatialMap 100k inserts: {(t1-t0)*1000:.2f}ms")

# QuadTree benchmark
import sys
sys.path.insert(0, ".")
from game import QuadTree, Rect, Particle, WORLD_SIZE
import random
boundary = Rect(0, 0, 4000, 4000)
particles = [Particle(random.uniform(-4000, 4000),
                      random.uniform(-4000, 4000))
             for _ in range(10000)]

t0 = time.perf_counter()
qt = QuadTree(boundary, 4)
for p in particles:
    qt.insert(p)
t1 = time.perf_counter()
print(f"QuadTree 10k inserts: {(t1-t0)*1000:.2f}ms")

found = []
t0 = time.perf_counter()
for _ in range(1000):
    found.clear()
    qt.query(Rect(0, 0, 500, 500), found)
t1 = time.perf_counter()
print(f"QuadTree 1000 queries: {(t1-t0)*1000:.2f}ms, avg found: {len(found)}")
```

Record the output:

```
SpatialMap 100k inserts: ______ ms
QuadTree 10k inserts: ______ ms
QuadTree 1000 queries: ______ ms, avg found: ______
```

Keep these numbers. You will compare against optimized versions in Phase 06.

## 4. Understanding Check

Write 2-3 sentences explaining:

- Why does the main loop use `time.sleep(FRAME_TIME - elapsed)` instead of
  a fixed `time.sleep(FRAME_TIME)`?
- Under what condition does `game.py`'s QuadTree degrade to a linked list?
- Why is `daemon = True` both useful and dangerous in `DataLogger`?

---

**Gate status:** `[PASSED]` / `[NOT YET]`

Once all boxes are checked, proceed to `02-algorithms-depth/README.md`.
