# Profiling & Bottleneck Analysis

## Learning Objective

Systematically profile the drone simulation to identify performance
bottlenecks. Apply the 80/20 rule: 80% of time is spent in 20% of
the code.

## Profiling Tools

### 1. cProfile (Built-in)

```bash
python -m cProfile -o profile.stats game.py --n 1000
python -m pstats profile.stats
```

```python
# In pstats interactive mode:
sort cumtime
stats 20  # Show top 20 functions
```

### 2. py-spy (Sampling Profiler)

```bash
pip install py-spy
py-spy record -o profile.svg -- python game.py --n 1000
# Opens as SVG flamegraph in browser
```

### 3. timeit (Microbenchmarks)

```python
import timeit

# How fast is QuadTree insert?
qt_time = timeit.timeit(
    'qt.insert(p)',
    setup='''
from game import QuadTree, Rect, Particle
qt = QuadTree(Rect(0,0,4000,4000), 4)
p = Particle(100, 200)
''',
    number=100000)

print(f"QuadTree insert: {qt_time/100000*1e6:.1f}μs")
```

## Bottleneck Candidates

Based on profiling, the likely bottlenecks are:

| # | Component | Expected % of frame time | Why |
|---|---|---|---|
| 1 | QuadTree insert (spatial build) | 30-40% | Python function calls per insert |
| 2 | Physics step | 20-30% | Per-entity vector math |
| 3 | Collision detection | 15-25% | Distance checks |
| 4 | Render | 10-15% | Pygame draw calls |
| 5 | AI/Policy inference | 5-10% | Neural net forward pass |

## Profiling Script

```python
# profile_sim.py
import cProfile
import pstats
import io

def run_profile(n_entities=1000, n_frames=100):
    from game import Simulation
    sim = Simulation(n_entities)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(n_frames):
        sim.step()

    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats(20)
    return s.getvalue()

print(run_profile(n_entities=1000, n_frames=100))
```

## Expected Profile Output

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
  1000   3.204    0.003    4.891    0.005   game.py:240(step)
 100000  1.892    0.000    2.345    0.000   game.py:120(insert)  ← QuadTree
  98000  0.891    0.000    0.891    0.000   game.py:80(contains) ← Boundary checks
  50000  0.445    0.000    0.445    0.000   game.py:200(update)  ← Physics
```

## Implementation Task

### 1. Profile Your Simulation

Run the profiler at N = 100, 1000, 5000 entities:

```python
for n in [100, 1000, 5000]:
    print(f"\n=== N={n} ===")
    print(run_profile(n, 100))
    # Measure total frame time
    # Measure per-component time
```

### 2. Create a Flamegraph

```bash
py-spy record -o flamegraph.svg -- python game.py --n 1000
# Open flamegraph.svg in browser
```

### 3. Measure Scaling

```
N      | Total (ms) | Physics | Spatial | Render | AI
-------+------------+---------+---------+--------+-----
100    |    ___     |   ___   |   ___   |  ___   | ___
500    |    ___     |   ___   |   ___   |  ___   | ___
1000   |    ___     |   ___   |   ___   |  ___   | ___
5000   |    ___     |   ___   |   ___   |  ___   | ___
10000  |    ___     |   ___   |   ___   |  ___   | ___
```

## Check-In Questions

1. cProfile adds overhead to function calls (~10-50% slowdown).
   How does py-spy avoid this? What's the tradeoff?

2. The QuadTree insert appears to take 40% of frame time at N=1000.
   Is this the insert cost, the query cost, or both? How would you
   separate them?

3. The physics update is O(N) per frame. The QuadTree insert is
   O(N log N) for uniform data. At what N does QuadTree insert
   exceed physics?

4. A bottleneck that takes 5% of frame time can be optimized to 0%
   and you gain only 5% total speedup. What's this principle called
   and why does it matter?

## 🤖 Connection

Profiling guides ML optimization:

- If inference is the bottleneck: batch more, use smaller network
- If physics is the bottleneck: vectorize, use simpler dynamics
- If logging is the bottleneck: reduce log frequency, use binary
- Each optimization directly increases RL training throughput

**Next:** `02-vectorization-numba.md`
