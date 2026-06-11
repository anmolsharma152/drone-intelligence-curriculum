# Code Audit: What You Already Built

## File-by-File Breakdown

### `main.py` — Basic Tracking Loop

**What it does:**
- Defines an 8km x 8km world with center at (0,0,0)
- `SpatialMap` stores object positions as `np.float64` arrays keyed by ID
- A single simulated drone flies diagonally from (-4000, -4000, 150) at 100 m/s
- Main loop runs at 20Hz with a fixed-timestep + compensation sleep
- Each frame: update position → compute distance-from-center via `np.linalg.norm` → print → advance → sleep

**What's wrong:**
- **Bug 1 (crash):** When the drone exits ±4000m bounds (after ~80s), `update_object` silently returns without storing. Next frame `get_object_pos` returns `None`, and `np.linalg.norm(None)` raises `TypeError`.
- **Bug 2 (speed):** Comment says 100 m/s "diagonally", but both x and y are incremented by 100 m/s each frame. True diagonal speed is `sqrt(100² + 100²) ≈ 141.4 m/s`.
- No `if __name__ == "__main__"` guard on `SpatialMap` — harmless for single-file usage, but poor practice for reuse.
- Uses `time.sleep()` with no guard against negative sleep (has lag detection but no mechanism to recover).

### `main2.py` — Concurrent Logging Version

**What it does:**
- Same `SpatialMap` and movement logic as `main.py`
- Adds `DataLogger`: a background thread that writes CSV telemetry
- Producer-consumer via `queue.Queue`: main loop enqueues data, logger drains it
- Daemon thread auto-dies if main crashes; graceful `stop()` + `join()` on Ctrl+C
- Prints every frame to terminal (unlike the comment that claims it prints every 20th)

**What's wrong:**
- Same two bugs as `main.py` (None crash, speed calc)
- Comment on line 94 says "Print less frequently (every 20th frame)" but the code prints every frame unconditionally
- `DataLogger` uses `daemon = True` — if main crashes mid-write, CSV may truncate
- Timestamp formatting `[:-3]` on microseconds is fragile (works for 6-digit micro but won't adapt to platform differences)

### `game.py` — Pygame QuadTree Visualization

**What it does:**
- 400 particles with random velocities in ±4000m world
- `QuadTree` spatial index with capacity=4, subdivides on overflow
- Mouse acts as radar: 500m box query highlights particles
- Renders grid lines, radar box, particles at 60 fps
- Bounce-off-walls boundary handling

**What's good:**
- `Rect.contains()` and `Rect.intersects()` are efficient and correct
- QuadTree insert/query/draw recursion is clean
- Screen-to-sim coordinate conversion is correct
- No obvious bugs

**What could be better (not bugs):**
- No `__slots__` on `Particle` — each particle pays dict overhead (~200 bytes vs ~56 bytes)
- No depth limit on QuadTree — pathological clustering could over-subdivide
- `query` returns a list built by appending — repeated allocation pressure at 60Hz

### `flight_log.csv` — Output from `main2.py`

388 rows of telemetry. One entry per frame at 20Hz = ~19.4 seconds of simulated
flight. Shows drone moving from (-4000, -4000) toward the center, with distance
decreasing as expected.

## Self-Check Questions

1. Why does the drone eventually crash in `main.py`? Trace the exact call sequence.
2. What is the actual diagonal speed of the drone? Prove it with vector math.
3. How does `DataLogger` ensure no data is lost on shutdown? (Hint: trace `stop()` + `join()`.)
4. In `game.py`, why might the QuadTree subdivide indefinitely? How would you fix it?
5. The `flight_log.csv` shows distance decreasing from 5658m toward 0m. At what point does it stop? Why?

**Next:** `02-big-o-of-current-code.md`
