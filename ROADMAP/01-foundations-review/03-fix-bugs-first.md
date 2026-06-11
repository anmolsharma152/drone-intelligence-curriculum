# Fix Bugs First

Before we build anything new, fix what's broken. Two bugs exist across
`main.py` and `main2.py`.

## Bug 1: None Crash (main.py:57, main2.py:86)

**Root cause:**  
The drone starts at (-4000, -4000). At 100 m/s diagonally with 20Hz timing,
it moves 5 meters per frame along each axis. After 800 frames (~40 seconds),
it reaches (0, 0) — the center. After 1600 frames (~80 seconds), it reaches
(+4000, +4000) — the boundary.

On frame 1601, the drone would be at (+4005, +4005), which exceeds +4000.
`update_object` executes the `return` on line 25/61 without storing the
position. The dict still has the previous frame's position.

On frame 1602, the drone would be at (+4010, +4010) — still out of bounds.
`update_object` returns without storing. But the position was never stored
from frame 1601 either, so... wait, let me re-read.

Actually, let me re-trace: `update_object` prints a warning but does NOT
store the position when out of bounds. The next frame calls `get_object_pos`,
which does `self.objects.get(obj_id, None)` — BUT the dict still has the
last *in-bounds* position from frame 1600. So `get_object_pos` returns
that last position, not None. The crash doesn't happen immediately.

It will only crash after the drone leaves bounds AND the dict key is
explicitly deleted, or if `update_object` is never called (so the key
was never inserted). Wait — the drone *starts* at (-4000, -4000) which
is within bounds (inclusive). The first frame inserts it. Each subsequent
frame re-inserts. When the drone goes out of bounds, `update_object`
returns early without updating the dict, so the *last in-bounds* position
persists. `get_object_pos` returns that old position. `np.linalg.norm`
works fine on that old position.

**So the None crash only happens if the dict key doesn't exist at all,**
which doesn't happen here because the drone starts in-bounds. The actual
problem is subtler: once out of bounds, the position visibly freezes at
(+4000, +4000) while the drone conceptually keeps moving. The position
reported is stale. The code continues running but is now lying to you.

To actually trigger a None crash, you'd need to query an ID that was
never inserted, or delete the key after the last in-bounds frame. As
written, it's a **silent data corruption** bug, not a crash.

Regardless, the fix is to handle the None case properly AND to clamp the
drone or wrap it around, depending on your intent.

### Fix (both files)

```python
def update_object(self, obj_id, x, y, z):
    if not (-BOUNDS_METERS <= x <= BOUNDS_METERS) or \
       not (-BOUNDS_METERS <= y <= BOUNDS_METERS):
        print(f"WARNING: Object {obj_id} out of bounds at ({x:.2f}, {y:.2f})!")
        # Option A: Don't update (position freezes)
        # Option B: Remove from tracking
        # self.objects.pop(obj_id, None)
        return
    self.objects[obj_id] = np.array([x, y, z], dtype=np.float64)

def get_object_pos(self, obj_id):
    pos = self.objects.get(obj_id, None)
    if pos is None:
        print(f"WARNING: Object {obj_id} not found in spatial map!")
    return pos
```

And in the main loop:

```python
pos = radar.get_object_pos("drone_1")
if pos is None:
    print("Drone lost! Resetting...")
    drone_x, drone_y = -4000.0, -4000.0
    continue
```

## Bug 2: Speed Calculation

**Root cause:**  
```
speed_mps = 100.0   # comment says 100 m/s diagonally
move_step = speed_mps * FRAME_TIME   # 100 * 0.05 = 5.0
drone_x += move_step
drone_y += move_step
```

The drone moves 5m in x AND 5m in y per frame. The resulting velocity
vector is (5, 5, 0) per frame, which is 100 m/s in each axis. The true
speed is `sqrt(100² + 100²) ≈ 141.42 m/s`.

**Fix:**
If you want 100 m/s at a 45-degree diagonal:
```python
speed_mps = 100.0
# Diagonal movement: each axis gets speed / sqrt(2)
diag_step = (speed_mps / np.sqrt(2)) * FRAME_TIME
drone_x += diag_step
drone_y += diag_step
```

Or, if you want 100 m/s in each axis:
```python
speed_mps = 100.0   # but rename to clarify: speed_xy_mps
move_step = speed_mps * FRAME_TIME   # 5.0 m/frame in each axis
```

## Verification

After fixing both bugs:

1. Run `main.py` for 90 simulated seconds (let it fly past the boundary).
   Confirm it prints "WARNING" but does not crash.
2. Measure distance traveled per second: should be exactly 100 m/s (or
   whatever you chose).
3. Confirm `flight_log.csv` from `main2.py` shows continuous tracking
   without gaps or freezes.

## Self-Check Questions

1. In `main.py`, the drone flies diagonally at 100 m/s. How far from center
   does it get after 60 seconds, assuming it starts at (-4000, -4000)?
   Show the math.

2. If you chose Option B (`pop` from dict when out of bounds), explain what
   happens on the next frame and why the None-guard in the loop is necessary.

3. The main loop uses `time.perf_counter()` for timing. Why is `perf_counter`
   preferred over `time.time()` for this use case?

**Next:** `GATE-00.md`
