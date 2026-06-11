# Formal Complexity Analysis: Current Code

## `SpatialMap` (in `main.py` and `main2.py`)

### `update_object(obj_id, x, y, z)`

```
dict.__setitem__  →  O(1) average, O(n) worst-case on hash collision
boundary check    →  O(1) (two comparisons)
np.array creation →  O(3) (trivially O(1))
```
**Overall: O(1) amortized.**  
**Space:** O(n) where n = number of tracked objects.

### `get_object_pos(obj_id)`

```
dict.get  →  O(1) average, O(n) worst-case
```
**Overall: O(1) amortized.**

### Complexity Note

The `dict` in `SpatialMap` uses Python's built-in hash table with open
addressing. Average case assumes a good hash function (Python uses SipHash
for strings) and load factor < 2/3. Worst case occurs when all keys collide
— possible with adversarial input, but not realistic here.

**Auxiliary space:** Both methods are O(1) extra beyond storage.

---

## `QuadTree` (in `game.py`)

### `insert(point)`

Let n = number of points in the tree, c = capacity (constant = 4).

**Average case (uniform distribution):**
The tree is roughly balanced. Each insert traverses from root to leaf:
O(log₄ n) levels. At each level, `boundary.contains()` is O(1).

**Worst case (pathological clustering):**
All points cluster in one quadrant. The tree degenerates into a deep spine:
O(n) levels, O(n) per insert.

**Proof sketch for average case:**
For uniformly distributed points in a square of side L, each subdivision
creates 4 equal-area quadrants. The expected number of points in any
quadrant after k subdivisions is n / 4ᵏ. The leaf is reached when
n / 4ᵏ ≤ c, so k ≈ log₄(n/c) = O(log n).

**Amortized with subdivision:**
Subdivision (creating 4 child QuadTrees) costs O(1). Each point causes at
most O(log n) subdivisions along its path. Amortized cost per insertion
remains O(log n).

### `query(range_rect, found)`

**Best case (empty range):** O(1) — root boundary test fails, return immediately.

**Average case:** O(k + log n) where k = number of reported points.  
Query visits all nodes whose boundary intersects the range. For uniform
distributions and a compact query range, this is proportional to the
number of boundary-crossing nodes (~O(log n)) plus the cost to collect
matching points (O(k)).

**Worst case (range covers entire space):** O(n). The query visits every
node and collects every point because the range intersects all boundaries.

### `subdivide()`

**O(1).** Creates 4 new QuadTree nodes with halved boundaries. No iteration.

### Draw

**O(m)** where m = number of nodes in the tree. Each node draws one
rectangle. In a balanced tree with n points, m ≈ 4n/c = O(n).

---

## `DataLogger` (in `main2.py`)

### `add_entry(obj_id, pos, dist, timestamp)`

```
queue.put  →  O(1) amortized (lock + list append under the hood)
```
**Overall: O(1).** The whole point: add_entry is non-blocking and fast.

### `run()` (the background thread loop)

Per entry:
```
queue.get     →  O(1) amortized
csv.writerow  →  O(len(row)) = O(1) since row has fixed width (6 fields)
file.write    →  O(1) amortized (buffered I/O)
```
**Overall per entry: O(1) amortized.**  
The thread blocks on `queue.get(timeout=1.0)` when the queue is empty,
consuming effectively zero CPU.

**Total space:** O(q) where q = maximum queue length. In steady state at
20Hz with a fast disk, q should stay small (< 10 entries). Under sustained
disk lag, q grows up to available memory.

---

## Main Loop (both files)

Each frame:

| Operation | Cost |
|---|---|
| `update_object` | O(1) |
| `get_object_pos` | O(1) |
| `np.linalg.norm` | O(3) = O(1) |
| `print` | O(len(string)) = depends on terminal |
| `sleep` | O(1) (blocks until timer expires) |

**Per-frame: O(1).**  
The loop is bounded by `sleep` — it runs at exactly 20Hz regardless of
computation time, until `elapsed > FRAME_TIME` triggers the lag warning.

---

## Empirical Verification (TODO)

After fixing the bugs, add timing instrumentation:

```python
import time
t0 = time.perf_counter()
for _ in range(10000):
    radar.update_object("test", 0.0, 0.0, 0.0)
t1 = time.perf_counter()
print(f"10000 updates: {(t1-t0)/10000*1e9:.1f} ns per call")
```

Expected: `update_object` should take < 500ns per call on modern hardware.
If it takes > 5µs, something is wrong (e.g., numpy overhead dominating).

## Self-Check Questions

1. Distinguish best-case, average-case, and worst-case for QuadTree `insert`.
   Provide a concrete input that triggers each case.

2. The `SpatialMap` uses a Python dict. Could a list of tuples be faster for
   < 10 objects? At what n does the dict's O(1) lookup beat the list's O(n)?

3. `np.linalg.norm` computes sqrt(x² + y² + z²). How many floating-point
   operations does this cost? Write the exact count.

4. The QuadTree `draw` method visits every node. Can you draw the tree
   without visiting leaf nodes that are off-screen? What's the complexity
   improvement?

5. The main loop uses `time.sleep()` to maintain 20Hz. What happens to the
   timing if `np.linalg.norm` suddenly takes 100ms due to a large array?
   (Trick question — it only computes norm of 3 elements.)

**Next:** `03-fix-bugs-first.md`
