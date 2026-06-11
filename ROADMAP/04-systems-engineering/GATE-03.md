# Gate 03: Systems Engineering & ML Pipeline

Complete all of the following before proceeding to Phase 05.

## 1. Multi-Drone Architecture

- [ ] `run_swarm.py` spawns N independent drone simulator processes
- [ ] Each drone runs independently and sends telemetry
- [ ] Ground station receives and displays telemetry from all drones
- [ ] Keyboard command interface works (pause, reset, waypoint)

Verify with:
```bash
python run_swarm.py --n 4  # Should show 4 drones flying simultaneously
```

## 2. ECS Architecture

- [ ] `ecs.py` — Entity-Component-System with entity creation, component
      add/remove, and query system
- [ ] At least 4 component types (Position, Velocity, Renderable, AIState)
- [ ] At least 3 systems (Physics, Render, Collision)
- [ ] Game runs identically to monolithic version after refactor
- [ ] Performance benchmark recorded:

```
N     | Monolithic (ms) | ECS (ms) | Speedup
100   |    ___          |   ___    |   ___
1000  |    ___          |   ___    |   ___
```

## 3. UDP Protocol

- [ ] `protocol.py` — Binary protocol with header, payload, checksum
- [ ] At least 3 message types implemented (waypoint, telemetry, status)
- [ ] Pack/unpack round-trip verified (pack then unpack gives original)
- [ ] `drone_server.py` + `ground_control.py` communicate correctly
- [ ] Latency benchmark:

```
Send 1000 telemetry packets: ___ ms round-trip
Packet loss at 100Hz: ___%
```

## 4. Data Logging

- [ ] `telemetry_logger.py` — Binary log format with header and typed entries
- [ ] Logs include state, actions, events
- [ ] `replay.py` — Can replay a log file visually
- [ ] `analyze_logs.py` — Computes flight time, speed, energy, efficiency

Benchmark:
```
1 hour log size: ___ MB (binary) vs ___ MB (CSV)
```

## 5. 🤖 Feature Pipeline

- [ ] `feature_pipeline.py` — FeatureEngine with 15+ derived features
- [ ] RunningNormalizer for online mean/std
- [ ] Feature selection (correlation analysis)
- [ ] Feature extraction benchmark:

```
100,000 observations processed in ___ ms
Feature vector dimension: ___
Features dropped by correlation: ___
```

## 6. 🤖 Inference Server

- [ ] `inference_server.py` — Loads trained policy, serves single + batch
- [ ] Batching worker with configurable timeout and batch size
- [ ] Model versioning and A/B testing (ModelRouter)
- [ ] Benchmark:

```
Single inference: ___ μs (CPU)
Batch 64: ___ μs / inference
Throughput: ___ inferences/sec
N drones served at 60Hz: ___
```

## 7. 🤖 Online Learning

- [ ] `online_learning.py` — ExperienceBuffer + OnlinePPO
- [ ] Drone adapts mid-flight with adjustable update interval
- [ ] PolicyMonitor detects improvement vs divergence
- [ ] Benchmark:

```
Static policy:             ___ avg reward
Online (100 updates):      ___ avg reward
Online (1000 updates):     ___ avg reward
Improvement over static:   ___%
```

## 8. Understanding Check

1. Draw the full system architecture: N drones, ground station,
   inference server, feature pipeline. Show the data flow.

2. ECS separates data from logic. How does this help the ML inference
   pipeline specifically? (Hint: batch processing)

3. Trace the path of a single observation from drone sensor through
   feature pipeline through inference to action. What are the latency
   bottlenecks?

4. Online learning must not block the main control loop. Describe two
   architectural patterns to achieve non-blocking online updates.

---

**Gate status:** `[PASSED]` / `[NOT YET]`

Once all boxes are checked, proceed to `05-visualization-ui/README.md`.
