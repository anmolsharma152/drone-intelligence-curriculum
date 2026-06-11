# Data Logging & Telemetry

## Learning Objective

Build a structured telemetry logging system that records flight data
for analysis, replay, and ML training.

## Why Structured Logging?

Your current `flight_log.csv` is minimal. A real system needs:

- **Structured binary format** (not CSV) — faster, smaller, typed
- **Time-indexed** — precise timing of every event
- **Replayable** — reconstruct the full simulation from logs
- **Queryable** — extract specific episodes, time ranges, or events

## Log Format

### Schema (protobuf-like, but manual struct)

```python
LOG_ENTRY_TYPES = {
    'STATE': 0x01,    # Full state snapshot
    'ACTION': 0x02,   # Action taken
    'REWARD': 0x03,   # Reward received
    'EVENT': 0x04,    # Event (collision, goal reached, etc.)
    'METRIC': 0x05,   # Custom metric
}

# STATE entry format:
# [timestamp (f64)] [pos: 3xf32] [vel: 3xf32] [euler: 3xf32]
# [omega: 3xf32] [wind: 3xf32] [goal: 3xf32] = 56 bytes

# ACTION entry format:
# [timestamp (f64)] [action: 4xf32] = 24 bytes

# EVENT entry format:
# [timestamp (f64)] [event_type (u8)] [data: variable] = 9+ bytes
```

### Writer

```python
import struct
import time
import os

class TelemetryLogger:
    def __init__(self, filename, drone_id):
        self.f = open(filename, 'wb')
        self.drone_id = drone_id
        self.start_time = time.time()
        # Write header
        self.f.write(b'DRONELOG\x00')  # Magic
        self.f.write(struct.pack('!I', 1))  # Version
        self.f.write(struct.pack('!H', drone_id))  # Drone ID
        self.counts = {k: 0 for k in LOG_ENTRY_TYPES}

    def log_state(self, pos, vel, euler, omega, wind, goal):
        t = time.time() - self.start_time
        fmt = '!B d ' + 'f'*18  # type + time + 18 floats
        data = struct.pack(fmt,
            0x01, t,
            *pos, *vel, *euler, *omega, *wind, *goal)
        self.f.write(struct.pack('!I', len(data)))
        self.f.write(data)
        self.counts['STATE'] += 1

    def log_action(self, action):
        t = time.time() - self.start_time
        fmt = '!B d 4f'
        data = struct.pack(fmt, 0x02, t, *action)
        self.f.write(struct.pack('!I', len(data)))
        self.f.write(data)
        self.counts['ACTION'] += 1

    def log_event(self, event_type, message=b''):
        t = time.time() - self.start_time
        fmt = '!B d B ' + str(len(message)) + 's'
        data = struct.pack(fmt, 0x04, t, event_type, message)
        self.f.write(struct.pack('!I', len(data)))
        self.f.write(data)
        self.counts['EVENT'] += 1

    def close(self):
        self.f.close()
        print(f"Logged: {self.counts}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### Reader / Replayer

```python
class TelemetryReader:
    def __init__(self, filename):
        self.f = open(filename, 'rb')
        header = self.f.read(8)
        assert header == b'DRONELOG\x00', 'Bad magic'
        self.version = struct.unpack('!I', self.f.read(4))[0]
        self.drone_id = struct.unpack('!H', self.f.read(2))[0]

    def __iter__(self):
        return self

    def __next__(self):
        size_bytes = self.f.read(4)
        if not size_bytes:
            raise StopIteration
        size = struct.unpack('!I', size_bytes)[0]
        data = self.f.read(size)
        entry_type = data[0]
        if entry_type == 0x01:  # STATE
            t, *floats = struct.unpack('!d 18f', data[1:])
            return {
                'type': 'state', 'time': t,
                'pos': floats[0:3], 'vel': floats[3:6],
                'euler': floats[6:9], 'omega': floats[9:12],
                'wind': floats[12:15], 'goal': floats[15:18],
            }
        elif entry_type == 0x02:  # ACTION
            t, *action = struct.unpack('!d 4f', data[1:])
            return {'type': 'action', 'time': t, 'action': action}
```

## Implementation Task

### 1. Integrate Logging into Drone Process

Add logging to your drone simulator:

```python
class LoggedDroneProcess(DroneProcess):
    def __init__(self, drone_id, port, log_dir='logs/'):
        super().__init__(drone_id, port)
        os.makedirs(log_dir, exist_ok=True)
        self.logger = TelemetryLogger(
            f'{log_dir}/drone_{drone_id}_{int(time.time())}.bin',
            drone_id)

    def run_frame(self):
        # ... (existing logic)
        self.logger.log_state(
            self.env.drone.pos, self.env.drone.vel,
            self.env.drone.euler, self.env.drone.omega,
            self.env.wind.base_wind, self.env.goal)
        action, _, _ = self.policy.get_action(obs)
        self.logger.log_action(action)
        # ... step env, check done ...
```

### 2. Build `replay.py`

A tool that reads a log file and replays the simulation visually:

```python
def replay(log_file, speed=1.0):
    reader = TelemetryReader(log_file)
    # Use pygame or matplotlib to render the trajectory
    # Speed up or slow down with [ / ]
```

### 3. Build Log Analyzer

Create `analyze_logs.py` that computes:

- Total flight time per episode
- Average speed, max speed, max acceleration
- Number of near-misses (dist < 5m from obstacle)
- Energy consumption (∫thrust · velocity dt)
- Path efficiency (actual path / straight-line distance)

## Benchmark

```
Format   | 1 hour flight (60Hz) | Write speed | Read speed
---------+----------------------+-------------+------------
CSV      |      ~50 MB          |   ___ MB/s  |  ___ MB/s
Binary   |      ~20 MB          |   ___ MB/s  |  ___ MB/s
```

## Check-In Questions

1. The log format writes length-prefixed records. What's the
   advantage over CSV for time-series data?

2. How would you add compression to the binary format? What
   compression ratio would you expect for flight data?

3. The replay tool must re-create the simulation state exactly.
   What data is needed to deterministically replay? What if you
   logged actions but the physics timestep varied?

## 🤖 ML Connection

The telemetry log IS your ML training dataset:

- **Offline RL:** Train policies from logged experience (no
  simulator needed). The log contains state, action, reward tuples.
- **Imitation learning:** Expert trajectories (from PID or human
  pilot) can train a policy via behavioral cloning.
- **Dataset for supervised learning:** Predict next state from
  current state + action (learn a world model).
- **Experience replay buffer:** The log reader feeds directly into
  the RL training loop as a replay buffer.

**Next:** `05-feature-engineering-pipeline.md`
