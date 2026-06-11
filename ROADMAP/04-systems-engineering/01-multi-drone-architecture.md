# Multi-Drone Architecture

## Learning Objective

Design a system where multiple drones run simultaneously, controlled
by a central ground station, communicating over a network protocol.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Ground Station                     │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Commander│  │ Telemetry│  │ ML Inference     │   │
│  │ (UI)     │  │ Dashboard│  │ Server (Phase 06)│   │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │              │                 │             │
│  ┌────▼──────────────▼─────────────────▼──────────┐ │
│  │            Protocol Handler (UDP)              │ │
│  └────────────────────┬───────────────────────────┘ │
└───────────────────────┼─────────────────────────────┘
                        │ Network (localhost or LAN)
    ┌───────────────────┼───────────────────┐
    │                   │                   │
┌───▼──────┐    ┌──────▼───┐    ┌─────────▼──┐
│ Drone 1  │    │  Drone 2  │    │  Drone N   │
│ Simulator│    │  Simulator│    │  Simulator │
└──────────┘    └──────────┘    └────────────┘
```

### Design Decisions

| Decision | Option A | Option B | Choice | Why |
|---|---|---|---|---|
| Comms | UDP | TCP | UDP | Lower latency, simpler, loss tolerance |
| Serialization | JSON | Protobuf | Protobuf | Smaller, faster, typed |
| Sync | Lockstep | Async | Async | Drones can run at different speeds |
| Discovery | Static config | mDNS | Static | Simple for local sim |

## Components

### Drone Simulator Process

Each drone runs as a separate Python process:

```python
# drone_process.py
class DroneProcess:
    def __init__(self, drone_id, udp_port):
        self.id = drone_id
        self.env = DroneEnv()        # From Phase 03
        self.policy = load_policy()  # From Phase 03 PPO training
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', udp_port))
        self.sock.setblocking(False)

    def run(self):
        obs = self.env.reset()
        while True:
            # Check for commands
            try:
                data, addr = self.sock.recvfrom(4096)
                cmd = parse_command(data)
                self.handle_command(cmd, obs)
            except BlockingIOError:
                pass

            # Step policy
            action, _, _ = self.policy.get_action(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)

            # Send telemetry
            self.send_telemetry(info)

            if done:
                obs = self.env.reset()

            time.sleep(1/60)
```

### Ground Station

```python
# ground_station.py
class GroundStation:
    def __init__(self):
        self.drones = {}  # drone_id -> (address, port, last_seen)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 9000))

    def send_command(self, drone_id, command):
        """Send waypoint or control command to specific drone."""
        data = serialize(command)
        addr = self.drones[drone_id]['address']
        self.sock.sendto(data, addr)

    def broadcast(self, command):
        """Send command to all drones."""
        for drone_id in self.drones:
            self.send_command(drone_id, command)

    def receive_telemetry(self):
        """Non-blocking receive from all drones."""
        try:
            while True:
                data, addr = self.sock.recvfrom(4096)
                telemetry = deserialize(data)
                self.drones[telemetry.id]['last_seen'] = time.time()
                self.drones[telemetry.id]['telemetry'] = telemetry
        except BlockingIOError:
            pass
```

## Implementation Task

### 1. Build the Multi-Drone Runner

Create `run_swarm.py` that:
1. Spawns N drone simulator processes
2. Each drone starts at a random position with a random goal
3. Each drone runs its PPO policy independently
4. Ground station receives telemetry from all drones
5. Prints a real-time table: drone positions, distances to goal

### 2. Build a Command Interface

Extend the ground station to accept keyboard commands:

```
[Space]  — Pause/resume all drones
[R]      — Reset a specific drone
[G]      — Set new goal for all drones
[1]-[9]  — Select specific drone for commands
[WASD]   — Manual override (take control of selected drone)
```

### 3. Benchmark

```
N Drones | Mean FPS (each) | Total entities | Network (KB/s)
---------+-----------------+----------------+---------------
   1     |    60.0         |      1         |     ___
   4     |    ___          |      4         |     ___
  16     |    ___          |     16         |     ___
  64     |    ___          |     64         |     ___
```

## Check-In Questions

1. UDP packets can be lost. If a command packet is lost, the drone
   continues with its previous goal. Is this acceptable? When would
   it not be?

2. Protobuf is mentioned for serialization but isn't installed.
   What's the simplest Python serialization you could use that's
   faster than JSON?

3. Each drone runs as a separate Python process. At what N does
   process overhead (memory, context switching) become a problem?
   What alternative architecture would you use for 1000 drones?

## 🤖 ML Connection

The multi-drone architecture directly enables:

- **Centralized training, distributed execution:** One training server
  updates a shared policy, deployed to all drones
- **Federated learning:** Each drone collects experience, sends
  gradient updates to the server (no raw data leaves drone)
- **Multi-agent RL:** Drones sharing airspace learn cooperative/
  competitive behaviors

**Next:** `02-ecs-pattern.md`
