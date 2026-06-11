# Live Dashboard & Telemetry

## Learning Objective

Build a real-time dashboard that displays telemetry data from all
drones simultaneously: altitude, speed, attitude, reward, and policy
confidence.

## Dashboard Architecture

```
Drone Processes ──UDP──► Ground Station ──► Dashboard (matplotlib animation)
                            │
                            ▼
                      Log file (redundant storage)
```

## Implementation

### 1. Build `dashboard.py`

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

class TelemetryDashboard:
    def __init__(self, n_drones=4, history_seconds=30, fps=10):
        self.n_drones = n_drones
        self.history = history_seconds * fps  # Samples to keep
        self.fps = fps
        self.data = {
            d: {
                'time': deque(maxlen=self.history),
                'altitude': deque(maxlen=self.history),
                'speed': deque(maxlen=self.history),
                'heading': deque(maxlen=self.history),
                'dist_to_goal': deque(maxlen=self.history),
                'reward': deque(maxlen=self.history),
                'value': deque(maxlen=self.history),
                'action_thrust': deque(maxlen=self.history),
                'action_roll': deque(maxlen=self.history),
            } for d in range(n_drones)
        }

        self.fig = plt.figure(figsize=(16, 10))
        self._setup_layout()

    def _setup_layout(self):
        """Create subplot layout."""
        gs = self.fig.add_gridspec(4, 4)

        # Top row: 3D position scatter for all drones
        self.ax_3d = self.fig.add_subplot(gs[0, :], projection='3d')

        # Middle: altitude, speed, heading (one per drone in same plot)
        self.ax_alt = self.fig.add_subplot(gs[1, 0:2])
        self.ax_speed = self.fig.add_subplot(gs[1, 2:4])

        # Bottom left: individual drone detail
        self.ax_attitude = self.fig.add_subplot(gs[2, 0])
        self.ax_action = self.fig.add_subplot(gs[2, 1])
        self.ax_reward = self.fig.add_subplot(gs[2, 2])

        # Bottom right: policy confidence
        self.ax_value = self.fig.add_subplot(gs[2, 3])

        # Bottom row: per-drone metrics table
        self.ax_table = self.fig.add_subplot(gs[3, :])
        self.ax_table.axis('off')

    def update(self, frame):
        """Called by matplotlib animation at self.fps Hz."""
        self._poll_telemetry()  # Read from UDP
        self._update_plots()
        self._update_table()
        return []

    def _poll_telemetry(self):
        """Read telemetry from ground station UDP buffer."""
        # From ground_station.py: self.telemetry dict
        pass

    def _update_plots(self):
        self.ax_alt.clear()
        self.ax_speed.clear()

        for d in range(self.n_drones):
            t = list(self.data[d]['time'])
            alt = list(self.data[d]['altitude'])
            speed = list(self.data[d]['speed'])

            color = f'C{d}'
            self.ax_alt.plot(t, alt, color=color,
                           label=f'Drone {d}')
            self.ax_speed.plot(t, speed, color=color,
                             label=f'Drone {d}')

        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.legend()
        self.ax_speed.set_ylabel('Speed (m/s)')
        self.ax_speed.legend()

    def _update_table(self):
        """Show latest metrics for all drones."""
        rows = []
        for d in range(self.n_drones):
            latest = {k: v[-1] if v else 0
                      for k, v in self.data[d].items()}
            rows.append([
                f'Drone {d}',
                f'{latest["altitude"]:.1f}m',
                f'{latest["speed"]:.1f}m/s',
                f'{latest["dist_to_goal"]:.0f}m',
                f'{latest["reward"]:+.1f}',
                f'{latest["value"]:.2f}',
            ])

        self.ax_table.clear()
        self.ax_table.axis('off')
        table = self.ax_table.table(
            cellText=rows,
            colLabels=['ID', 'Alt', 'Speed', 'Dist', 'Reward', 'Value'],
            loc='center',
            cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)

    def run(self):
        anim = animation.FuncAnimation(
            self.fig, self.update, interval=1000//self.fps)
        plt.tight_layout()
        plt.show()
```

### 2. Add Per-Drone Detail Panel

When clicking on a drone in the table, show its detail panel:

```
Detail Panel for Drone 2:
┌─────────────────────────────────────────┐
│ Attitude: roll=2.1° pitch=-1.3° yaw=45°│
│ Controls: throttle=0.65 roll=0.1 ...    │
│ Policy confidence: 87%                  │
│ Last 10 rewards: +5, +3, -2, ...        │
│ Current action Q-values: [0.3, 0.5, ...]│
│ Observation features: [24 values]       │
└─────────────────────────────────────────┘
```

## Implementation Task

### 1. Build the Dashboard

- Receive live telemetry via UDP from running drone processes
- Plot altitude, speed, heading over a rolling 30-second window
- Show a table of current metrics for all drones
- Add keyboard shortcuts: [1]-[9] select drone for detail

### 2. Add Stress Markers

Visual indicators when things go wrong:

```python
def stress_markers(self):
    markers = []
    for d in range(self.n_drones):
        if self.data[d]['dist_to_goal'][-1] < 10:
            markers.append(('GOAL', d, 'green'))
        if self.data[d]['altitude'][-1] < 1:
            markers.append(('LOW ALT', d, 'red'))
        # Policy uncertainty
        value = self.data[d]['value'][-1]
        if abs(value) < 0.1:
            markers.append(('UNCERTAIN', d, 'yellow'))
    return markers
```

## Check-In Questions

1. The dashboard updates at 10Hz while the sim runs at 60Hz. Why not
   update at 60Hz? What limits dashboard update rate?

2. matplotlib animation has a fixed interval but the actual frame time
   varies. How does `FuncAnimation` handle variable compute time?

3. The dashboard receives UDP telemetry. If a packet is lost, the plot
   misses a frame. How would you handle missing data in the time-series
   plots?

## 🤖 Connection

The dashboard becomes an ML monitoring tool:
- Plot training reward curves live
- Show policy value estimates vs true returns
- Highlight state space regions where the policy is uncertain
- Detect reward hacking visually

**Next:** `03-policy-visualization.md`
