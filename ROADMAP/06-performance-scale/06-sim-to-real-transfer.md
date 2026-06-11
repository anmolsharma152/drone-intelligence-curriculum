# 🤖 Sim-to-Real Transfer

## Learning Objective

Bridge the gap between simulation and real hardware. Understand domain
randomization, system identification, and the practical steps to
deploy a trained policy on a real drone.

## The Sim-to-Real Gap

A policy trained in simulation fails on real hardware because:

| Factor | Simulation | Reality |
|---|---|---|
| Physics | Perfect integrator | Friction, flex, motor dynamics |
| Sensors | Ground truth | Noise, bias, delay |
| Actuators | Instantaneous | Response time, saturation |
| Environment | Perfect model | Unknown wind, lighting, texture |
| Timing | Fixed dt | Variable, interrupts |

## Domain Randomization

The most effective technique: randomize simulation parameters during
training so the policy learns to be robust to variation.

```python
class RandomizedDroneEnv(DroneEnv):
    """Drone environment with domain randomization."""

    def _randomize_dynamics(self):
        """Randomize physics parameters."""
        # Mass: ±20%
        self.drone.mass = np.random.uniform(0.8, 1.2)

        # Inertia: ±30%
        scale = np.random.uniform(0.7, 1.3)
        self.drone.J *= scale
        self.drone.J_inv = np.linalg.inv(self.drone.J)

        # Motor thrust curve: thrust = a * cmd^2 + b * cmd
        self.thrust_a = np.random.uniform(8, 12)  # N per unit cmd²
        self.thrust_b = np.random.uniform(0, 2)

        # Drag coefficient: ±50%
        self.drag_coeff = np.random.uniform(0.05, 0.15)

        # Gravity: ±10% (simulates different planets/masses)
        self.drone.GRAVITY = np.array([0, 0, np.random.uniform(8.83, 10.79)])

    def _randomize_sensors(self):
        """Randomize sensor noise parameters."""
        # GPS noise: 1-10m std
        self.gps_noise_std = np.random.uniform(1.0, 10.0)

        # IMU noise: 0.01-0.1 rad/s
        self.gyro_noise_std = np.random.uniform(0.01, 0.1)

        # IMU bias drift rate
        self.accel_bias_drift = np.random.uniform(0.001, 0.01)

        # Sensor latency: 1-5 frames
        self.sensor_latency = np.random.randint(1, 5)

    def _randomize_environment(self):
        """Randomize environment conditions."""
        # Wind: 0-10 m/s from random direction
        wind_speed = np.random.uniform(0, 10)
        wind_dir = np.random.uniform(0, 2*np.pi)
        self.wind.base_wind = np.array([
            wind_speed * np.cos(wind_dir),
            wind_speed * np.sin(wind_dir),
            0.0
        ])

        # Obstacle positions: random
        self.obstacles = generate_course(
            np.random.choice(['forest', 'slalom', 'gate', 'canyon']),
            np.random.randint(10, 40))

        # Goal position: random within bounds
        self.goal = np.random.uniform(-500, 500, size=3)
        self.goal[2] = np.random.uniform(5, 30)  # Altitude

    def reset(self, seed=None):
        self._randomize_dynamics()
        self._randomize_sensors()
        self._randomize_environment()
        return super().reset(seed)
```

### Randomization Schedule

```python
class CurriculumRandomization:
    """Gradually increase randomization difficulty."""

    def __init__(self, total_steps=10_000_000):
        self.total = total_steps

    def noise_std(self, step):
        """Start low, increase over training."""
        progress = step / self.total
        return 0.1 + 9.9 * progress  # 0.1 → 10.0

    def wind_speed(self, step):
        """No wind → full wind over training."""
        progress = step / self.total
        return 10.0 * (1 - np.exp(-5 * progress))
```

## Policy Distillation for Edge Deployment

The trained PyTorch policy is too large for a microcontroller.
Distill it into a smaller model:

```python
class TinyPolicy(nn.Module):
    """Small enough for microcontroller inference."""

    def __init__(self, obs_dim=24, act_dim=4, hidden=32):
        super().__init__()
        # Single hidden layer, half the size
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        return self.fc2(x)  # Deterministic (no std for exploration)

def distill(teacher, student, n_samples=100000):
    """Train student to mimic teacher."""
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    for step in range(n_samples):
        obs = torch.randn(24)  # Sample from observation distribution
        with torch.no_grad():
            teacher_action, _ = teacher.get_action(obs, deterministic=True)

        student_action = student(obs)
        loss = nn.MSELoss()(student_action, teacher_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}: loss = {loss.item():.6f}")

    return student
```

### Export for Edge

```python
# Export to ONNX
def export_onnx(model, filename='drone_policy.onnx'):
    model.eval()
    dummy = torch.randn(1, 24)
    torch.onnx.export(model, dummy, filename,
                      input_names=['obs'],
                      output_names=['action'],
                      dynamic_axes={'obs': {0: 'batch'},
                                    'action': {0: 'batch'}},
                      opset_version=11)

# Export to TorchScript
def export_torchscript(model, filename='drone_policy.pt'):
    model.eval()
    example = torch.randn(1, 24)
    traced = torch.jit.trace(model, example)
    traced.save(filename)
```

## Real Hardware Path

### Minimum Viable System

```
Component          | Option (Cheap)       | Option (Pro)
-------------------+----------------------+-----------------------
Flight controller  | Pixhawk PX4 ($100)   | Custom STM32
Radio              | ESP32 + wifi ($5)    | SiK Telemetry ($50)
Compute for policy | Raspberry Pi 4 ($35) | Jetson Nano ($150)
Frame              | 250mm quad ($50)     | Custom carbon ($500)
```

### Integration

```python
# On Raspberry Pi / Jetson:
import onnxruntime as ort
import numpy as np

class OnboardPolicy:
    def __init__(self, model_path='drone_policy.onnx'):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, obs):
        # obs is numpy array (24,)
        result = self.session.run(None, {self.input_name: obs.astype(np.float32)})
        return result[0]  # (4,) action

# Convert sensor readings to observation vector
def build_observation(gps, imu, baro, goal):
    return np.array([
        *gps.position_rel,  # Relative to goal (3)
        *gps.velocity,       # (3)
        *imu.euler,          # (3)
        *imu.gyro,           # (3)
        *imu.wind_estimate,  # (3)
        *obstacle_sensor(),  # (9)
    ], dtype=np.float32)
```

## Sim-to-Real Checklist

- [ ] Domain randomization implemented for:
      - [ ] Mass and inertia
      - [ ] Motor thrust curve
      - [ ] Sensor noise (GPS, IMU)
      - [ ] Wind and disturbances
      - [ ] Obstacle configuration
- [ ] Policy trained with randomization
- [ ] Policy distilled to small model (< 50KB, < 100K params)
- [ ] Model exported to ONNX or TorchScript
- [ ] Model runs on target hardware at > 30Hz
- [ ] Observation pipeline matches simulator exactly
- [ ] Safety: emergency stop, geofence, failsafe

## Check-In Questions

1. Domain randomization relies on the "simulation" covering the range
   of real-world variation. If the real motor thrust curve is outside
   the randomized range, what happens? How do you find the right range?

2. Policy distillation compresses a 128×128 network to 32×32. The
   distilled policy might have slightly worse performance. How much
   degradation is acceptable? How do you test this before deployment?

3. The observation pipeline must exactly match between sim and real.
   The sim uses ground-truth state; real hardware uses Kalman-filtered
   estimates. If the Kalman filter introduces 100ms latency, what
   happens to the policy? How would you account for this in training?

4. Safety: a real drone that crashes can injure people or damage
   property. Describe a safety architecture that prevents the learned
   policy from causing harm.

## Connection Back

This is the final destination:

```
Phase 01-02: Algorithms and data structures
Phase 03:    Physics simulation + RL training
Phase 04:    Systems engineering + ML pipeline
Phase 05:    Visualization + policy insight
Phase 06:    Performance + scale + real deployment

            ┌─────────────────────────────┐
            │ A drone that learns to fly. │
            │ And you built everything.   │
            └─────────────────────────────┘
```

**Next:** `GATE-05.md`
