# PID Control

## Learning Objective

Implement PID (Proportional-Integral-Derivative) controllers for
drone flight stabilization: altitude hold, velocity control, and
attitude stabilization.

## Why PID?

Open-loop control (applying forces directly) is unstable — small
disturbances (wind, gravity, sensor noise) cause the drone to drift.
Closed-loop control corrects errors by feeding back measurements.

PID is the simplest effective closed-loop controller:

```
u(t) = Kp · e(t) + Ki · ∫e(τ)dτ + Kd · de/dt
```

Where:
- e(t) = setpoint - measurement (the error)
- Kp: proportional gain — responds to current error
- Ki: integral gain — eliminates steady-state error
- Kd: derivative gain — dampens oscillations, improves settling

## Controlling a Drone

A drone typically has cascaded PID controllers:

```
Position loop (outer, 5-10 Hz):
  desired_pos → PID → desired_vel

Velocity loop (mid, 20-50 Hz):
  desired_vel → PID → desired_attitude (roll/pitch angles)

Attitude loop (inner, 100-400 Hz):
  desired_attitude → PID → motor mixing → motor speeds
```

Each outer loop runs slower and generates setpoints for the inner loop.
The inner loop must be at least 2x faster (preferably 5-10x) for
stability.

### Your Implementation

You'll implement three PID controllers:

1. **Altitude PID:** z-position → collective thrust
2. **Velocity PID:** (vx, vy) desired → roll/pitch angles
3. **Yaw rate PID:** ψ desired → yaw torque

```
class PID:
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limit = output_limit

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
```

### Tuning Method: Ziegler-Nichols

1. Set Ki = Kd = 0. Increase Kp until the system oscillates at
   constant amplitude — this is Ku (ultimate gain).
2. Measure the oscillation period Tu.
3. Use the table:

| Controller | Kp | Ki | Kd |
|---|---|---|---|
| P | 0.5·Ku | — | — |
| PI | 0.45·Ku | 0.54·Ku/Tu | — |
| PID | 0.6·Ku | 1.2·Ku/Tu | 0.075·Ku·Tu |

## Implementation Task

### 1. Build `pid_controller.py`

Implement the PID class above plus a cascaded controller:

```python
class CascadedController:
    """Position → Velocity → Attitude cascade."""
    def __init__(self, drone_mass=1.0):
        self.alt_pid = PID(2.0, 0.1, 0.5, output_limit=5.0)
        self.vel_x_pid = PID(1.0, 0.05, 0.2, output_limit=np.radians(30))
        self.vel_y_pid = PID(1.0, 0.05, 0.2, output_limit=np.radians(30))
        self.yaw_pid = PID(0.5, 0.01, 0.1, output_limit=2.0)
        self.mass = drone_mass

    def compute(self, state, target_pos, target_yaw, dt):
        """Compute body-frame forces and torques from state and target."""
        # Altitude control (z is down in NED)
        z_error = target_pos[2] - state.pos[2]
        thrust = self.mass * 9.81 + self.alt_pid.update(z_error, dt)

        # Velocity control → desired attitude
        vx_error = target_pos[0] - state.pos[0]  # Simple P for position
        vy_error = target_pos[1] - state.pos[1]
        # In practice, use a proper position PID output as vel setpoint

        desired_roll = self.vel_y_pid.update(vy_error, dt)
        desired_pitch = -self.vel_x_pid.update(vx_error, dt)

        # Attitude control → torques (simplified: P only)
        roll_torque = 0.1 * (desired_roll - state.euler[0] - 1.0 * state.omega[0])
        pitch_torque = 0.1 * (desired_pitch - state.euler[1] - 1.0 * state.omega[1])
        yaw_torque = self.yaw_pid.update(target_yaw - state.euler[2], dt)

        # Body frame forces: thrust is along body z-axis
        R = state.rotation_matrix()
        force_body = np.array([0, 0, -thrust])  # Thrust up in body frame
        torque_body = np.array([roll_torque, pitch_torque, yaw_torque])

        return force_body, torque_body
```

### 2. Test with RigidBody

Create a test script that:

1. Spawns a drone at (0, 0, 10) — 10m altitude
2. Targets position (100, 50, 5) with yaw = 0
3. Runs the simulation with PID control for 30 seconds
4. Plots: position vs time, attitude vs time, thrust vs time

Expected behavior:
- Drone descends to 5m smoothly (altitude PID)
- Drone flies to (100, 50) with controlled velocity (velocity PID)
- Drone maintains level attitude during hover (attitude PID)
- Settling time < 5 seconds with < 10% overshoot

### 3. Disturbance Rejection

Add a wind disturbance of 5 m/s in the +x direction for 2 seconds.
Plot the position error. The controller should recover after the
disturbance stops.

### 🤖 RL Note

The PID controller is a baseline. In the RL phase, you'll compare it
against a learned policy:

```
PID policy: u = f(error, ∫error, derror)  [hand-tuned gains]
RL policy:  u = π(s)                       [learned from experience]
```

Questions for later:
- Can RL beat PID on this task?
- What happens when the drone dynamics change (payload mass increases)?
- Can RL handle nonlinearities that PID struggles with?

## Check-In Questions

1. What happens if Ki is too large? What happens if Kd is too large?

2. Why must the inner attitude loop run faster than the outer position
   loop? What specific instability occurs if this is violated?

3. In the cascaded controller, the velocity PID outputs desired roll/
   pitch angles. Why is this correct? Derive the relationship between
   horizontal acceleration and bank angle for a fixed-wing vs multirotor.

4. Derivative kick: when the setpoint changes abruptly, the derivative
   term spikes. What's a common fix?

## Complexity Analysis

Each PID update is O(1) — three multiplications and a summation.
For a full cascaded controller (5 PIDs), that's 5 · O(1) = O(1) per
timestep. No asymptotic bottleneck.

However: PID tuning is O(k) where k is the number of tuning attempts.
Ziegler-Nichols gives a starting point, but optimal tuning for
coupled systems (roll/pitch coupling, motor dynamics) can require
many iterations.

## Connection Back

Your 6-DOF rigid body can now fly under PID control. The next step
adds realistic sensor noise — because in the real world, the drone
doesn't know its state perfectly. It must estimate it from noisy
sensor readings.

**Next:** `03-sensor-noise-filtering.md`
