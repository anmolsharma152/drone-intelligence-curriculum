# 6-DOF Kinematics

## Learning Objective

Model a drone as a 6-degree-of-freedom rigid body with position,
velocity, orientation, and angular velocity. Understand Euler
integration and its numerical properties.

## From 2D to 6-DOF

Your current simulation models drones as 2D points with (x, y) position
and (vx, vy) velocity. A real drone has:

```
State vector (12 dimensions):
  Position:      x,  y,  z         (3)
  Orientation:   φ, θ, ψ           (3)  (roll, pitch, yaw)
  Linear vel:    vx, vy, vz        (3)
  Angular vel:   p,  q,  r         (3)  (roll rate, pitch rate, yaw rate)
```

### Reference Frames

```
World frame (inertial, NED — North-East-Down):
  x: north, y: east, z: down (positive = altitude loss)

Body frame (attached to drone):
  x: forward, y: right, z: down
```

### Rotation Matrix

The orientation of the drone in the world frame is given by the rotation
matrix R (3×3), which is a function of (φ, θ, ψ):

```
R = R_z(ψ) · R_y(θ) · R_x(φ)

Where:
R_x(φ) = [[1, 0, 0],
          [0, cos φ, -sin φ],
          [0, sin φ,  cos φ]]

R_y(θ) = [[cos θ,  0, sin θ],
          [0,     1,     0],
          [-sin θ, 0, cos θ]]

R_z(ψ) = [[cos ψ, -sin ψ, 0],
          [sin ψ,  cos ψ, 0],
          [0,      0,     1]]
```

### Kinematic Equations

```
Position derivative (world frame):
  ṗ = v                            (linear velocity in world)

Velocity derivative (world frame):
  v̇ = (1/m) · R · F_body + g       (F_body = thrust + drag + disturbances)

Orientation derivative:
  Θ̇ = W · ω                        (ω = angular rates in body frame)

Where W converts body rates to Euler angle rates:
  W = [[1, sin φ tan θ,  cos φ tan θ],
       [0, cos φ,       -sin φ],
       [0, sin φ / cos θ, cos φ / cos θ]]

Angular acceleration:
  ω̇ = J^(-1) · (τ - ω × J · ω)    (Euler's equation for rotation)
```

### Euler Integration

The simplest numerical integration method:

```
state(t + dt) = state(t) + dt · state_derivative(t)

Properties:
- O(dt²) local truncation error
- O(dt) global error
- Energy can grow over time (not symplectic)
- Stable only if dt < 2 / max_eigenvalue
```

For drone simulation, Euler integration at dt = 1/60 ≈ 16.7ms is
usually sufficient. For higher accuracy, use **Runge-Kutta 4** (RK4):

```
k1 = f(state, t)
k2 = f(state + dt/2 · k1, t + dt/2)
k3 = f(state + dt/2 · k2, t + dt/2)
k4 = f(state + dt · k3, t + dt)
next_state = state + dt/6 · (k1 + 2·k2 + 2·k3 + k4)
```

RK4 has O(dt⁵) local error, allowing larger timesteps for the same
accuracy.

## Implementation Task

### 1. Build `rigid_body.py`

```python
import numpy as np

class RigidBody:
    """6-DOF rigid body drone."""
    GRAVITY = np.array([0, 0, 9.81])  # m/s², z-down

    def __init__(self, mass=1.0, inertia=None):
        self.mass = mass
        self.J = inertia or np.eye(3) * 0.01  # Inertia matrix
        self.J_inv = np.linalg.inv(self.J)

        # State
        self.pos = np.zeros(3)      # [x, y, z]
        self.vel = np.zeros(3)      # [vx, vy, vz]
        self.euler = np.zeros(3)    # [roll, pitch, yaw]
        self.omega = np.zeros(3)    # [p, q, r] body rates

    def rotation_matrix(self):
        """Compute R from euler angles."""
        phi, theta, psi = self.euler
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        Rx = np.array([[1, 0, 0],
                       [0, cphi, -sphi],
                       [0, sphi, cphi]])
        Ry = np.array([[cth, 0, sth],
                       [0, 1, 0],
                       [-sth, 0, cth]])
        Rz = np.array([[cpsi, -spsi, 0],
                       [spsi, cpsi, 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def state_derivative(self, forces_body, torques_body):
        """Compute state derivative given body-frame forces and torques."""
        R = self.rotation_matrix()
        phi, theta = self.euler[0], self.euler[1]

        # Linear acceleration (world frame)
        accel_world = R @ forces_body / self.mass + self.GRAVITY

        # Angular acceleration (body frame)
        Jw = self.J @ self.omega
        alpha = self.J_inv @ (torques_body - np.cross(self.omega, Jw))

        # Euler angle rates from body rates
        W = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        euler_dot = W @ self.omega

        return np.concatenate([
            self.vel,          # pos derivative
            accel_world,       # vel derivative
            euler_dot,         # euler derivative
            alpha              # omega derivative
        ])

    def step_euler(self, forces_body, torques_body, dt):
        """Forward Euler step."""
        deriv = self.state_derivative(forces_body, torques_body)
        # Unpack derivative
        dpos = deriv[0:3]
        dvel = deriv[3:6]
        deuler = deriv[6:9]
        domega = deriv[9:12]

        self.pos += dt * dpos
        self.vel += dt * dvel
        self.euler += dt * deuler
        self.omega += dt * domega
```

### 2. Verify with Simple Forces

Test your rigid body with simple force inputs:

1. **Hover:** Apply thrust = mg upward. Drone should maintain altitude.
2. **Forward flight:** Apply pitch moment + forward thrust.
3. **Yaw turn:** Apply yaw torque only.

Plot position and attitude over 10 seconds for each.

### 3. Compare Euler vs RK4

Implement RK4 step and compare:
- Run both with dt = 0.01 for 10 seconds (a "ground truth" trajectory)
- Run Euler with dt = 0.0167 (60Hz) and dt = 0.05 (20Hz)
- Measure position error vs RK4 baseline at each step
- Plot error vs time for each timestep

## Check-In Questions

1. At what bank angle (φ) does a drone start losing altitude rather than
   turning? Derive the relationship.

2. The rotation matrix uses Euler angles, which have a singularity at
   θ = ±90°. What is this called and how is it handled in practice?

3. For a drone with mass 0.5kg and inertia 0.005 kg·m², what is the
   maximum stable Euler timestep? (Hint: consider the fastest dynamics —
   angular velocity.)

4. Why does Euler integration cause energy drift? Give a physical
   intuition.

## Connection Back

Your current simulation has 2D (x, y) particles. 6-DOF is the foundation
for realistic drone physics. Every subsequent component (PID, sensors,
RL environment) depends on this.

**Embedded note:** A real IMU (MPU6050/BNO055) returns raw accelerometer
and gyroscope readings at 100-400Hz. Your 6-DOF state would be estimated
from these via sensor fusion — coming in Section 03.

**🤖 RL note:** The 12-dimensional state vector is your RL observation
space. Every physics feature you add expands what the agent can perceive.

**Next:** `02-pid-control.md`
