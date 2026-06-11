# Sensor Noise & Filtering

## Learning Objective

Model realistic GPS and IMU noise. Implement a complementary filter
and a Kalman filter for state estimation from noisy sensors.

## The Problem

The PID controller assumes perfect state knowledge. Real drones have:

| Sensor | Measurement | Noise | Rate |
|---|---|---|---|
| GPS | Position (lat, lon, alt) | ~3m RMS | 5-10 Hz |
| IMU accelerometer | Acceleration (body frame) | ~0.1 m/s² RMS | 100-400 Hz |
| IMU gyroscope | Angular velocity (body frame) | ~0.01 rad/s RMS | 100-400 Hz |
| Magnetometer | Heading (yaw) | ~5° RMS | 50-100 Hz |
| Barometer | Altitude | ~0.5m RMS | 25 Hz |
| Optical flow | Velocity (low altitude) | ~0.05 m/s | 50 Hz |

### Noise Models

```python
def noisy_gps(true_pos, noise_std=3.0):
    """GPS gives position with ~3m standard deviation."""
    return true_pos + np.random.normal(0, noise_std, size=3)

def noisy_imu(true_accel, true_gyro):
    """IMU with bias and noise."""
    accel_noise = np.random.normal(0, 0.1, size=3)
    gyro_noise = np.random.normal(0, 0.01, size=3)
    accel_bias = np.array([0.05, 0.05, 0.1])  # Slowly varying bias
    gyro_bias = np.array([0.001, 0.001, 0.005])
    return true_accel + accel_noise + accel_bias, \
           true_gyro + gyro_noise + gyro_bias
```

## Filtering Approaches

### 1. Complementary Filter (Simple, Low-Pass + High-Pass)

Best for attitude estimation: combine gyroscope (good high-frequency)
with accelerometer/magnetometer (good low-frequency).

```
angle = α · (angle + gyro_rate · dt) + (1 - α) · accel_angle
```

Where α is typically 0.96-0.98.

```python
class ComplementaryFilter:
    def __init__(self, alpha=0.96):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0

    def update(self, gyro, accel, dt):
        # Gyro integration (high frequency)
        self.roll += gyro[0] * dt
        self.pitch += gyro[1] * dt

        # Accelerometer gives gravity vector (low frequency)
        accel_roll = np.arctan2(accel[1], accel[2])
        accel_pitch = np.arctan2(-accel[0],
                                  np.sqrt(accel[1]**2 + accel[2]**2))

        # Complementary fusion
        self.roll = self.alpha * self.roll + (1 - self.alpha) * accel_roll
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * accel_pitch

        return self.roll, self.pitch
```

### 2. Kalman Filter (Optimal, More Complex)

The Kalman filter is the optimal linear estimator for Gaussian noise.

**Prediction step (time update):**
```
x̂ₖ|ₖ₋₁ = F · x̂ₖ₋₁|ₖ₋₁ + B · uₖ      (predict state)
Pₖ|ₖ₋₁ = F · Pₖ₋₁|ₖ₋₁ · Fᵀ + Q       (predict covariance)
```

**Update step (measurement update):**
```
ỹₖ = zₖ - H · x̂ₖ|ₖ₋₁                  (innovation/residual)
Sₖ = H · Pₖ|ₖ₋₁ · Hᵀ + R              (innovation covariance)
Kₖ = Pₖ|ₖ₋₁ · Hᵀ · Sₖ⁻¹              (Kalman gain)
x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + Kₖ · ỹₖ              (update state)
Pₖ|ₖ = (I - Kₖ · H) · Pₖ|ₖ₋₁         (update covariance)
```

Where:
- F: state transition matrix
- B: control input matrix
- H: measurement matrix
- Q: process noise covariance (how much we trust the model)
- R: measurement noise covariance (how much we trust the sensors)

## Implementation Task

### 1. Build `sensor_noise.py`

Implement GPS and IMU noise models as described above.

### 2. Build `kalman_filter.py`

Implement a simple 1D Kalman filter for altitude estimation:

```python
class KalmanFilter1D:
    """1D Kalman filter for altitude."""
    def __init__(self, process_noise=0.01, measurement_noise=9.0):
        # State: [position, velocity]
        self.x = np.array([0.0, 0.0])  # [pos, vel]
        self.P = np.eye(2) * 100.0     # Initial uncertainty
        self.F = np.array([[1, 1],     # dt = 1 (we handle dt scaling)
                           [0, 1]])
        self.H = np.array([[1, 0]])    # We measure position
        self.Q = np.eye(2) * process_noise
        self.R = measurement_noise

    def predict(self, dt):
        F = np.array([[1, dt], [0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, measurement):
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0]  # Filtered position

    def update_with_gps(self, gps_altitude, gps_std, dt):
        self.predict(dt)
        self.R = gps_std**2
        return self.update(gps_altitude)
```

### 3. Extend to 6-DOF State Estimation

Implement a full 6-DOF Kalman filter or Extended Kalman Filter (EKF)
that estimates:

```
State: [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]  (12 dimensions)
Input: [thrust, roll_torque, pitch_torque, yaw_torque]  (4 dimensions)
Measurement: GPS [x, y, z] + IMU [ax, ay, az, p, q, r]  (9 dimensions)
```

### 4. Benchmark: Filtered vs Raw vs True

Create `bench_filters.py` that:

1. Simulates a drone flying a figure-8 pattern for 60 seconds
2. Logs true state, noisy sensors, filtered estimate
3. Plots position error over time: raw GPS vs Kalman filtered
4. Measures: RMSE, max error, 95th percentile error

## Expected Results

```
Filter         | RMSE pos (m) | RMSE vel (m/s) | Max error (m)
---------------+--------------+-----------------+---------------
Raw GPS        |     3.2      |      N/A        |     9.8
Complementary  |     1.5      |      0.8        |     4.2
Kalman (1D)    |     0.9      |      0.4        |     2.8
Kalman (12D)   |     0.6      |      0.2        |     1.9
```

## Check-In Questions

1. The complementary filter parameter α controls the tradeoff between
   trusting the gyroscope vs accelerometer. What α value gives equal
   weight at what frequency?

2. The Kalman filter assumes Gaussian noise. What happens if the sensor
   has non-Gaussian noise (e.g., GPS multipath causing outliers)?
   What modification addresses this?

3. Why does the EKF linearize the system? What assumption about the
   state trajectory is required for the linearization to be valid?

4. The Kalman gain K balances "trust the prediction" vs "trust the
   measurement". Express K in terms of process noise Q and measurement
   noise R. What happens as Q → 0? As R → 0?

## 🤖 RL Connection

Sensor noise is critical for sim-to-real transfer:

```
Train policy on:      perfect state + noise
Deploy policy on:     Kalman-filtered estimated state
Gap:                  estimation errors → unexpected behavior
```

The solution: **Domain randomization** — during RL training, randomly
vary noise parameters (GPS std, IMU bias drift) so the policy learns
to be robust. You'll implement this in the RL environment.

**Embedded note:** The Kalman filter runs at 100-200Hz on a Pixhawk/
STM32 flight controller. The 12D EKF uses ~100 μs per update, leaving
plenty of CPU for higher-level control. On a Pi Pico (264KB SRAM),
you'd use the simpler complementary filter.

**Next:** `04-collision-avoidance-geometric.md`
