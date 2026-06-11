# Environment: Obstacle Fields & Wind

## Learning Objective

Build the environmental simulation that the drone operates in: wind
fields, procedurally generated obstacle courses, and multi-drone
interaction. This completes the "physics world" before wrapping it
as an RL environment.

## Wind Model

### Constant Wind + Gusts

```python
class WindField:
    def __init__(self, base_wind=np.array([0, 0, 0]),
                 gust_strength=2.0, gust_frequency=0.5):
        self.base = np.array(base_wind)
        self.gust_strength = gust_strength
        self.gust_freq = gust_frequency
        self.time = 0.0

    def wind_at(self, position, dt):
        self.time += dt
        # Gust as Perlin-like turbulence (simplified)
        gust = np.zeros(3)
        for i in range(3):
            gust[i] = self.gust_strength * math.sin(
                self.time * self.gust_freq + position[i] * 0.001)

        # Shear near ground (wind drops near z=0 due to terrain)
        shear = 1.0 - math.exp(-position[2] / 10.0)

        return self.base + gust * shear
```

### Dryden Turbulence Model (More Realistic)

For real aircraft simulation, the Dryden model shapes white noise
through a transfer function to produce turbulence with realistic
spatial correlation:

```
Φ_u(Ω) = σ²_u · (2L_u/π) · 1/(1 + (L_u · Ω)²)

Where:
- σ_u: turbulence intensity (m/s)
- L_u: turbulence scale length (m) — ~530m at low altitude
- Ω: spatial frequency (rad/m)
```

## Generator: Procedural Obstacle Courses

Create a function that generates structured obstacle courses, not just
random placements:

```python
def generate_course(style="slalom", obstacles=20):
    """Generate structured obstacle courses."""
    if style == "slalom":
        # Zigzag: obstacles alternating left and right
        return [Sphere((i*200 - 1000, (-1)**i * 150, 20), 50)
                for i in range(obstacles)]

    elif style == "canyon":
        # Narrow corridor with walls
        obstacles = []
        for i in range(obstacles):
            wall_side = random.choice(["left", "right"])
            if wall_side == "left":
                obstacles.append(Sphere((-200 + i*80, -300, 20), 60))
            else:
                obstacles.append(Sphere((-200 + i*80, 300, 20), 60))
        return obstacles

    elif style == "forest":
        # Random tall thin obstacles
        return [Sphere((random.uniform(-1000, 1000),
                        random.uniform(-1000, 1000),
                        15), random.uniform(10, 30))
                for _ in range(obstacles)]

    elif style == "gate":
        # Gate: obstacles forming arches to fly through
        obstacles = []
        for i in range(0, obstacles, 4):
            x = i * 100 - 500
            # Left pillar, right pillar, top beam
            obstacles.append(Sphere((x, -30, 15), 10))
            obstacles.append(Sphere((x, 30, 15), 10))
            obstacles.append(Sphere((x, 0, 35), 10))
        return obstacles
```

## Multi-Drone Interaction

For multiple drones in the same airspace:

```python
class Airspace:
    def __init__(self, size=4000):
        self.drones = []
        self.wind = WindField()
        self.obstacle_field = []
        self.time = 0.0

    def step(self, dt):
        self.time += dt
        wind_map = {}  # Cache wind per drone position
        for drone in self.drones:
            pos = drone.state.pos
            wind = self.wind.wind_at(pos, dt)
            wind_map[id(drone)] = wind
            drone.apply_wind(wind)
            drone.step(dt)

        # Check inter-drone collisions
        # (Use spatial index from Phase 02 for efficiency)
        for i, a in enumerate(self.drones):
            for j, b in enumerate(self.drones):
                if i >= j: continue
                dist = np.linalg.norm(a.state.pos - b.state.pos)
                if dist < a.radius + b.radius:
                    self.handle_drone_collision(a, b)
```

## Implementation Task

### 1. Build `environment.py`

Implement the full environment:
- WindField with gusts and shear
- Obstacle course generation (4 styles)
- Airspace class with multi-drone stepping
- Spatial-index accelerated collision queries (reuse QuadTree/Grid)

### 2. Visualization

Create `visualize_environment.py`:

1. 3D plot of the obstacle course (matplotlib's `ax.scatter` or
   `plot_surface` for spherical obstacles)
2. Wind vectors at grid points (quiver plot)
3. Drone trajectory overlay after flying through the course

### 3. Benchmark

Compare simulation speed with and without spatial acceleration for
collision queries:

```
Drones | Obstacles | No Spatial | QuadTree | Grid
-------+-----------+------------+----------+-------
   1   |    10     |   ___ ms   |  ___ ms  | ___ ms
   1   |   100     |   ___ ms   |  ___ ms  | ___ ms
  10   |   100     |   ___ ms   |  ___ ms  | ___ ms
 100   |  1000     |   ___ ms   |  ___ ms  | ___ ms
```

## Check-In Questions

1. The Dryden turbulence model has a spatial correlation length L_u.
   What does this mean physically, and how does it affect drone control
   at different wind speeds?

2. In the slalom course, what is the optimal path through alternating
   obstacles? How does your chosen avoidance method perform?

3. For multi-drone operation with N drones, what is the complexity of
   inter-drone collision checking with and without spatial indexing?

## 🤖 RL Connection

The environment is the MDP (Markov Decision Process) that the RL agent
interacts with:

```
State:   [drone_pos, drone_vel, drone_euler, drone_omega,
          wind_at_drone, nearest_obstacle_vector × num_sensors]
Action:  [thrust, roll_torque, pitch_torque, yaw_torque]
Reward:  [to be designed in file 07]
```

Everything in this file feeds the RL environment:
- Wind field → disturbance in dynamics
- Obstacle courses → collision penalties, success rewards
- Multi-drone → cooperative/competitive RL scenarios

**Next:** `06-drone-as-rl-environment.md`
