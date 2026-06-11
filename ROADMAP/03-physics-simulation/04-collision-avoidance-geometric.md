# Geometric Collision Avoidance

## Learning Objective

Implement collision detection and geometric collision avoidance for
drones navigating through obstacle fields.

## Collision Detection

### Shape Representations

For obstacles, use simple geometric primitives:

```python
class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def contains(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

class AABB:
    """Axis-Aligned Bounding Box."""
    def __init__(self, min_corner, max_corner):
        self.min = np.array(min_corner)
        self.max = np.array(max_corner)

    def contains(self, point):
        return np.all(point >= self.min) and np.all(point <= self.max)
```

### Drone as a Sphere

Model the drone as a sphere with radius r (say 0.5m). Collision occurs
when distance from drone center to obstacle surface < r:

```python
def collision_detected(drone_pos, obstacle_spheres, drone_radius=0.5):
    for obs in obstacle_spheres:
        dist = np.linalg.norm(drone_pos - obs.center)
        if dist < obs.radius + drone_radius:
            return True, obs
    return False, None
```

### Spatial Acceleration

For N obstacles and M drones, naive collision is O(N·M). Your spatial
structures from Phase 02 solve this:

```python
def check_collisions_fast(drones, obstacles, spatial_index):
    """Use QuadTree/Grid to find near obstacles for each drone."""
    for drone in drones:
        # Query spatial index for obstacles within drone_radius + max_obs_r
        nearby = spatial_index.query(
            drone.pos, max_obstacle_radius + drone.drone_radius)
        for obs in nearby:
            if np.linalg.norm(drone.pos - obs.center) < obs.radius + drone.drone_radius:
                drone.handle_collision(obs)
```

## Collision Avoidance (Geometric)

### 1. Potential Fields

Treat obstacles as repulsive forces, the goal as an attractive force:

```python
def potential_field_force(drone_pos, goal_pos, obstacles, repulsion_gain=10.0):
    # Attractive force (pull toward goal)
    to_goal = goal_pos - drone_pos
    attractive = to_goal / np.linalg.norm(to_goal) * 5.0  # 5 m/s²

    # Repulsive forces (push away from obstacles)
    repulsive = np.zeros(3)
    for obs in obstacles:
        to_obs = obs.center - drone_pos
        dist = np.linalg.norm(to_obs)
        if dist < obs.radius + 2.0:  # Within safety margin
            if dist < 0.01:
                dist = 0.01  # Avoid division by zero
            repulsive += -repulsion_gain * (to_obs / dist) / (dist * dist)

    return attractive + repulsive
```

**Vulnerability:** Local minima — can get stuck when repulsive forces
exactly cancel attractive force.

### 2. Velocity Obstacle (VO)

Compute collision cones in velocity space:

```python
def velocity_obstacle(drone_pos, drone_vel, drone_r, obstacle_pos,
                      obstacle_vel, obstacle_r, time_horizon=3.0):
    """Returns True if current velocity leads to collision."""
    relative_pos = obstacle_pos - drone_pos
    relative_vel = drone_vel - obstacle_vel
    combined_r = drone_r + obstacle_r

    # Check if relative velocity points within collision cone
    # Use law of sines to find angle of collision cone
    dist = np.linalg.norm(relative_pos)
    cone_half_angle = np.arcsin(combined_r / dist)

    angle_to_obstacle = np.arctan2(relative_pos[1], relative_pos[0])
    velocity_angle = np.arctan2(relative_vel[1], relative_vel[0])

    angular_diff = abs(velocity_angle - angle_to_obstacle)
    if angular_diff < cone_half_angle:
        # On collision course — need to steer
        return True, cone_half_angle - angular_diff  # Severity
    return False, 0.0
```

### 3. RRT with Collision Checks (Reinforcement)

Connect back to Phase 02 RRT: the collision check in RRT already uses
line-of-sight sampling against obstacles. After implementing geometric
collision, you can give the RRT planner realistic collision radii.

## Implementation Task

### 1. Build `collision.py`

```python
import numpy as np
from quadtree import QuadTree, Rect

class CollisionWorld:
    def __init__(self, world_size=4000):
        self.size = world_size
        self.obstacles = []
        self.tree = QuadTree(Rect(0, 0, world_size, world_size), 4)
        self.tree_points = []  # HACK: QuadTree stores particles, not spheres

    def add_obstacle(self, center, radius):
        obs = Sphere(center, radius)
        self.obstacles.append(obs)
        # For spatial indexing, store center as point with radius
        # (requires generalizing QuadTree or using Grid)
        return obs

    def generate_random_obstacles(self, n=20, min_r=50, max_r=200):
        """Fill world with random circular obstacles."""
        for _ in range(n):
            x = np.random.uniform(-self.size, self.size)
            y = np.random.uniform(-self.size, self.size)
            z = np.random.uniform(0, 100)  # Altitude 0-100m
            r = np.random.uniform(min_r, max_r)
            self.add_obstacle((x, y, z), r)

    def check(self, drone_pos, drone_radius=0.5):
        for obs in self.obstacles:
            if np.linalg.norm(drone_pos - obs.center) < obs.radius + drone_radius:
                return True
        return False

    def nearest_obstacle_distance(self, drone_pos):
        """Distance to nearest obstacle surface."""
        min_dist = float('inf')
        for obs in self.obstacles:
            d = np.linalg.norm(drone_pos - obs.center) - obs.radius
            min_dist = min(min_dist, d)
        return min_dist
```

### 2. Build `potential_field_avoidance.py`

Implement the potential field method. Test by placing 10 obstacles
between start (-200, -200, 10) and goal (200, 200, 10). Run the
simulation and plot the trajectory.

### 3. Benchmark

Compare collision avoidance methods:

```
Method              | Success Rate | Avg Path Length | Avg Time | Computation
--------------------+--------------+-----------------+----------+------------
Potential Field     |    ___%      |    ___ m        |  ___ s   |   ___ μs
Velocity Obstacle   |    ___%      |    ___ m        |  ___ s   |   ___ μs
RRT (from Phase 02) |    ___%      |    ___ m        |  ___ s   |   ___ ms
```

## Check-In Questions

1. The potential field method has local minima. Prove that for two
   obstacles placed symmetrically between start and goal, the drone
   can get stuck. How would you detect and escape this?

2. Velocity Obstacles assume constant velocity for the obstacle. What
   happens when an obstacle changes direction? How does RVO (Reciprocal
   Velocity Obstacles) address this for multi-drone scenarios?

3. For a drone with radius 0.5m approaching a wall at 10 m/s, what is
   the minimum braking distance assuming maximum deceleration of 5 m/s²?
   What safety margin should be added for the controller lag (PID, 50ms)?

## 🤖 RL Integration

Geometric avoidance is a baseline. An RL agent can learn more
sophisticated strategies:

```
Geometric: dodge all obstacles with fixed rules
RL:        learn to dodge, and learn when to brake vs accelerate
           through a gap based on uncertainty
```

In your RL environment (files 06-08), the obstacle field becomes part
of the observation space. The agent must learn to navigate through it.

**Next:** `05-environment-obstacles-wind.md`
