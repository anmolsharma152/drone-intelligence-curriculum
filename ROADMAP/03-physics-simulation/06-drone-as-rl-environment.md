# 🤖 Drone as an RL Environment

## Learning Objective

Wrap the full physics simulation as a Gym-compatible reinforcement
learning environment. Define state/action spaces and implement the
step/reset interface.

## The Gym Interface

OpenAI Gym defines a standard RL environment interface:

```python
class Env:
    def step(self, action) -> (obs, reward, done, info):
        """Apply action, advance simulation, return new state."""
        pass

    def reset(self) -> obs:
        """Reset to initial state, return first observation."""
        pass

    def render(self, mode='human'):
        """Visualize the environment."""
        pass
```

### For Your Drone

```
Observation Space (24 dimensions):
  [dx, dy, dz]         → 3  (relative position to goal)
  [vx, vy, vz]         → 3  (linear velocity)
  [φ, θ, ψ]            → 3  (attitude euler angles)
  [p, q, r]            → 3  (angular velocity)
  [wx, wy, wz]         → 3  (wind at drone position)
  [obs_1_x, y, d]      → 3  (nearest obstacle relative pos + dist)
  [obs_2_x, y, d]      → 3  (2nd nearest obstacle)
  [obs_3_x, y, d]      → 3  (3rd nearest obstacle)

Action Space (4 dimensions, continuous):
  [thrust]             → 1  (collective thrust, 0 to 20 N)
  [roll_cmd]           → 1  (desired roll torque, -1 to 1)
  [pitch_cmd]          → 1  (desired pitch torque, -1 to 1)
  [yaw_cmd]            → 1  (desired yaw rate, -1 to 1)
```

## Implementation

### 1. Build `drone_env.py`

```python
import gym
import numpy as np
from gym import spaces

class DroneEnv(gym.Env):
    """Gym environment for drone flight control."""

    def __init__(self, max_steps=2000, dt=1/60, render_mode=None):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode

        # Observation space: 24-dim continuous
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        # Action space: 4-dim continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Physics components (built in previous steps)
        self.drone = None     # RigidBody
        self.wind = None      # WindField
        self.controller = None  # PIDController (for reference)
        self.obstacles = []   # List of Sphere obstacles
        self.goal = np.array([0, 0, 5])

        self._init_world()

    def _init_world(self):
        """Reset world and drone to initial state."""
        from rigid_body import RigidBody
        from environment import WindField, generate_course

        self.drone = RigidBody(mass=1.0)
        self.drone.pos = np.array([-500, -500, 20])
        self.drone.vel = np.zeros(3)
        self.drone.euler = np.zeros(3)
        self.drone.omega = np.zeros(3)

        self.wind = WindField(base_wind=[2, 1, 0])
        self.obstacles = generate_course("forest", 30)
        self.goal = np.array([500, 500, 10])

    def _get_obs(self):
        """Build observation vector from current state."""
        drone = self.drone
        # Relative position to goal
        to_goal = self.goal - drone.pos

        # Wind at drone position
        wind = self.wind.wind_at(drone.pos, self.dt)

        # Find 3 nearest obstacles (use spatial index!)
        obs_distances = []
        for ob in self.obstacles:
            rel = ob.center - drone.pos
            d = np.linalg.norm(rel)
            obs_distances.append((d, rel / (d + 1e-8)))
        obs_distances.sort(key=lambda x: x[0])

        obs_vec = np.concatenate([
            to_goal,                           # 3
            drone.vel,                         # 3
            drone.euler,                       # 3
            drone.omega,                       # 3
            wind,                              # 3
            obs_distances[0][1] * [obs_distances[0][0]],  # 3 (nearest)
            obs_distances[1][1] * [obs_distances[1][0]],  # 3 (2nd)
            obs_distances[2][1] * [obs_distances[2][0]],  # 3 (3rd)
        ])
        return obs_vec.astype(np.float32)

    def _compute_reward(self):
        """Reward function — designed in file 07."""
        return 0.0  # Placeholder

    def step(self, action):
        """Apply action, step physics, return (obs, reward, done, info)."""
        # Scale actions to physical ranges
        thrust = (action[0] + 1) * 10.0  # [0, 20] N
        torques = action[1:4] * 2.0      # [-2, 2] N·m

        # Apply wind as external force
        wind = self.wind.wind_at(self.drone.pos, self.dt)
        wind_force = 0.1 * wind  # Simple drag model

        forces = np.array([0, 0, -thrust]) + wind_force
        self.drone.step_euler(forces, torques, self.dt)

        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()

        done = self._check_done()
        info = {
            'pos': self.drone.pos.copy(),
            'dist_to_goal': np.linalg.norm(self.goal - self.drone.pos),
            'steps': self.step_count
        }
        return obs, reward, done, info

    def _check_done(self):
        """Termination conditions."""
        if self.step_count >= self.max_steps:
            return True
        if np.linalg.norm(self.drone.pos - self.goal) < 5.0:
            return True  # Reached goal
        for ob in self.obstacles:
            if np.linalg.norm(self.drone.pos - ob.center) < ob.radius + 0.5:
                return True  # Collision
        if self.drone.pos[2] <= 0:
            return True  # Ground strike
        return False

    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._init_world()
        self.step_count = 0
        return self._get_obs()
```

### 2. Register the Environment

```python
# At the end of drone_env.py
from gym.envs.registration import register

register(
    id='DroneFlight-v0',
    entry_point='drone_env:DroneEnv',
    max_episode_steps=2000,
)
```

### 3. Test the Environment

```python
# test_env.py
import gym
import drone_env  # Registers the env

env = gym.make('DroneFlight-v0')
obs = env.reset()
print(f"Obs shape: {obs.shape}")  # Should be (24,)

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    print(f"Step {_:3d}: dist={info['dist_to_goal']:.1f}m")
    if done:
        print(f"Done after {_} steps: {info}")
        break
```

## Verification Tasks

1. **Sanity check:** Run 1000 random-action episodes. What's the
   average reward? Average steps survived?

2. **Action mapping:** Verify that action[0] = 1 produces ~20N thrust
   and action[0] = -1 produces ~0N thrust.

3. **Observation ranges:** For 1000 random steps, record min/max of
   each observation dimension. Are any exceeding reasonable bounds?

4. **Determinism:** Reset twice with the same seed. Are observations
   identical at step 0?

## Check-In Questions

1. The observation space is 24 dimensions. Most RL algorithms struggle
   above ~100 dimensions (curse of dimensionality). Is 24 OK? What's
   the minimum viable observation for a drone to navigate to a goal?

2. The action space is continuous (4 dimensions). Why can't we use
   discrete actions like "go left", "go right" for this task?

3. Termination conditions: we return `done=True` on collision. Is this
   always correct? What if we want the agent to learn to recover from
   near-collision?

4. The step function calls `self._compute_reward()` which currently
   returns 0. What happens if you train an agent with reward = 0 always?

## Next

The environment is ready. Now it needs a reward function that produces
the right behavior.

**Next:** `07-reward-shaping-for-flight.md`
