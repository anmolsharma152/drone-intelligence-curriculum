# 🤖 Reward Shaping for Drone Flight

## Learning Objective

Design reward functions that produce desired flight behaviors:
navigation, obstacle avoidance, smooth flight, and energy efficiency.

## The Reward Design Problem

Reward design is arguably the hardest part of RL. The reward function
defines "what good looks like." A poorly designed reward produces
unexpected or degenerate behavior.

### Sparse vs Dense Rewards

**Sparse reward:** +1 for reaching goal, 0 otherwise.
- Pro: Simple, no reward hacking
- Con: Agent never sees a nonzero reward in early episodes → no learning

**Dense reward:** Continuous feedback at every step.
- Pro: Agent always gets signal, learns faster
- Con: Easy to misspecify, leading to reward hacking

**Strategy:** Start dense for learning, then shape toward sparse for
robustness.

## Reward Components

### 1. Goal Proximity

```python
def goal_proximity(dist_to_goal, prev_dist):
    """Reward progress toward goal."""
    progress = prev_dist - dist_to_goal  # Positive if moving closer
    return 10.0 * progress  # Scale to ~1-10 per step
```

**Risk:** Agent oscillates around goal without reaching it (no
terminal reward for actually arriving).

### 2. Goal Arrival Bonus

```python
def goal_arrival(dist_to_goal, threshold=5.0):
    """Large bonus for reaching the goal."""
    if dist_to_goal < threshold:
        return 100.0
    return 0.0
```

### 3. Collision Penalty

```python
def collision_penalty(min_obstacle_dist, collision_radius=1.0):
    """Heavy penalty for collision (or near-miss)."""
    if min_obstacle_dist < collision_radius:
        return -100.0
    elif min_obstacle_dist < 3.0:  # Near miss
        return -10.0 * (3.0 - min_obstacle_dist) / 2.0
    return 0.0
```

### 4. Smoothness Penalty

```python
def smoothness_penalty(action, prev_action):
    """Penalize jerky control inputs."""
    return -0.1 * np.sum((action - prev_action)**2)
```

### 5. Energy Penalty

```python
def energy_penalty(thrust, velocity):
    """Minimize energy consumption."""
    # Power ∝ thrust * speed (simplified)
    power = thrust * np.linalg.norm(velocity)
    return -0.01 * power
```

### 6. Altitude Penalty

```python
def altitude_penalty(z, target_z=10.0):
    """Stay near target altitude."""
    return -0.1 * (z - target_z)**2
```

## Complete Reward Function

```python
class DroneEnv(gym.Env):
    # ... (from file 06)

    def _compute_reward(self):
        drone = self.drone
        prev_dist = getattr(self, '_prev_goal_dist',
                            np.linalg.norm(self.goal - drone.pos))
        dist = np.linalg.norm(self.goal - drone.pos)

        # 1. Goal proximity
        reward = 10.0 * (prev_dist - dist)

        # 2. Goal arrival
        if dist < 5.0:
            reward += 100.0
            print(f"GOAL REACHED at step {self.step_count}")

        # 3. Collision checking
        min_obs_dist = float('inf')
        for ob in self.obstacles:
            d = np.linalg.norm(drone.pos - ob.center) - ob.radius
            if d < 0.5:  # Collision!
                reward -= 100.0
                # Don't print here — breaks the "done" logic
            if d < min_obs_dist:
                min_obs_dist = d
        if min_obs_dist < 3.0:  # Near-miss penalty
            reward -= 5.0 * (3.0 - min_obs_dist) / 3.0

        # 4. Smoothness
        if hasattr(self, '_prev_action'):
            reward -= 0.5 * np.sum((self._prev_action - self._current_action)**2)

        # 5. Altitude discipline
        reward -= 0.1 * (drone.pos[2] - 10.0)**2

        self._prev_goal_dist = dist
        return reward
```

## Reward Hacking: What Can Go Wrong?

| Reward Design | What the Agent Learns | Why It's Wrong |
|---|---|---|
| `+100 for dist < 5m` | Oscillate around 5.1m until max steps, then zip in | Exploits step limit for free bonus |
| `+10 * progress` | Fly in circles gaining small increments | Progress toward then away from goal |
| `-100 for collision` | Stay still at start | Collision avoidance fails to differentiate action quality |
| `-0.1 * altitude^2` | Descend to z=0 then crash | Terminates with ground collision, "solves" episode |

### Fixes

1. **Shaping reward must be potential-based:**
   `F(s, s') = γ·Φ(s') - Φ(s)` where Φ is a potential function.
   This guarantees the shaped reward doesn't change the optimal policy.

2. **Add "don't just stand there" penalty:**
   `-0.01 * steps` to incentivize completion.

3. **Normalize rewards** to a stable range (~[-10, 10]) for neural
   network training.

## Implementation Task

### 1. Implement 5 Reward Variants

```python
reward_variants = {
    'sparse': lambda env: 100.0 if env._at_goal() else 0.0,
    'dense_progress': lambda env: 10.0 * (env._prev_dist - env._dist),
    'dense_full': lambda env: env._dense_full_reward(),
    'potential': lambda env: env._potential_based_reward(),
    'curriculum': lambda env: env._curriculum_reward(env.step_count),
}
```

### 2. Benchmark

For each variant, run 1000 random episodes and record:

```
Variant         | Avg Reward | Avg Steps | Goal Rate | Collision Rate
----------------+------------+-----------+-----------+---------------
sparse          |  _____     |  _____    |  _____%   |  _____%
dense_progress  |  _____     |  _____    |  _____%   |  _____%
dense_full      |  _____     |  _____    |  _____%   |  _____%
potential       |  _____     |  _____    |  _____%   |  _____%
curriculum      |  _____     |  _____    |  _____%   |  _____%
```

### 3. Domain Randomization

Make the training robust by randomizing:

```python
def _randomize_physics(self):
    """Randomize physics parameters each episode."""
    self.drone.mass = np.random.uniform(0.8, 1.2)  # ±20%
    self.wind.base_wind = np.random.uniform(-5, 5, size=3)
    self.drone.inertia *= np.random.uniform(0.8, 1.2)
```

## Check-In Questions

1. Prove that potential-based shaping doesn't change the optimal
   policy. (Hint: the shaped Q-function differs from the true Q by a
   term that doesn't depend on action.)

2. Why does the agent learn to oscillate at the goal boundary with
   `+100 for dist < 5m`? What fix prevents this?

3. In the curriculum variant, you might start with no wind and sparse
   obstacles, then increase difficulty. What's the theoretical
   justification for curriculum learning?

4. The smoothness penalty `-(a_t - a_{t-1})^2` creates a dependency
   between consecutive actions. How does this affect the Markov
   assumption in RL?

## Connection Back

You now have:
- A Gym environment (file 06)
- A reward function (file 07)
- A physics world with obstacles and wind (files 01-05)

The final piece: an agent that learns to fly.

**Next:** `08-rl-agents-from-scratch.md`
