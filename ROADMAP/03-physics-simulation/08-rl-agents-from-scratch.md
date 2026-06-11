# 🤖 RL Agents from Scratch

## Learning Objective

Implement PPO (Proximal Policy Optimization) from scratch in PyTorch.
Train it on your drone environment to fly from start to goal through
obstacles.

## Why PPO?

PPO is the most widely-used RL algorithm for continuous control:

- **Stable:** Clipped objective prevents catastrophic policy updates
- **Sample efficient:** Reuses data for multiple epochs
- **Simple:** No complicated trust region optimization (like TRPO)
- **Proven:** Works on drone/robot control tasks

### Other Options

| Algorithm | Continuous? | On/Off Policy | Why (Not) Choose |
|---|---|---|---|
| DQN | No | Off | Only discrete actions |
| DDPG | Yes | Off | Sensitive to hyperparameters |
| SAC | Yes | Off | SOTA but more complex |
| PPO | Yes | On | Best balance of simplicity + performance |
| A2C/A3C | Yes | On | Synchronous PPO is simpler |

## PPO Algorithm

### Core Idea

PPO maximizes a **clipped surrogate objective**:

```
L^CLIP(θ) = E_t[min(r_t(θ) · A_t,
                    clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) — probability ratio
- A_t = advantage estimate (how much better was this action than average)
- ε = clip range (typically 0.2)

The clipping prevents the policy from changing too much in one update.

### Actor-Critic Architecture

```
Actor (policy network):
  obs (24) → Linear(128) → ReLU → Linear(128) → ReLU →
    mean (4)    ← action distribution
    log_std (4) ← learned exploration noise

Critic (value network):
  obs (24) → Linear(128) → ReLU → Linear(128) → ReLU → value (1)
```

### GAE (Generalized Advantage Estimation)

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

Where δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

λ = 0.95 trades bias vs variance:
- λ → 0: high bias, low variance (like TD(0))
- λ → 1: low bias, high variance (like Monte Carlo)

## Implementation

### 1. Build `ppo.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorNetwork(nn.Module):
    """Gaussian policy: outputs mean, uses learned log_std."""
    def __init__(self, obs_dim=24, act_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean, torch.zeros_like(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs, actions):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)

    def entropy(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum(dim=-1).mean()

class CriticNetwork(nn.Module):
    """Value function: estimates expected return from state."""
    def __init__(self, obs_dim=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim=24, act_dim=4, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.actor = ActorNetwork(obs_dim, act_dim)
        self.critic = CriticNetwork(obs_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def get_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(obs_t, deterministic)
            value = self.critic(obs_t)
        return action.numpy(), log_prob.numpy(), value.numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) \
                    - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, trajectories):
        """Update policy using collected trajectories."""
        obs, actions, old_log_probs, rewards, dones, values = trajectories
        next_value = self.critic(torch.FloatTensor(obs[-1:])).item()
        advantages, returns = self.compute_gae(
            rewards, values, dones, next_value)

        obs_t = torch.FloatTensor(np.array(obs))
        act_t = torch.FloatTensor(np.array(actions))
        old_log_t = torch.FloatTensor(np.array(old_log_probs))
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)

        # Normalize advantages (stabilizes training)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Multiple epochs
        for _ in range(10):
            log_probs = self.actor.log_prob(obs_t, act_t)
            ratio = torch.exp(log_probs - old_log_t)

            # Clipped surrogate objective
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (MSE)
            values_pred = self.critic(obs_t)
            critic_loss = nn.MSELoss()(values_pred, ret_t)

            # Entropy bonus (exploration)
            entropy = self.actor.entropy(obs_t)

            total_loss = actor_loss + self.value_coef * critic_loss \
                         - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_adv': adv_t.mean().item(),
        }
```

### 2. Build `train_ppo.py`

```python
import gym
import numpy as np
import drone_env
from ppo import PPOAgent
from collections import deque

env = gym.make('DroneFlight-v0')
agent = PPOAgent(obs_dim=24, act_dim=4)

N_STEPS = 2048   # Trajectory length per update
N_UPDATES = 1000 # Total updates

reward_history = deque(maxlen=100)

for update in range(N_UPDATES):
    trajectories = [[] for _ in range(6)]  # obs, act, logp, rew, done, val
    obs = env.reset()
    ep_rewards = []
    ep_lengths = []

    # Collect trajectory
    for step in range(N_STEPS):
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)

        trajectories[0].append(obs)
        trajectories[1].append(action)
        trajectories[2].append(log_prob)
        trajectories[3].append(reward)
        trajectories[4].append(done)
        trajectories[5].append(value)

        obs = next_obs
        ep_rewards.append(reward)

        if done:
            reward_history.append(sum(ep_rewards))
            ep_rewards = []
            obs = env.reset()

    # Update agent
    stats = agent.update(trajectories)

    if update % 10 == 0:
        avg_reward = np.mean(reward_history) if reward_history else 0
        print(f"Update {update:4d} | "
              f"Avg Reward: {avg_reward:+7.1f} | "
              f"Entropy: {stats['entropy']:.2f} | "
              f"Actor Loss: {stats['actor_loss']:.2f}")

# Save policy
torch.save(agent.actor.state_dict(), 'drone_policy.pth')
```

## Training Process

### Expected Learning Curve

```
Update Range | Avg Reward | Behavior
-------------+------------+----------
  0 - 50     |  -500      | Random flailing, crashes immediately
 50 - 200    |  -200      | Learning to throttle up, directional control
200 - 500    |    50      | Can fly toward goal but still crashes
500 - 1000   |   300      | Reliable goal-reaching, avoids most obstacles
1000+        |   500+     | Smooth flight, efficient paths
```

### Diagnostics

Plot these over training:
1. **Reward curve** (smoothed) — should increase then plateau
2. **Entropy** — should decrease as policy becomes confident
3. **Value loss** — should decrease then stabilize
4. **KL divergence** — should stay within clip range
5. **Episode length** — should increase (agent survives longer)

## Evaluation

```python
def evaluate(agent, env, episodes=10):
    """Measure trained policy performance."""
    rewards = []
    successes = 0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total = 0
        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += reward
            if info.get('dist_to_goal', float('inf')) < 5.0:
                successes += 1
        rewards.append(total)
    return {
        'mean_reward': np.mean(rewards),
        'success_rate': successes / episodes,
        'std_reward': np.std(rewards),
    }
```

### Compare to PID Baseline

| Metric | PID Controller | RL Policy (after training) |
|---|---|---|
| Success rate (10 runs) | ___% | ___% |
| Avg flight time | ___ s | ___ s |
| Avg path length | ___ m | ___ m |
| Energy use | ___ J | ___ J |
| Max wind resistance | ___ m/s | ___ m/s |

## Extensions

### 1. Recurrent PPO

Add an LSTM layer to the policy: the agent can "remember" past
observations, crucial for coping with sensor noise (the Kalman filter
is effectively a fixed recurrent structure — why not learn one?).

### 2. Multi-Agent PPO

Train two drones: one chases, one evades. Use parameter sharing for
efficiency.

### 3. Curriculum Learning

Start with no wind, sparse obstacles, then gradually increase
difficulty as the agent improves:

```python
def curriculum_step(env, performance):
    if performance > 100:
        env.wind.base_wind = np.random.uniform(-3, 3)
    if performance > 200:
        env.max_obstacles = 50
```

## Check-In Questions

1. The probability ratio r_t(θ) can be > 1 (new policy assigns higher
   probability to the action) or < 1 (new policy assigns lower
   probability). Why does clipping at [1-ε, 1+ε] prevent destructive
   updates?

2. GAE uses a parameter λ to trade bias vs variance. What happens if
   λ = 0? What happens if λ = 1? Which would you use for a task with
   very noisy rewards?

3. The entropy bonus encourages exploration. As training progresses,
   entropy should naturally decrease (policy becomes more certain).
   If entropy drops to 0 early in training, what's happening and what
   hyperparameter should you adjust?

4. Why do we normalize advantages within each batch? What happens if
   we don't?

## Connection Back

You have now built:
- A drone physics simulation (6-DOF)
- Sensor noise and filtering
- A Gym environment wrapper
- A reward function
- A PPO agent trained from scratch

This is the core AI pipeline. Phase 04 takes it to production:
multi-drone systems, feature pipelines, model serving, and online
learning.

**Next:** `GATE-02.md`
