# 🤖 Online Learning Loop

## Learning Objective

Implement an online learning system where drones continuously adapt
their policies mid-flight based on recent experience. This is the
bridge between offline training (Phase 03) and real-time adaptation.

## Why Online Learning?

The PPO policy trained in Phase 03 works well for the training
distribution. But the real world has:

- **Distribution shift:** Wind, obstacles, drone dynamics change
- **Novel situations:** Never-seen-before obstacle configurations
- **Degradation:** Motors wear, battery voltage drops

Online learning adapts the policy continuously:

```
Offline (Phase 03):     train → deploy → static
Online (Phase 07):      train → deploy → adapt → keep adapting
```

## Architecture

### Experience Buffer

Each drone maintains a ring buffer of recent experiences:

```python
class ExperienceBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (obs, action, reward, next_obs, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=256):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        return tuple(np.array(x) for x in zip(*batch))

    def __len__(self):
        return len(self.buffer)
```

### Online PPO Update

```python
class OnlinePPO:
    """PPO with online updates — adapts while flying."""

    def __init__(self, agent, buffer, update_interval=100):
        self.agent = agent
        self.buffer = buffer
        self.update_interval = update_interval
        self.steps = 0

    def infer(self, obs):
        """Get action from policy (no gradient)."""
        self.steps += 1
        action, log_prob, value = self.agent.get_action(obs)
        return action

    def observe(self, obs, action, reward, next_obs, done):
        """Store experience and maybe update."""
        self.buffer.add(obs, action, reward, next_obs, done)

        if self.steps % self.update_interval == 0 and len(self.buffer) >= 256:
            self._online_update()

    def _online_update(self):
        """Single PPO update from sampled batch."""
        obs_b, act_b, rew_b, next_b, done_b = self.buffer.sample(256)

        # Compute advantages (simplified — use GAE for full)
        next_values = self.agent.critic(torch.FloatTensor(next_b))
        values = self.agent.critic(torch.FloatTensor(obs_b))
        advantages = rew_b + 0.99 * next_values.detach().numpy() * (1 - done_b) \
                     - values.detach().numpy()

        # Single PPO update
        obs_t = torch.FloatTensor(obs_b)
        act_t = torch.FloatTensor(act_b)

        log_probs = self.agent.actor.log_prob(obs_t, act_t)
        old_log_probs = log_probs.detach()

        ratio = torch.exp(log_probs - old_log_probs)
        adv_t = torch.FloatTensor((advantages - advantages.mean()) /
                                  (advantages.std() + 1e-8))

        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t
        actor_loss = -torch.min(surr1, surr2).mean()

        values_pred = self.agent.critic(obs_t)
        critic_loss = nn.MSELoss()(
            values_pred, torch.FloatTensor(rew_b + 0.99 * next_values.numpy() * (1 - done_b)))

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * self.agent.actor.entropy(obs_t)

        self.agent.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
        self.agent.optimizer.step()
```

## Challenges with Online Learning

### 1. Stability

Online updates can destroy the policy. Solutions:

- **Small learning rates** (1e-5 vs 3e-4 for offline)
- **Trust region clipping** (PPO clip already helps)
- **KL penalty:** Add `-β · KL(π_new || π_old)` to the loss
- **Target network:** Keep a frozen copy, update slowly

### 2. Non-Stationarity

The data distribution changes as the policy improves. Solutions:

- **Importance sampling correction** (already in PPO via ratio r_t)
- **Larger replay buffers** to include older, diverse experience
- **Mixed training:** Interleave online data with offline pretraining data

### 3. Compute Budget

Each frame must be processed in <16.7ms (60Hz). Neural network updates
take tens of milliseconds. Solutions:

- **Async update thread:** Separate thread runs updates while main
  thread handles inference
- **Skipped updates:** Only update every K frames (K=100-1000)
- **Prioritized sampling:** Update more often on surprising transitions

## Implementation

### 1. Build `online_learning.py`

Complete online learning system:

```python
class OnlineLearningDrone:
    def __init__(self, pretrained_path='drone_policy.pth',
                 buffer_capacity=10000, update_interval=100):
        # Load pretrained policy
        self.agent = PPOAgent()
        self.agent.actor.load_state_dict(torch.load(pretrained_path))
        self.agent.optimizer = optim.Adam(
            list(self.agent.actor.parameters()) +
            list(self.agent.critic.parameters()), lr=1e-5)  # Small LR

        self.buffer = ExperienceBuffer(buffer_capacity)
        self.online = OnlinePPO(self.agent, self.buffer, update_interval)
        self.env = DroneEnv()
        self.policy_performance = []

    def run_episode(self):
        obs = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.online.infer(obs)
            next_obs, reward, done, info = self.env.step(action)

            self.online.observe(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

        self.policy_performance.append(total_reward)
        return total_reward
```

### 2. Benchmark: Static vs Online Policy

```
Environment     | Static Policy | Online (100 updates) | Online (1000 updates)
----------------+---------------+---------------+-----------------------
Training dist   |    500        |       510            |       515
+wind 10m/s     |    200        |       350            |       450
+new obstacles  |    150        |       280            |       420
+mass +50%      |    100        |       220            |       380
```

## Monitoring

Detect when online learning is helping vs hurting:

```python
class PolicyMonitor:
    def __init__(self, window=50):
        self.rewards = deque(maxlen=window)
        self.kls = deque(maxlen=window)

    def log(self, reward, kl_divergence):
        self.rewards.append(reward)
        self.kls.append(kl_divergence)

    @property
    def improving(self):
        if len(self.rewards) < 10:
            return True
        # Positive trend in last 10 rewards
        recent = list(self.rewards)[-10:]
        return recent[-1] > recent[0] * 1.1

    @property
    def policy_diverging(self):
        if len(self.kls) < 5:
            return False
        return np.mean(self.kls) > 0.1  # KL divergence too high
```

## Check-In Questions

1. Online learning updates use a 10x smaller learning rate than
   offline training. Why? What happens with the original learning rate?

2. The experience buffer stores (obs, action, reward, next_obs, done).
   These are sampled uniformly for updates. What sampling strategy
   would better prioritize "surprising" transitions?

3. KL divergence measures how much the policy changed. If KL is very
   small (~0.001) after 1000 updates, what does this mean? If it's
   very large (~0.5), what does this mean?

4. Online learning can cause catastrophic forgetting — the agent
   forgets how to handle situations it saw early in training. How
   would you detect and prevent this?

## Connection Back

This is the final link in the ML pipeline:

```
Phase 03: Train base policy (offline)  ────────────┐
                                                    ▼
Phase 04-05: Feature pipeline + inference server ──► Deploy to drone
                                                    │
Phase 04-07: Online learning loop ◄─────────────────┘ (continuous adaptation)
```

The result: a drone that not only flies, but learns to fly better
the longer it's in the air.

**Next:** `GATE-03.md`
