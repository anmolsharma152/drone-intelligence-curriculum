# 🤖 Distributed RL Training

## Learning Objective

Scale PPO training to 100+ parallel environments across multiple CPU
cores. Implement the full distributed training loop that collects
experience in parallel, aggregates it, and updates the policy.

## Architecture

```
                    ┌──────────────────────────┐
                    │      Learner (GPU)        │
                    │  - Aggregates experience  │
                    │  - PPO update             │
                    │  - Sends updated policy   │
                    └──────┬───────────────────┘
                           │ policy parameters (shared memory / Redis)
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Worker 1 │     │ Worker 2 │ ... │ Worker N │
    │ CPU core │     │ CPU core │     │ CPU core │
    │ Env × 4  │     │ Env × 4  │     │ Env × 4  │
    └──────────┘     └──────────┘     └──────────┘
    Total: N_workers × 4 = 100+ parallel environments
```

### Implementation Approaches

| Approach | Complexity | Throughput | Use Case |
|---|---|---|---|
| Python multiprocessing + Queue | Low | ~50K steps/s | Single machine |
| Ray RLlib | Medium | ~200K steps/s | Cluster |
| gRPC + custom worker | High | ~1M steps/s | Multi-node |
| Nvidia Isaac Gym | Low (GPU) | ~100M steps/s | GPU sim |

You'll implement the first approach (multiprocessing) — it's the most
educational and sufficient for this project.

## Distributed PPO Implementation

### Worker Process

```python
class PPOWorker(mp.Process):
    """Collects experience using current policy."""

    def __init__(self, rank, n_envs=4, shared_policy=None):
        super().__init__()
        self.rank = rank
        self.n_envs = n_envs
        self.envs = [DroneEnv() for _ in range(n_envs)]
        self.shared_policy = shared_policy  # Shared memory tensor

    def run(self):
        """Continuously collect experience."""
        from ppo import ActorNetwork
        local_actor = ActorNetwork()

        obs = [e.reset() for e in self.envs]
        while True:
            # Copy latest policy from shared memory
            local_actor.load_state_dict(self.shared_policy.state_dict())

            # Step all envs
            actions = []
            for i, ob in enumerate(obs):
                action, _, _ = local_actor.get_action(ob, deterministic=False)
                actions.append(action)
                next_ob, reward, done, info = self.envs[i].step(action)
                # Store experience in local buffer
                self.buffer.add(ob, action, reward, next_ob, done)
                obs[i] = next_ob if not done else self.envs[i].reset()

            # Send experience to learner periodically
            if len(self.buffer) >= 2048 // self.n_envs:
                self.send_experience()
```

### Learner Process

```python
class PPOLearner(mp.Process):
    """Aggregates experience and updates the policy."""

    def __init__(self, shared_policy, n_workers=8):
        super().__init__()
        self.shared_policy = shared_policy
        self.n_workers = n_workers
        self.experience_queues = [mp.Queue() for _ in range(n_workers)]

    def run(self):
        optimizer = optim.Adam(self.shared_policy.parameters(), lr=3e-4)
        batch_size = 2048 * self.n_workers  # Total experience per update

        while True:
            # Collect experience from all workers
            all_obs, all_act, all_rew, all_done = [], [], [], []
            for q in self.experience_queues:
                obs_b, act_b, rew_b, done_b = q.get()
                all_obs.append(obs_b)
                all_act.append(act_b)
                all_rew.append(rew_b)
                all_done.append(done_b)

            obs = np.concatenate(all_obs)
            act = np.concatenate(all_act)
            rew = np.concatenate(all_rew)
            done = np.concatenate(all_done)

            # PPO update (from Phase 03-08)
            stats = ppo_update(self.shared_policy, obs, act, rew, done)

            # Policy is already in shared memory — workers pick it up
            # on their next `load_state_dict` call

            print(f"Update: reward={rew.mean():.1f} "
                  f"entropy={stats['entropy']:.2f}")
```

### Shared Memory Policy

For the policy to be shared between processes, store it in shared
memory:

```python
import torch.multiprocessing as mp

class SharedPolicy:
    """Neural network policy stored in shared memory."""

    def __init__(self, obs_dim=24, act_dim=4):
        from ppo import ActorNetwork
        self.actor = ActorNetwork(obs_dim, act_dim)
        self.actor.share_memory()  # Makes all tensors shared

    def state_dict(self):
        return self.actor.state_dict()

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict)

    def parameters(self):
        return self.actor.parameters()
```

## Performance Targets

```
N Workers | N Envs/worker | Total Envs | Steps/sec | Hours to 10M steps
----------+---------------+------------+-----------+-------------------
    1     |       4       |     4      |   ~4K     |       ~0.7
    4     |       4       |    16      |   ~16K    |       ~0.17
    8     |       4       |    32      |   ~32K    |       ~0.09
   16     |       8       |   128      |   ~100K   |       ~0.03
```

## Implementation Task

### 1. Build `distributed_ppo.py`

Implement:

- SharedPolicy with `share_memory()`
- PPOWorker that copies policy, collects experience, sends to learner
- PPOLearner that aggregates experience, runs PPO update
- Main script that launches all processes

### 2. Train at Scale

Run distributed training:

```bash
python distributed_ppo.py --n_workers 4 --n_envs_per_worker 4
```

### 3. Benchmark

```
N Workers | Steps/sec | Wall time to 1M steps | Final reward
----------+-----------+-----------------------+-------------
   1      |   ___     |        ___            |    ___
   2      |   ___     |        ___            |    ___
   4      |   ___     |        ___            |    ___
   8      |   ___     |        ___            |    ___
```

## Check-In Questions

1. Shared memory allows the learner to update the policy while workers
   read it. What happens if a worker reads a partially-updated policy?
   (Hint: look at how PyTorch's `share_memory()` works.)

2. In this design, workers send full experience batches to the learner.
   The learner aggregates and updates. How does this compare to A3C
   (Asynchronous Advantage Actor-Critic), where workers compute
   gradients and send them?

3. If one worker is much slower than others (straggler effect), what
   happens in sync mode? In async mode? How would you handle it?

## Connection Back

Distributed training closes the ML pipeline:

```
Phase 03: Single env PPO proof of concept
Phase 04: Feature pipeline + inference server
Phase 05: Visualization and monitoring
Phase 06: Distributed training at scale

Result: A system that trains drone policies 100× faster
        than the single-env prototype.
```

**Next:** `06-sim-to-real-transfer.md`
