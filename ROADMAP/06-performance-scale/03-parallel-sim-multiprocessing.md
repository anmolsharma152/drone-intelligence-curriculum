# Parallel Simulation with Multiprocessing

## Learning Objective

Run multiple simulation instances in parallel across CPU cores to
generate training data faster.

## Why Parallel Sims?

RL training needs millions of environment steps. A single simulation
at 60Hz produces 3600 steps/minute. With 8 parallel simulations on
8 cores:

```
8 sims × 3600 steps/min = 28,800 steps/min
```

For a PPO training run needing 10M steps:

```
Single sim: 10M / 3600 = 46 hours
8 parallel: 10M / 28800 = 5.8 hours
```

## Multiprocessing Architecture

```
                    ┌─────────────────────┐
                    │   Training Server   │
                    │  (PyTorch, PPO opt) │
                    └────┬──────┬────┬────┘
                         │      │    │
         ┌───────────────┘      │    └───────────────┐
         ▼                      ▼                    ▼
    ┌─────────┐           ┌─────────┐          ┌─────────┐
    │ Sim 1   │           │ Sim 2   │   ...    │ Sim N   │
    │ (worker)│           │ (worker)│          │ (worker)│
    └─────────┘           └─────────┘          └─────────┘
```

### Communication

Use multiprocessing.Queue for sending actions and receiving
observations:

```python
import multiprocessing as mp
import numpy as np

def worker_process(rank, obs_queue, act_queue, n_steps):
    """Simulation worker process."""
    from drone_env import DroneEnv
    env = DroneEnv()

    for _ in range(n_steps):
        obs = env.reset() if done else obs
        obs_queue.put((rank, obs))

        action = act_queue.get()  # Wait for action from trainer
        obs, reward, done, info = env.step(action)
        obs_queue.put((rank, obs, reward, done, info))

    obs_queue.put((rank, None))  # Signal done

class ParallelEnvRunner:
    def __init__(self, n_workers=4):
        self.n_workers = n_workers
        self.obs_queues = [mp.Queue() for _ in range(n_workers)]
        self.act_queues = [mp.Queue() for _ in range(n_workers)]
        self.workers = []

    def start(self):
        for i in range(self.n_workers):
            p = mp.Process(target=worker_process,
                           args=(i, self.obs_queues[i],
                                 self.act_queues[i], 1000))
            p.start()
            self.workers.append(p)

    def get_observations(self):
        """Collect observations from all workers."""
        obs_list = []
        for q in self.obs_queues:
            obs_list.append(q.get())
        return obs_list

    def send_actions(self, actions):
        """Send actions to all workers."""
        for i, q in enumerate(self.act_queues):
            q.put(actions[i])

    def close(self):
        for p in self.workers:
            p.terminate()
```

## Sync vs Async Workers

### Sync (Lockstep)

All workers complete a step before the next step begins.

```
t=0: Workers all send obs → trainer computes actions → all receive → step
t=1: Workers all send obs → trainer computes actions → all receive → step
```

**Pro:** Simple, deterministic
**Con:** Slowest worker determines overall speed

### Async (Independent)

Workers run continuously; trainer processes batches as they arrive.

```
Worker 1: ──obs──act──obs──act──obs──act──
Worker 2: ────obs──act──obs──act──obs──act
Worker 3: ──────obs──act──obs──act──obs──act
          ^ trainer processes batches
```

**Pro:** Better hardware utilization
**Con:** Non-deterministic, stale policy for inference

## Implementation Task

### 1. Build `parallel_env.py`

Implement the parallel environment runner with:

- N worker processes (configurable)
- Sync mode for deterministic training
- Async mode for maximum throughput
- Graceful shutdown

### 2. Benchmark

```
N Workers | Steps/sec (sync) | Steps/sec (async) | Speedup vs single
----------+------------------+-------------------+------------------
   1      |      ___         |      ___          |      1.0×
   2      |      ___         |      ___          |      ___×
   4      |      ___         |      ___          |      ___×
   8      |      ___         |      ___          |      ___×
  16      |      ___         |      ___          |      ___×
```

### 3. Amdahl's Law Check

```
Fraction parallelizable = physics + env step (estimate: 95%)
Max speedup at:
  4 cores: S_max = 1 / (0.05 + 0.95/4) = ___×
  8 cores: S_max = 1 / (0.05 + 0.95/8) = ___×
 16 cores: S_max = 1 / (0.05 + 0.95/16) = ___×
```

Do your measurements match Amdahl's Law? If not, what's the bottleneck?

## Check-In Questions

1. Pickling (serialization) is required to send data between processes.
   NumPy arrays pickle efficiently; Python objects don't. What's the
   cost of sending a (24,) observation array vs a Drone object?

2. In sync mode, all workers wait for the slowest. If one worker is
   2× slower, what happens to overall throughput?

3. In async mode, the policy used for inference might be several
   updates behind the training policy. What's the impact on training
   stability?

## 🤖 Connection

Parallel environments are essential for modern RL:

- **PPO needs ~2000 steps per update** — parallel envs collect this
  in `2000 / (N_workers × fps)` seconds
- **Sample efficiency** — more parallel = more diverse data per batch
- **Distributed PPO (file 05)** builds directly on this infrastructure

**Next:** `04-10k-entities-optimization.md`
