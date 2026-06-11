# 🤖 Model Serving Infrastructure

## Learning Objective

Build a low-latency inference server that deploys trained policies to
drones. Handle multiple simultaneous inference requests, batching, and
model versioning.

## Architecture

```
                     ┌──────────────────┐
                     │  Model Registry   │
                     │  ┌──┐ ┌──┐ ┌──┐  │
                     │  │v1│ │v2│ │v3│  │
                     │  └──┘ └──┘ └──┘  │
                     └────────┬─────────┘
                              │
┌─────────┐  gRPC/HTTP  ┌────▼──────────┐  batch   ┌──────────────┐
│ Drone 1 ├────────────►│ Inference      ├─────────►│ GPU (if av)  │
├─────────┤             │ Server         │          │ or CPU       │
│ Drone 2 ├────────────►│ (torch.no_grad)│          └──────────────┘
├─────────┤             └────────────────┘
│ Drone N │
└─────────┘
```

## Server Implementation

### 1. Build `inference_server.py`

```python
import torch
import numpy as np
import threading
import queue
import time
from collections import defaultdict

class InferenceServer:
    """Multi-model inference server with batching."""

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.models = {}       # model_name -> (actor, critic, version)
        self.lock = threading.Lock()
        self.request_queue = queue.Queue()
        self.batch_size = 1
        self.running = True

    def load_model(self, name, actor_path, version=1):
        """Load a trained policy from disk."""
        from ppo import ActorNetwork
        actor = ActorNetwork(obs_dim=24, act_dim=4)
        actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        actor.to(self.device)
        actor.eval()
        with self.lock:
            self.models[name] = (actor, version)
        print(f"Loaded model '{name}' v{version}")

    def unload_model(self, name):
        with self.lock:
            self.models.pop(name, None)

    def infer(self, model_name, obs, deterministic=True):
        """Single observation inference (thread-safe)."""
        with self.lock:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not loaded")
            actor, version = self.models[model_name]

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = actor(obs_t)
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        return action.squeeze(0).numpy(), version

    def infer_batch(self, model_name, obs_batch, deterministic=True):
        """Batch inference for multiple observations."""
        with self.lock:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not loaded")
            actor, version = self.models[model_name]

        obs_t = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        with torch.no_grad():
            mean, std = actor(obs_t)
            if deterministic:
                actions = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
        return actions.numpy(), version
```

### 2. Batching Worker

For maximum throughput, batch requests arriving in a short window:

```python
class BatchingWorker(threading.Thread):
    """Collects requests into batches for GPU utilization."""

    def __init__(self, server, max_batch=32, timeout_ms=5):
        super().__init__(daemon=True)
        self.server = server
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000

    def run(self):
        while self.server.running:
            batch = []
            deadline = time.time() + self.timeout

            # Collect requests until batch full or timeout
            while len(batch) < self.max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    req = self.server.request_queue.get(
                        timeout=remaining)
                    batch.append(req)
                except queue.Empty:
                    break

            if not batch:
                continue

            # Group by model name
            by_model = defaultdict(list)
            for req in batch:
                by_model[req['model']].append(req)

            for model_name, requests in by_model.items():
                obs_batch = [r['obs'] for r in requests]
                actions, version = self.server.infer_batch(
                    model_name, obs_batch)

                for req, action in zip(requests, actions):
                    req['callback'](action, version)
```

### 3. Performance Benchmark

```python
# bench_inference.py
server = InferenceServer()
server.load_model('drone_policy', 'drone_policy.pth')

# Warmup
for _ in range(10):
    server.infer('drone_policy', np.random.randn(24))

# Benchmark
obs = np.random.randn(24)
times = []
for _ in range(1000):
    t0 = time.perf_counter()
    action, _ = server.infer('drone_policy', obs)
    times.append(time.perf_counter() - t0)

print(f"Single inference: {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
print(f"Throughput: {1000/np.mean(times):.0f} inferences/sec")

# Batch benchmark
for batch_size in [1, 4, 16, 64, 256]:
    obs_batch = np.random.randn(batch_size, 24)
    t0 = time.perf_counter()
    actions, _ = server.infer_batch('drone_policy', obs_batch)
    dt = time.perf_counter() - t0
    print(f"Batch size {batch_size:3d}: {dt*1000:.2f}ms total, "
          f"{dt/batch_size*1e6:.1f}μs per inference")
```

### Expected Results

```
Single inference: 0.8ms ± 0.1ms (CPU)
Throughput: ~1250 inferences/sec

Batch size   1: 0.8ms total, 800.0μs/inference
Batch size   4: 1.0ms total, 250.0μs/inference
Batch size  16: 1.5ms total,  93.8μs/inference
Batch size  64: 2.8ms total,  43.8μs/inference
Batch size 256: 8.0ms total,  31.3μs/inference
```

## Model Versioning & A/B Testing

```python
class ModelRouter:
    """Route drone inference requests to different model versions."""

    def __init__(self, server):
        self.server = server
        self.routes = {}  # drone_id -> model_name

    def assign_model(self, drone_id, model_name):
        self.routes[drone_id] = model_name

    def canary_deploy(self, new_model, fraction=0.1):
        """Deploy new model to 10% of drones for testing."""
        for i, drone_id in enumerate(self.routes):
            if i / len(self.routes) < fraction:
                self.routes[drone_id] = new_model

    def infer(self, drone_id, obs):
        model = self.routes.get(drone_id, 'default')
        return self.server.infer(model, obs)
```

## Integration with Drone Server

```python
class InferenceDroneProcess:
    def __init__(self, drone_id, inference_server):
        self.id = drone_id
        self.server = inference_server
        self.env = DroneEnv()
        self.model_name = 'drone_policy_v2'

    def run_frame(self):
        obs = self.env._get_obs()
        action, model_version = self.server.infer(
            self.model_name, obs, deterministic=True)
        obs, reward, done, info = self.env.step(action)
        # Log which model version was used
        return obs, reward, done, {'model_version': model_version, **info}
```

## Check-In Questions

1. Batch inference gives 10-20x throughput improvement over single
   inference for the same compute. Why? (Hint: think about matrix
   multiplication on GPU vs CPU.)

2. The batching worker uses a 5ms timeout. What happens to latency
   for the first request in a batch? What happens to throughput?

3. In the canary deploy strategy, 10% of drones get the new model.
   What metric would you monitor to decide whether to roll out to
   100%?

4. The inference server runs on CPU. After batching, each inference
   costs ~44μs. At 60Hz, each drone needs 16.7ms per frame. How many
   drones can a single CPU core serve?

## Connection Back

```
Phase 03: Train PPO on single drone
Phase 04-06: Serve trained policy to N drones in production
             with batching, versioning, A/B testing
```

The inference server closes the loop: training happens offline, but
inference runs online at 60Hz for every drone.

**Next:** `07-online-learning-loop.md`
