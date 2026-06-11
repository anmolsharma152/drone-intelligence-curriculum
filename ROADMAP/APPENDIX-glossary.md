# Glossary

Terms and concepts used throughout the curriculum, roughly ordered by
phase of first appearance.

## Phase 01: Foundations

| Term | Definition |
|---|---|
| **Big-O** | Asymptotic upper bound. `f(n) = O(g(n))` means there exist
  c, n₀ such that `f(n) ≤ c·g(n)` for all `n ≥ n₀`. |
| **Ω (Omega)** | Asymptotic lower bound. |
| **Θ (Theta)** | Asymptotic tight bound — both O and Ω. |
| **Amortized analysis** | Average cost per operation over a sequence,
  even if individual operations are expensive. |
| **Space complexity** | Memory usage as a function of input size. |
| **Euler integration** | Forward numerical integration: `y(t+dt) = y(t) + y'(t)·dt`. O(dt) global error. |

## Phase 02: Algorithms

| Term | Definition |
|---|---|
| **PR-Quadtree** | Point-Region QuadTree. Points stored only in leaf
  nodes; internal nodes only have children, no points. |
| **Uniform Grid** | Fixed-size 2D array of cells. Each cell contains
  points in that spatial region. O(1) insert, tunable query. |
| **KD-Tree** | K-dimensional tree. Binary tree that splits on one
  dimension at each level, cycling through dimensions. |
| **Nearest neighbor (NN)** | Find the closest point to a query point
  under a distance metric. |
| **Range query** | Find all points within a given spatial region. |
| **A\*** | Best-first graph search using `f(n) = g(n) + h(n)`.
  Optimal when `h` is admissible. |
| **Admissible heuristic** | Never overestimates the true cost to goal. |
| **Consistent heuristic** | `h(n) ≤ cost(n, n') + h(n')` for all
  neighbors n' of n. Also called monotone. |
| **RRT** | Rapidly-exploring Random Tree. Samples random points in
  continuous space, steers toward them from nearest tree node. |
| **RRT\*** | Optimal variant of RRT. Rewires the tree to minimize
  path cost; probabilistically converges to optimal. |
| **Probabilistic completeness** | As iterations → ∞, probability of
  finding a path (if one exists) → 1. |

## Phase 03: Physics & RL

| Term | Definition |
|---|---|
| **6-DOF** | Six degrees of freedom: 3 position (x, y, z) + 3
  orientation (roll, pitch, yaw). |
| **Euler angles** | Roll (φ), pitch (θ), yaw (ψ). 3-angle representation
  of orientation. Suffers from gimbal lock at θ = ±90°. |
| **Rotation matrix** | 3×3 matrix R that transforms vectors from body
  frame to world frame. Orthogonal: Rᵀ = R⁻¹. |
| **PID** | Proportional-Integral-Derivative controller. `u(t) = Kp·e +
  Ki·∫e dt + Kd·de/dt`. |
| **Ziegler-Nichols** | Heuristic PID tuning method. Find ultimate gain
  Ku and period Tu, then derive gains from a table. |
| **Cascaded control** | Multiple control loops in series: outer (slow)
  generates setpoints for inner (fast). |
| **Kalman filter** | Optimal recursive estimator for linear Gaussian
  systems. Two steps: predict (time update) and correct (measurement
  update). |
| **Complementary filter** | Simple sensor fusion: high-pass gyro,
  low-pass accelerometer, blended by parameter α. |
| **Domain randomization** | During training, randomly vary simulation
  parameters so the policy generalizes to the real world. |
| **EKF** | Extended Kalman Filter. Linearizes nonlinear dynamics via
  first-order Taylor expansion. |
| **MDP** | Markov Decision Process — the formal framework for RL:
  (S, A, P, R, γ). |
| **Gym environment** | Standard RL interface: `step(action) → (obs, reward, done, info)` and `reset() → obs`. |
| **Reward shaping** | Augmenting a sparse reward with dense feedback
  to accelerate learning. Must satisfy potential-based condition to
  preserve optimal policy. |
| **Potential-based shaping** | `F(s, s') = γ·Φ(s') - Φ(s)`. Guarantees
  the optimal policy is unchanged. |
| **PPO** | Proximal Policy Optimization. Clipped surrogate objective
  `min(r·A, clip(r, 1-ε, 1+ε)·A)` prevents destructive updates. |
| **Actor-Critic** | Architecture with two networks: actor (policy)
  and critic (value function). |
| **GAE** | Generalized Advantage Estimation. λ parameter trades bias
  (λ→0, TD-like) vs variance (λ→1, MC-like). |
| **KL divergence** | `D_KL(P || Q) = ∑ P(i)·log(P(i)/Q(i))`. Measures
  how much distribution Q diverges from P. Used to constrain policy
  updates. |

## Phase 04: Systems & ML

| Term | Definition |
|---|---|
| **ECS** | Entity-Component-System. Data-oriented architecture:
  Entity = ID, Component = data, System = logic. |
| **SoA vs AoS** | Struct of Arrays (columnar storage, cache-friendly)
  vs Array of Structs (row-oriented). |
| **UDP** | User Datagram Protocol. Connectionless, no guaranteed
  delivery, low overhead. |
| **Protobuf** | Protocol Buffers. Binary serialization format by
  Google. Smaller and faster than JSON/XML. |
| **A/B testing** | Deploy two model versions simultaneously to compare
  performance. |
| **Canary deploy** | Roll out a new model to a small subset (e.g., 10%)
  before full rollout. |
| **Offline RL** | Learning a policy from a static dataset without
  interacting with the environment. |
| **Behavioral cloning** | Supervised learning: imitate expert
  demonstrations. A form of imitation learning. |
| **Online learning** | Continuously updating the policy while the agent
  is running, using streaming experience. |
| **Catastrophic forgetting** | Neural network loses previously learned
  skills when trained on new data. |

## Phase 05: Visualization

| Term | Definition |
|---|---|
| **Saliency map** | Gradient of model output w.r.t. input. Shows which
  input dimensions most influence the output. |
| **Value heatmap** | 2D plot of the critic's value estimate across a
  slice of state space. |
| **Action distribution** | The probability distribution over actions
  for a given state — reveals policy confidence. |
| **Failure mode analysis** | Systematic collection and analysis of
  states where the policy performs poorly. |

## Phase 06: Performance & Sim-to-Real

| Term | Definition |
|---|---|
| **Flamegraph** | Stack-sampled profiling visualization. Width =
  time spent in function, color = call stack depth. |
| **Numba** | JIT compiler for Python. `@njit` decorator compiles
  functions to machine code via LLVM. |
| **SIMD** | Single Instruction, Multiple Data. CPU vector instructions
  that operate on multiple data points at once. |
| **Amdahl's Law** | `S_max = 1 / ((1-P) + P/N)`. Maximum speedup from
  parallelization is limited by the serial portion. |
| **Py-spy** | Sampling profiler for Python. No code modification
  needed, low overhead. |
| **Distilled model** | Smaller model trained to mimic a larger teacher.
  Used for edge deployment. |
| **ONNX** | Open Neural Network Exchange. Standard format for ML model
  interoperability. |
| **Sim-to-real gap** | The performance difference between a policy
  tested in simulation vs on real hardware. |
| **System identification** | Estimating real-world parameters (mass,
  inertia, drag) to make the simulation match reality. |
