# References & Further Reading

Resources referenced or implied by the curriculum, organized by topic.

## Algorithms & Data Structures

| Resource | Why |
|---|---|
| *Introduction to Algorithms* (CLRS, 4th ed.) | Big-O proofs, spatial data structures, graph search |
| Samet, *Foundations of Multidimensional and Metric Data Structures* | The definitive reference on QuadTrees, KD-Trees, and spatial indexing |
| *Planning Algorithms* (LaValle, 2006) | RRT, RRT*, motion planning theory |
| Hart, Nilsson, Raphael (1968), "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" | The original A* paper |
| [Red Blob Games: A* Introduction](https://www.redblobgames.com/pathfinding/a-star/introduction.html) | Interactive visual explanation of A* |

## Physics & Control

| Resource | Why |
|---|---|
| Beard & McLain, *Small Unmanned Aircraft: Theory and Practice* | Practical drone dynamics, PID control, sensor fusion |
| Stevens, Lewis, Johnson, *Aircraft Control and Simulation* | 6-DOF equations, rotation matrices, numerical integration |
| Welch & Bishop, "An Introduction to the Kalman Filter" (UNC-CH, 2006) | The standard Kalman filter tutorial |
| Ziegler & Nichols (1942), "Optimum Settings for Automatic Controllers" | Original PID tuning paper |

## Reinforcement Learning

| Resource | Why |
|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) | The RL bible. All concepts from first principles |
| Schulman et al. (2017), "Proximal Policy Optimization Algorithms" | The PPO paper — clipped surrogate objective |
| Schulman et al. (2016), "High-Dimensional Continuous Control Using Generalized Advantage Estimation" | GAE derivation |
| Ng, Harada, Russell (1999), "Policy Invariance Under Reward Transformations: Potential-Based Reward Shaping" | Why potential-based shaping preserves optimality |
| Levine et al. (2020), "Offline Reinforcement Learning: Tutorial, Review, and Perspectives" | Offline RL methods for learning from logged data |

## Sim-to-Real & Robotics

| Resource | Why |
|---|---|
| Tobin et al. (2017), "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" | The original domain randomization paper |
| OpenAI et al. (2018), "Learning Dexterous In-Hand Manipulation" | Sim-to-real at scale with domain randomization |
| Tan et al. (2018), "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots" | Real-world deployment of learned policies |
| [MuJoCo](https://mujoco.org/) / [PyBullet](https://pybullet.org/) | Alternative physics engines for sim-to-real |

## Systems Engineering

| Resource | Why |
|---|---|
| Gamma et al., *Design Patterns: Elements of Reusable Object-Oriented Software* | ECS is derived from patterns here |
| [Unity ECS Documentation](https://docs.unity3d.com/Manual/EntityComponentSystem.html) | Practical ECS in a game engine |
| Postel (1980), "User Datagram Protocol" (RFC 768) | The UDP spec — 3 pages, worth reading |

## Mathematics

| Resource | Why |
|---|---|
| Strang, *Linear Algebra and Learning from Data* | Rotation matrices, eigendecomposition, SVD |
| Boyd & Vandenberghe, *Convex Optimization* | Lagrangian duality, convex analysis for RL theory |
| Press et al., *Numerical Recipes* | RK4, numerical integration, random number generation |

## Tools

| Resource | Why |
|---|---|
| [Numba Documentation](https://numba.pydata.org/) | JIT compilation, @njit, prange |
| [ModernGL Documentation](https://moderngl.readthedocs.io/) | Python OpenGL wrapper used in Phase 05 |
| [ONNX Runtime](https://onnxruntime.ai/) | Cross-platform model inference |
| [PyTorch Documentation](https://pytorch.org/docs/) | Neural network building blocks |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/) | Distributed RL framework (for scaling beyond this curriculum) |
