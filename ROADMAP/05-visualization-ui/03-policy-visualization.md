# 🤖 Policy Visualization

## Learning Objective

Visualize what the neural network policy is "thinking" — attention
maps, value heatmaps, action distributions, and failure mode analysis.

## Why Policy Visualization?

Neural network policies are black boxes. Visualization answers:

- **Where is the policy looking?** Which observation dimensions matter?
- **What would it do?** Action distribution for a given state.
- **How confident is it?** Value estimate vs action entropy.
- **Where does it fail?** Regions of state space with high error.

## 1. Value Function Heatmap

Plot the critic's value estimate across the state space:

```python
def value_heatmap(agent, x_range=(-500, 500), y_range=(-500, 500),
                  goal=np.array([0, 0, 10]), resolution=50):
    """Plot value function over 2D slice of state space."""
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    values = np.zeros((resolution, resolution))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Construct observation for this position
            obs = np.zeros(24)
            obs[0:3] = goal - np.array([x, y, 10])  # Relative pos
            obs[6:9] = [0, 0, 0]  # Level attitude
            # ... other obs dimensions zeroed

            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                value = agent.critic(obs_t).item()
            values[i, j] = value

    plt.figure(figsize=(8, 6))
    plt.imshow(values.T, origin='lower',
               extent=[*x_range, *y_range],
               cmap='RdYlBu', vmin=-10, vmax=100)
    plt.colorbar(label='Predicted Return')
    plt.scatter(*goal[:2], color='green', s=100, marker='*',
                label='Goal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Value Function Heatmap')
    plt.legend()
    plt.savefig('plots/value_heatmap.png')
```

**Interpretation:**
- High values (blue) = states the policy expects good outcomes from
- Low values (red) = states the policy expects failure
- Sharp transitions = decision boundaries the policy learned

## 2. Action Distribution Visualization

For a given state, show the action distribution:

```python
def action_distribution(agent, obs):
    """Plot action distribution for a single observation."""
    obs_t = torch.FloatTensor(obs)
    mean, std = agent.actor(obs_t)

    action_names = ['Thrust', 'Roll', 'Pitch', 'Yaw']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        # Create normal distribution
        x = np.linspace(mean[i].item() - 3*std[i].item(),
                        mean[i].item() + 3*std[i].item(), 100)
        y = 1/(std[i].item() * np.sqrt(2*np.pi)) * \
            np.exp(-0.5*((x - mean[i].item())/std[i].item())**2)

        ax.plot(x, y, 'b-', lw=2)
        ax.axvline(mean[i].item(), color='r', linestyle='--',
                   label=f'mean={mean[i].item():.2f}')
        ax.fill_between(x, y, alpha=0.2)
        ax.set_title(action_names[i])
        ax.set_xlabel('Action value')
        ax.set_ylabel('Probability density')
        ax.legend()

    plt.tight_layout()
    plt.savefig('plots/action_dist.png')
```

**Interpretation:**
- Narrow peak = high confidence in this action
- Wide distribution = uncertain, exploring
- Multiple modes = policy sees multiple viable options

## 3. Saliency Map (Input Attribution)

Which observation dimensions most influence the policy?

```python
def saliency_map(agent, obs):
    """Compute gradient of action w.r.t. observation."""
    obs_t = torch.FloatTensor(obs).requires_grad_(True)
    mean, _ = agent.actor(obs_t)

    # Sum all action dimensions
    agent.actor.zero_grad()
    mean.sum().backward()

    saliency = obs_t.grad.abs().numpy()

    obs_names = [
        'dx', 'dy', 'dz', 'vx', 'vy', 'vz',
        'roll', 'pitch', 'yaw', 'p', 'q', 'r',
        'wx', 'wy', 'wz',
        'obs1_x', 'obs1_y', 'obs1_d',
        'obs2_x', 'obs2_y', 'obs2_d',
        'obs3_x', 'obs3_y', 'obs3_d',
    ]

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(saliency)), saliency)
    plt.xticks(range(len(saliency)), obs_names, rotation=45, ha='right')
    plt.ylabel('Gradient magnitude')
    plt.title('Which inputs most affect the policy output?')
    plt.tight_layout()
    plt.savefig('plots/saliency.png')
```

**Interpretation:**
- Tall bars = observation dimensions the policy heavily weights
- Short bars = dimensions the policy mostly ignores
- Unexpected tall/short bars = debugging insight

## 4. Failure Mode Analysis

Cluster states where the policy performs poorly:

```python
def failure_analysis(agent, env, n_episodes=100):
    """Collect states where the policy fails and analyze them."""
    failures = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _, value = agent.get_action(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)

            # Record low-value states that led to failure
            if value < -5.0:  # Low confidence
                failures.append({
                    'obs': obs,
                    'value': value,
                    'next_obs': next_obs,
                    'reward': reward,
                    'info': info,
                })
            obs = next_obs

    # Analyze common patterns in failure states
    failure_obs = np.array([f['obs'] for f in failures])
    mean_failure_obs = failure_obs.mean(axis=0)
    print("Average failure observation:")
    for name, val in zip(obs_names, mean_failure_obs):
        print(f"  {name}: {val:.2f}")

    return failures
```

## 5. Rollout Video

Generate a video of the policy flying:

```python
def record_rollout(agent, env, filename='rollout.mp4', max_steps=2000):
    """Record a video of the policy flying."""
    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    obs = env.reset()
    done = False
    step = 0
    positions = []

    with writer.saving(fig, filename, 300):
        while not done and step < max_steps:
            action, _, value = agent.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            positions.append(info['pos'])

            # 2D trajectory view
            ax1.clear()
            traj = np.array(positions)
            ax1.plot(traj[:, 0], traj[:, 1], 'b-')
            ax1.scatter(env.goal[0], env.goal[1], color='g', s=100, marker='*')
            ax1.set_xlim(-1000, 1000)
            ax1.set_ylim(-1000, 1000)
            ax1.set_title(f'Step {step} | Value={value:.2f}')

            # Altitude profile
            ax2.clear()
            ax2.plot(traj[:, 2])
            ax2.axhline(env.goal[2], color='g', linestyle='--')
            ax2.set_ylabel('Altitude (m)')
            ax2.set_xlabel('Step')
            ax2.set_title('Altitude Profile')

            writer.grab_frame()
            step += 1
```

## Implementation Task

### 1. Build `policy_viz.py`

Implement all five visualization types:
- Value function heatmap (over 2D slice of state space)
- Action distribution plots
- Saliency map
- Failure mode analysis
- Rollout video recorder

### 2. Analyze Your Trained Policy

Run each visualization and answer:

1. **Value heatmap:** Where does the policy think it will succeed?
   Does it match reality?

2. **Action distributions:** Is the policy confident or uncertain in
   typical flight states? Which action dimensions have highest entropy?

3. **Saliency:** Which observation dimensions does the policy use most?
   Is it ignoring something important (e.g., obstacles)?

4. **Failure analysis:** What patterns precede a crash? Can you fix
   them with more training data, reward shaping, or architecture change?

## Check-In Questions

1. The value heatmap slices state space at constant velocity and
   attitude. How would the heatmap look different if we plotted a
   slice at high speed (20 m/s) vs hover?

2. The saliency map uses gradient magnitude. A dimension with small
   gradient might mean (a) the policy ignores it, or (b) the policy
   operates in a flat region of that dimension. How would you
   distinguish these?

3. Failure mode analysis collects states where value < -5.0. How do
   you choose this threshold? What if the optimal threshold depends
   on the state context?

## Connection Back

Visualization closes the loop:

```
Train → Evaluate → Visualize → Identify failure → Fix reward/architecture → Train again
```

Without visualization, RL training is a blind search. With it, you
can see what's wrong and fix it systematically.

**Next:** `GATE-04.md`
