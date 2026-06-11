# 🤖 Feature Engineering Pipeline

## Learning Objective

Build a feature engineering pipeline that transforms raw telemetry
into structured features for ML models. Understand feature selection,
normalization, and streaming computation.

## The Problem

Raw telemetry is high-dimensional, noisy, and contains redundant
information. ML models (especially neural networks) learn better
from well-structured features.

### Raw Telemetry Dimensions

```
Raw (24 dims, from Phase 03 env):
  pos (3), vel (3), euler (3), omega (3), wind (3),
  nearest obstacles (3 × 3)

Desired features:
  More dimensions with derived quantities, or fewer?
```

### Feature Engineering Approaches

```python
class FeatureEngine:
    """Transforms raw observations into ML features."""

    def __init__(self):
        self.history = deque(maxlen=10)  # Last 10 frames
        self.normalizers = {}  # Feature name -> (mean, std)

    def compute_features(self, raw_obs, dt):
        pos = raw_obs[0:3]
        vel = raw_obs[3:6]
        euler = raw_obs[6:9]
        omega = raw_obs[9:12]
        wind = raw_obs[12:15]
        obs = raw_obs[15:24]

        features = {}

        # 1. Raw values (already there)
        features['pos'] = pos
        features['vel'] = vel
        features['euler'] = euler

        # 2. Derived quantities
        features['speed'] = np.linalg.norm(vel)
        features['speed_xy'] = np.linalg.norm(vel[:2])
        features['altitude'] = pos[2]
        features['angular_speed'] = np.linalg.norm(omega)
        features['climb_rate'] = vel[2]
        features['heading'] = euler[2]
        features['wind_strength'] = np.linalg.norm(wind)

        # 3. Temporal features (history)
        if len(self.history) >= 2:
            prev = self.history[-1]
            features['accel'] = (vel - prev['vel']) / dt
            features['jerk'] = (features['accel'] - prev.get('accel', 0)) / dt

        # 4. Relative goal features
        to_goal = raw_obs[0:3]  # Already relative in current env design
        features['dist_to_goal'] = np.linalg.norm(to_goal)
        features['angle_to_goal'] = np.arctan2(to_goal[1], to_goal[0])

        # 5. Obstacle features (already in obs)
        features['min_obs_dist'] = min(
            np.linalg.norm(obs[i*3:(i+1)*3]) for i in range(3))

        # 6. Energy metrics
        features['kinetic_energy'] = 0.5 * np.sum(vel**2)

        self.history.append(features)
        return features

    def feature_vector(self, raw_obs, dt):
        features = self.compute_features(raw_obs, dt)
        # Flatten to ordered vector
        keys = [
            'pos', 'vel', 'euler', 'omega',
            'speed', 'speed_xy', 'altitude', 'climb_rate', 'heading',
            'dist_to_goal', 'angle_to_goal',
            'min_obs_dist', 'kinetic_energy',
        ]
        vec = np.concatenate([np.atleast_1d(features[k]) for k in keys])
        return vec.astype(np.float32)
```

## Normalization

ML models train best on inputs with zero mean and unit variance:

```python
class RunningNormalizer:
    """Online computation of mean and std (Welford's algorithm)."""
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)
```

## Feature Selection

Not all features are useful. Use correlation analysis:

```python
def select_features(log_file):
    """Load logged data and compute feature importance."""
    import pandas as pd

    # Load features from binary telemetry log
    df = load_log_to_dataframe(log_file)

    # Compute correlation with reward
    correlations = df.corr()['reward'].abs().sort_values(ascending=False)
    print("Top correlated features with reward:")
    print(correlations.head(10))

    # Remove highly correlated features (redundancy)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns
               if any(upper[col] > 0.95)]
    print(f"Dropping highly correlated: {to_drop}")

    return df.drop(columns=to_drop)
```

## Streaming Pipeline

For online inference, features must be computed incrementally:

```python
class StreamingFeaturePipeline:
    def __init__(self, feature_engine, normalizer):
        self.engine = feature_engine
        self.normalizer = normalizer
        self.buffer = deque(maxlen=100)  # For batching

    def process(self, raw_obs, dt):
        vec = self.engine.feature_vector(raw_obs, dt)
        normalized = self.normalizer.normalize(vec)
        self.buffer.append(normalized)
        return normalized

    def get_batch(self, batch_size=32):
        """Get batch of recent features for training."""
        if len(self.buffer) < batch_size:
            return None
        return np.array(list(self.buffer)[-batch_size:])
```

## Implementation Task

### 1. Build `feature_pipeline.py`

Implement the full pipeline:
- FeatureEngine with at least 15 derived features
- RunningNormalizer for online normalization
- StreamingFeaturePipeline for inference
- Correlation-based feature selection

### 2. Collect Telemetry Dataset

1. Run the trained PPO policy for 100 episodes
2. Log all raw observations + actions + rewards
3. Process through the feature pipeline
4. Save as a structured dataset (numpy `.npz` or parquet)

### 3. Feature Analysis

```python
# analyze_features.py
pipeline = FeaturePipeline()
features = []

# Load telemetry
for episode in telemetry_reader:
    for step in episode:
        vec = pipeline.process(step.obs, step.dt)
        features.append(vec)
        if step.done:
            pipeline.reset()

features = np.array(features)
print(f"Feature matrix: {features.shape}")
print(f"Mean: {features.mean(axis=0)}")  # Should be ~0 if normalized
print(f"Std: {features.std(axis=0)}")    # Should be ~1 if normalized
```

## Check-In Questions

1. The feature pipeline computes acceleration from successive velocity
   readings. Why is this better than relying on the IMU accelerometer
   reading directly?

2. RunningNormalizer uses Welford's algorithm. Why is this better than
   computing mean and std in batch?

3. If two features have correlation > 0.95, dropping one is recommended.
   Why? What problem does multicollinearity cause for neural networks?

4. The feature vector has approximately 20 dimensions after engineering
   vs 24 raw dimensions. Are we adding or removing information? What's
   the purpose?

## Connection Back

The feature pipeline sits between raw sensor data and the ML model:

```
Raw telemetry → FeatureEngine → Normalizer → ML model
     ↑                              ↑
  UDP stream              Learned from data
```

In Phase 03, the RL agent consumed raw observations directly. In
production, the feature pipeline adds robustness (normalization) and
interpretability (derived quantities).

**Next:** `06-model-serving-infrastructure.md`
