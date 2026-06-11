# Gate 02: Physics Simulation & RL Environment

You must complete all of the following before proceeding to Phase 04.

## 1. 6-DOF Physics Engine

- [ ] `rigid_body.py` — 6-DOF kinematics with Euler integration and
      rotation matrix
- [ ] RK4 integration also implemented (bonus)
- [ ] Euler integration error vs dt quantified and plotted

Verify with:
```bash
python test_rigid_body.py  # Should track a hover within 1% altitude
```

## 2. PID Control

- [ ] `pid_controller.py` — Cascaded PID (altitude + velocity + attitude)
- [ ] Drone can fly to a waypoint (100, 50, 5) and hold position
- [ ] Disturbance rejection: wind gust of 5 m/s, recovers within 3 seconds
- [ ] Settling time < 5s, overshoot < 10%

## 3. Sensor Noise & Filtering

- [ ] `sensor_noise.py` — GPS and IMU noise models
- [ ] `kalman_filter.py` — At minimum 1D altitude Kalman filter
- [ ] 12D EKF is implemented (bonus)
- [ ] Benchmark: Kalman filter reduces position RMSE vs raw GPS by
      at least 3x

## 4. Collision Avoidance

- [ ] `collision.py` — Collision detection with spatial acceleration
- [ ] At least one avoidance method implemented (potential field or
      velocity obstacle)
- [ ] Drone successfully navigates through 10+ obstacle course

## 5. Environment

- [ ] `environment.py` — WindField, obstacle course generator (4 styles),
      Airspace multi-drone stepping
- [ ] Spatial-index accelerated collision queries working

## 6. 🤖 RL Environment

- [ ] `drone_env.py` — Gym environment with:
      - 24-dim observation space ✓
      - 4-dim continuous action space ✓
      - step(), reset() implemented ✓
      - Correct termination conditions ✓
- [ ] Environment passes the "gym.make" test
- [ ] 1000 random-action episodes complete without errors

## 7. 🤖 Reward Design

- [ ] At least 3 reward variants implemented (sparse, dense, potential)
- [ ] Random policy benchmark recorded for each variant
- [ ] Physics domain randomization implemented

## 8. 🤖 PPO Training

- [ ] `ppo.py` — Full PPO with:
      - Actor (Gaussian policy) ✓
      - Critic (value function) ✓
      - GAE advantage estimation ✓
      - Clipped surrogate objective ✓
- [ ] `train_ppo.py` runs without errors
- [ ] Training produces a policy that beats random actions
      (mean reward at least 2x random baseline)
- [ ] Policy saved to `drone_policy.pth`

## 9. 🤖 Evaluation

- [ ] PID vs RL comparison table filled out:

```
Metric                | PID     | RL      | Winner
----------------------+---------+---------+-------
Success rate          | ___%    | ___%    | ___
Avg flight time       | ___s    | ___s    | ___
Avg path length       | ___m    | ___m    | ___
Wind rejection        | ___m/s  | ___m/s  | ___
```

## 10. Understanding Check

Answer each without looking at your code:

1. Draw the cascaded PID architecture for drone control. What are the
   three loops and what do they control?

2. Why is the Kalman filter "optimal" for linear Gaussian systems?
   What assumption breaks when you use it with the full 6-DOF dynamics?

3. An RL agent trained in simulation with no sensor noise fails
   catastrophically on real hardware. Name three domain randomization
   parameters that would fix this.

4. The PPO clipped objective prevents the new policy from diverging
   too far from the old policy. Write out the full objective function
   and explain what the clip() operation prevents.

## Performance Target

Your trained policy should achieve, at minimum:

```
Success rate:          > 60% (reach goal without collision)
Mean episode reward:   > 0   (positive = more reward than penalty)
Mean episode length:   > 500 steps (survive > 8 seconds)
```

---

**Gate status:** `[PASSED]` / `[NOT YET]`

Once all boxes are checked, proceed to `04-systems-engineering/README.md`.
