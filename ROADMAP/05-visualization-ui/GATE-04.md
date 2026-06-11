# Gate 04: Visualization, UI & Policy Insight

Complete all of the following before proceeding to Phase 06.

## 1. 3D Rendering

- [ ] `renderer_3d.py` — 3D renderer with ModernGL/OpenGL
- [ ] Ground plane, drone models (box + rotors), obstacle spheres
- [ ] Camera orbit with mouse drag, zoom with scroll
- [ ] Runs at 60fps for N ≤ 100 drones
- [ ] Benchmark recorded:

```
N       | Pygame 2D (fps) | OpenGL 3D (fps)
100     |     ___         |     ___
500     |     ___         |     ___
5000    |     ___         |     ___
```

## 2. Live Dashboard

- [ ] `dashboard.py` — Real-time telemetry dashboard
- [ ] Plots altitude, speed, heading over rolling window
- [ ] Table of current metrics for all drones
- [ ] Stress markers (GOAL, LOW ALT, UNCERTAIN)
- [ ] Keyboard shortcuts for drone selection and detail view

## 3. 🤖 Policy Visualization

- [ ] `policy_viz.py` with all five visualizations:
      - Value function heatmap
      - Action distribution plots
      - Saliency map (input attribution)
      - Failure mode analysis
      - Rollout video

- [ ] Each visualization saved as a PNG/video in `plots/` directory
- [ ] Analysis report written answering:

```
1. Value heatmap findings: _________________
2. Action distribution findings: ____________
3. Saliency findings: _______________________
4. Failure mode findings: ___________________
```

## 4. Understanding Check

1. The 3D renderer replaces pygame 2D. At what N does OpenGL become
   necessary vs pygame being sufficient?

2. The dashboard updates at 10Hz. How would you implement a real-time
   dashboard at 60Hz without matplotlib's animation overhead?

3. The saliency map shows which inputs affect the policy most. If the
   policy ignores obstacle distance entirely, what does this tell you
   about the training data or reward function?

4. The value heatmap shows high values near the goal and low values
   near obstacles. What shape would you expect for a well-trained
   policy vs an undertrained one?

---

**Gate status:** `[PASSED]` / `[NOT YET]`

Once all boxes are checked, proceed to `06-performance-scale/README.md`.
