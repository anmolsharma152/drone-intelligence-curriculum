# 3D Rendering

## Learning Objective

Replace 2D pygame circles with 3D rendering using ModernGL. Render
drones as 3D models, obstacles as spheres, and the ground plane.

## ModernGL Setup

```python
import moderngl as mgl
import pygame
import numpy as np

class Renderer3D:
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.display.set_mode((width, height),
                                pygame.OPENGL | pygame.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.width = width
        self.height = height
        self.camera = Camera()

    def clear(self):
        self.ctx.clear(0.2, 0.2, 0.3)  # Dark blue sky

    def render_drone(self, pos, euler, color=(1.0, 0.0, 0.0)):
        # Build model matrix from position + Euler angles
        model = np.eye(4, dtype='f4')
        model[:3, 3] = pos

        # Rotation from Euler angles
        rx = rotation_x(euler[0])
        ry = rotation_y(euler[1])
        rz = rotation_z(euler[2])
        model[:3, :3] = (rz @ ry @ rx)[:3, :3]

        # Camera view-projection
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix()
        mvp = proj @ view @ model

        # Render drone mesh (box + rotor discs)
        # ... shader setup, draw calls

    def render_obstacle(self, center, radius):
        model = np.eye(4, dtype='f4')
        model[:3, 3] = center
        model[:3, :3] *= radius  # Scale to sphere radius
        # Render sphere mesh
```

## Camera Control

```python
class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 200.0])  # Above origin
        self.target = np.zeros(3)
        self.up = np.array([0.0, 1.0, 0.0])

    def view_matrix(self):
        fwd = self.target - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, self.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        view = np.eye(4, dtype='f4')
        view[:3, :3] = np.vstack([right, up, -fwd])
        view[:3, 3] = -view[:3, :3] @ self.pos
        return view

    def projection_matrix(self, fov=60, near=1.0, far=10000.0):
        aspect = 1280 / 720
        t = near * np.tan(np.radians(fov/2))
        r = t * aspect
        proj = np.zeros((4, 4), dtype='f4')
        proj[0,0] = near / r
        proj[1,1] = near / t
        proj[2,2] = -(far + near) / (far - near)
        proj[2,3] = -2 * far * near / (far - near)
        proj[3,2] = -1
        return proj

    def orbit(self, dx, dy):
        """Orbit camera around target."""
        # Implement spherical coordinate orbit
        pass

    def zoom(self, delta):
        """Zoom in/out."""
        self.pos += (self.target - self.pos) * delta
```

## Implementation Task

### 1. Build `renderer_3d.py`

Implement the 3D renderer with:
- Ground plane (grid)
- Drone model (box body + 4 rotor discs)
- Obstacle spheres (with color coding: red = collision, green = safe)
- Goal marker (flashing sphere)
- Camera orbit (mouse drag)
- Camera zoom (scroll)

### 2. Integrate with Simulation

Replace the pygame 2D render loop:

```python
class SimWith3D:
    def __init__(self):
        self.renderer = Renderer3D()
        self.world = Airspace()
        self.drones = []

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.handle_input()
            self.world.step(1/60)
            self.renderer.clear()

            for drone in self.drones:
                self.renderer.render_drone(
                    drone.state.pos, drone.state.euler)

            for obs in self.world.obstacle_field:
                self.renderer.render_obstacle(
                    obs.center, obs.radius)

            self.renderer.render_goal(self.world.goal)
            pygame.display.flip()
            clock.tick(60)
```

### 3. Benchmark

```
Format  | N=100  | N=1000  | N=10000
--------+--------+---------+---------
pygame  | 60fps  |  ___    |  ___
OpenGL  | 60fps  |  ___    |  ___
```

## Check-In Questions

1. The camera projection matrix uses a near plane at 1.0 and far at
   10000. What rendering artifacts appear if near is too small? Too large?

2. The renderer draws each drone individually with a draw call per
   drone. For 10,000 drones, this is 10,000 draw calls — too many.
   How would you batch render all drones with a single draw call?

3. The drone mesh is a simple box. How many vertices does a box have?
   How many for a detailed drone model? What's the performance impact?

## 🤖 Connection

3D visualization is critical for RL debugging:
- Render the policy's value function as a heatmap on the ground plane
- Show the observation raycasts (what the agent "sees")
- Color-code drones by prediction confidence

**Next:** `02-live-dashboard-telemetry.md`
