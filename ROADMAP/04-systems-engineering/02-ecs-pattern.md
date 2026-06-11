# Entity-Component-System Architecture

## Learning Objective

Refactor the monolithic drone simulation into an Entity-Component-System
(ECS) architecture for better performance and extensibility.

## Why ECS?

Your current simulation has a monolithic `Drone` class that does
everything: physics, rendering, AI, networking. This works for 400
entities but breaks at 10,000.

ECS solves this by separating:

- **Entity:** Just an ID (integer)
- **Component:** Data only (position, velocity, health, etc.)
- **System:** Logic that processes entities with specific components

```
Monolithic:           ECS:
Drone class          Entity { id: int }
  ├─ pos              ├─ Position { x, y, z }
  ├─ vel              ├─ Velocity { vx, vy, vz }
  ├─ update()         ├─ Health { hp }
  ├─ render()         └─ ...
  ├─ collide()
  └─ think()

                     PhysicsSystem:        [queries Position + Velocity]
                     RenderSystem:         [queries Position]
                     CollisionSystem:      [queries Position + Collider]
                     AISystem:             [queries AIState]
```

## Implementation

### 1. Build `ecs.py`

```python
from collections import defaultdict
import numpy as np

class Component:
    """Base component — data only."""
    pass

class Position(Component):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z
        self.__slots__ = ('x', 'y', 'z')  # Memory optimization

class Velocity(Component):
    def __init__(self, vx=0.0, vy=0.0, vz=0.0):
        self.vx, self.vy, self.vz = vx, vy, vz

class Physics(Component):
    """6-DOF state."""
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.omega = np.zeros(3)

class Renderable(Component):
    def __init__(self, color='white', size=1.0):
        self.color = color
        self.size = size

class AIState(Component):
    def __init__(self, goal=None, mode='idle'):
        self.goal = goal or np.zeros(3)
        self.mode = mode  # 'idle', 'navigate', 'avoid', 'manual'

class World:
    """Simple ECS world."""
    def __init__(self):
        self.entities = set()
        self.components = defaultdict(dict)  # type -> entity_id -> component

    def create_entity(self):
        eid = max(self.entities) + 1 if self.entities else 0
        self.entities.add(eid)
        return eid

    def add_component(self, entity_id, component):
        self.components[type(component)][entity_id] = component

    def remove_entity(self, entity_id):
        self.entities.discard(entity_id)
        for comp_dict in self.components.values():
            comp_dict.pop(entity_id, None)

    def query(self, *component_types):
        """Get all entities that have ALL specified component types."""
        if not component_types:
            return list(self.entities)
        result = set(self.components[component_types[0]].keys())
        for ct in component_types[1:]:
            result &= set(self.components[ct].keys())
        return list(result)

    def get(self, entity_id, component_type):
        return self.components[component_type].get(entity_id)
```

### 2. Build Systems

```python
class PhysicsSystem:
    def update(self, world, dt):
        for eid in world.query(Physics):
            phys = world.get(eid, Physics)
            # Apply physics (from rigid_body.py)
            forces = self.compute_forces(world, eid, phys)
            # Euler step
            phys.pos += phys.vel * dt
            phys.vel += forces / self.mass * dt

    def compute_forces(self, world, eid, phys):
        # Wind, thrust, drag, gravity
        return np.array([0, 0, -9.81])

class RenderSystem:
    def update(self, world, surface):
        for eid in world.query(Position, Renderable):
            pos = world.get(eid, Position)
            ren = world.get(eid, Renderable)
            # Draw on surface
            pygame.draw.circle(surface, ren.color,
                             (int(pos.x), int(pos.y)), ren.size)

class CollisionSystem:
    def __init__(self):
        self.spatial = QuadTree(Rect(0, 0, 4000, 4000), 4)

    def update(self, world):
        self.spatial.clear()
        for eid in world.query(Position):
            pos = world.get(eid, Position)
            self.spatial.insert(eid, pos.x, pos.y)

        for eid in world.query(Position, Collider):
            pos = world.get(eid, Position)
            collider = world.get(eid, Collider)
            nearby = self.spatial.query(Rect(pos.x, pos.y,
                                           collider.radius, collider.radius))
            for other in nearby:
                if other != eid:
                    self.resolve_collision(world, eid, other)
```

## Implementation Task

### 1. Refactor `game.py` to Use ECS

Migrate the existing drone simulation:

1. Define components: Position, Velocity, Renderable, AIState, Collider
2. Define systems: PhysicsSystem, RenderSystem, CollisionSystem, AISystem
3. Create entities with appropriate components
4. Run the simulation loop

### 2. Measure Performance

```
N Entities | Monolithic (ms) | ECS (ms) | Speedup
-----------+-----------------+----------+---------
   100     |      ___        |   ___    |   ___
   1000    |      ___        |   ___    |   ___
   10000   |      ___        |   ___    |   ___
```

## Check-In Questions

1. ECS separates data (components) from logic (systems). What cache
   locality benefit does this provide on modern CPUs?

2. In this simple ECS, querying for entities with specific components
   is O(# components × # entities). How would you optimize this to
   O(# entities) per query?

3. The physics system updates position using `pos += vel * dt`. If
   the render system reads position in the same frame, can it see
   a partially-updated position? How do you handle this?

## 🤖 ML Connection

ECS makes ML integration clean:

- **Observation extraction:** An `ObservationSystem` queries entities
  with AIState + Physics, builds observation tensors in batch
- **Action application:** An `ActionSystem` reads a policy output
  tensor and applies it to all entities simultaneously
- **Batch inference:** Process 1000 drone observations through the
  neural network in one forward pass (GPU utilization!)

**Next:** `03-udp-command-protocol.md`
