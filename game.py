import pygame
import numpy as np
import sys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 1000   # Window size
WORLD_SIZE = 8000.0          # 8km simulation space
SCALE = WIDTH / WORLD_SIZE   # Zoom factor
NUM_PARTICLES = 400
CAPACITY = 4                 # Max objects per quad before splitting

# --- COLORS ---
BLACK = (10, 10, 10)
WHITE = (255, 255, 255)
GREEN = (50, 200, 50)
RED   = (255, 50, 50)
GRAY  = (50, 50, 50)

class Rect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h # Center x,y, half-width/height

    def contains(self, p):
        return (self.x - self.w <= p.x < self.x + self.w) and \
               (self.y - self.h <= p.y < self.y + self.h)

    def intersects(self, other):
        return not (other.x - other.w > self.x + self.w or
                    other.x + other.w < self.x - self.w or
                    other.y - other.h > self.y + self.h or
                    other.y + other.h < self.y - self.h)

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = np.random.uniform(-100, 100) # Speed in m/s
        self.vy = np.random.uniform(-100, 100)
        self.highlight = False

    def move(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off world edges (-4000 to 4000)
        limit = WORLD_SIZE / 2
        if self.x < -limit or self.x > limit: self.vx *= -1
        if self.y < -limit or self.y > limit: self.vy *= -1
        self.highlight = False

class QuadTree:
    def __init__(self, boundary, n):
        self.boundary = boundary
        self.capacity = n
        self.points = []
        self.divided = False

    def insert(self, p):
        if not self.boundary.contains(p): return False
        if len(self.points) < self.capacity:
            self.points.append(p)
            return True
        else:
            if not self.divided: self.subdivide()
            return (self.nw.insert(p) or self.ne.insert(p) or 
                    self.sw.insert(p) or self.se.insert(p))

    def subdivide(self):
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h
        mw, mh = w/2, h/2
        self.nw = QuadTree(Rect(x - mw, y - mh, mw, mh), self.capacity)
        self.ne = QuadTree(Rect(x + mw, y - mh, mw, mh), self.capacity)
        self.sw = QuadTree(Rect(x - mw, y + mh, mw, mh), self.capacity)
        self.se = QuadTree(Rect(x + mw, y + mh, mw, mh), self.capacity)
        self.divided = True

    def query(self, range_rect, found):
        if not self.boundary.intersects(range_rect): return
        for p in self.points:
            if range_rect.contains(p): found.append(p)
        if self.divided:
            self.nw.query(range_rect, found)
            self.ne.query(range_rect, found)
            self.sw.query(range_rect, found)
            self.se.query(range_rect, found)

    def draw(self, screen):
        # Convert Sim Coordinates to Screen Coordinates
        sx = int((self.boundary.x + WORLD_SIZE/2) * SCALE)
        sy = int((self.boundary.y + WORLD_SIZE/2) * SCALE)
        sw = int(self.boundary.w * 2 * SCALE)
        sh = int(self.boundary.h * 2 * SCALE)
        pygame.draw.rect(screen, GRAY, (sx - sw//2, sy - sh//2, sw, sh), 1)
        if self.divided:
            self.nw.draw(screen)
            self.ne.draw(screen)
            self.sw.draw(screen)
            self.se.draw(screen)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("QuadTree Spatial Sim (8km x 8km)")
    clock = pygame.time.Clock()

    particles = [Particle(np.random.uniform(-4000, 4000), 
                          np.random.uniform(-4000, 4000)) 
                 for _ in range(NUM_PARTICLES)]

    # MOUSE RADAR
    radar_range = 500 # 500m search radius

    while True:
        dt = clock.tick(60) / 1000.0 # Delta time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # 1. Update Physics
        for p in particles: p.move(dt)

        # 2. Build QuadTree
        boundary = Rect(0, 0, 4000, 4000)
        qt = QuadTree(boundary, CAPACITY)
        for p in particles: qt.insert(p)

        # 3. Query Mouse Position (Simulate Radar)
        mx, my = pygame.mouse.get_pos()
        # Convert screen mouse to sim coordinates
        sim_mx = (mx / SCALE) - 4000
        sim_my = (my / SCALE) - 4000
        
        found = []
        search_area = Rect(sim_mx, sim_my, radar_range, radar_range)
        qt.query(search_area, found)
        for p in found: p.highlight = True

        # 4. Draw
        screen.fill(BLACK)
        qt.draw(screen) # Draw the grid lines
        
        # Draw Radar Box
        rx = int((sim_mx + 4000) * SCALE)
        ry = int((sim_my + 4000) * SCALE)
        rw = int(radar_range * 2 * SCALE)
        pygame.draw.rect(screen, GREEN, (rx - rw//2, ry - rw//2, rw, rw), 2)

        for p in particles:
            sx = int((p.x + 4000) * SCALE)
            sy = int((p.y + 4000) * SCALE)
            col = RED if p.highlight else WHITE
            pygame.draw.circle(screen, col, (sx, sy), 3)

        pygame.display.flip()

if __name__ == "__main__":
    main()
