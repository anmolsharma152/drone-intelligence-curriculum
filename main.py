import time
import numpy as np

# CONFIGURATION
LANDFORM_SIZE_KM = 8.0
BOUNDS_METERS = (LANDFORM_SIZE_KM * 1000) / 2  # +/- 4000 meters
TARGET_FPS = 20
FRAME_TIME = 1.0 / TARGET_FPS  # 0.05 seconds (50ms)

class SpatialMap:
    def __init__(self):
        # 1. THE GEOMETRY
        # We define the center (0,0,0) as the middle of the 8km box.
        # We use float64 for 99.9% precision (15-17 decimal digits of accuracy).
        self.objects = {}  # ID -> Coordinates [x, y, z]

    def update_object(self, obj_id, x, y, z):
        """
        Receives a 'ping' and maps it to our local grid.
        """
        # 2. BOUNDARY CHECKS (The 8x8 Landform)
        # If an object is outside -4000 to +4000, it's off the map.
        if not (-BOUNDS_METERS <= x <= BOUNDS_METERS) or \
           not (-BOUNDS_METERS <= y <= BOUNDS_METERS):
            print(f"WARNING: Object {obj_id} out of bounds!")
            return

        # Store as numpy array for fast math later
        self.objects[obj_id] = np.array([x, y, z], dtype=np.float64)

    def get_object_pos(self, obj_id):
        return self.objects.get(obj_id, None)

def main():
    radar = SpatialMap()
    
    # Simulation: A drone starting at the bottom-left corner (-4000m, -4000m)
    # It flies diagonally at 100 meters/second.
    drone_x, drone_y, drone_z = -4000.0, -4000.0, 150.0 # Starting position
    speed_mps = 100.0 
    
    print(f"--- SYSTEM START: Tracking at {TARGET_FPS} Hz ---")
    print(f"--- Map Bounds: +/- {BOUNDS_METERS} meters ---")

    # 3. THE CONCURRENCY LOOP (The Heartbeat)
    try:
        while True:
            start_time = time.perf_counter()

            # --- A. READ DATA (Simulating the Ping) ---
            # In real life, this would come from a sensor/network socket
            radar.update_object("drone_1", drone_x, drone_y, drone_z)

            # --- B. PROCESS DATA (The Mapping Task) ---
            # Let's calculate distance from the center (0,0,0)
            pos = radar.get_object_pos("drone_1")
            dist_from_center = np.linalg.norm(pos) # High-precision vector math

            # Display output (formatted to show precision)
            # We use .4f to show we are tracking down to the millimeter
            print(f"PING: Drone at [{pos[0]:.4f}, {pos[1]:.4f}] | Dist: {dist_from_center:.4f}m")

            # --- C. SIMULATE MOVEMENT ---
            # Move the drone for the next frame (Speed * Time)
            move_step = speed_mps * FRAME_TIME / np.sqrt(2)
            drone_x += move_step
            drone_y += move_step

            # --- D. WAIT (Maintain 20Hz) ---
            # If we finished processing in 0.01s, we must sleep for 0.04s
            # to keep the rhythm steady.
            elapsed = time.perf_counter() - start_time
            sleep_time = FRAME_TIME - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print("LAGGING! Processing took too long!")

    except KeyboardInterrupt:
        print("\nSystem Shutdown.")

if __name__ == "__main__":
    main()
