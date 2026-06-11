import time
import numpy as np
import threading
import queue
import csv
from datetime import datetime

# CONFIGURATION
LANDFORM_SIZE_KM = 8.0
BOUNDS_METERS = (LANDFORM_SIZE_KM * 1000) / 2
TARGET_FPS = 20
FRAME_TIME = 1.0 / TARGET_FPS

class DataLogger(threading.Thread):
    def __init__(self, filename="flight_log.csv"):
        super().__init__()
        self.filename = filename
        self.data_queue = queue.Queue() # The buffer
        self.running = True
        self.daemon = True # Kills thread if main program crashes

    def add_entry(self, obj_id, pos, dist, timestamp):
        # This is FAST (nanoseconds). It just puts data in memory.
        # Main loop calls this.
        self.data_queue.put((obj_id, pos, dist, timestamp))

    def run(self):
        # This runs in the BACKGROUND. It handles slow disk I/O.
        print(f"--- LOGGER: Writing to {self.filename} ---")
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "ID", "X", "Y", "Z", "Dist_From_Center"])
            
            while self.running or not self.data_queue.empty():
                try:
                    # Wait for data, but verify 'running' every 1 second
                    data = self.data_queue.get(timeout=1.0)
                    
                    # Unpack and Write
                    obj_id, pos, dist, ts = data
                    writer.writerow([
                        ts, obj_id, 
                        f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}", 
                        f"{dist:.4f}"
                    ])
                    self.data_queue.task_done()
                except queue.Empty:
                    continue
        print("--- LOGGER: Finished writing queue ---")

    def stop(self):
        self.running = False

class SpatialMap:
    def __init__(self):
        self.objects = {} 

    def update_object(self, obj_id, x, y, z):
        if not (-BOUNDS_METERS <= x <= BOUNDS_METERS) or \
           not (-BOUNDS_METERS <= y <= BOUNDS_METERS):
            return
        self.objects[obj_id] = np.array([x, y, z], dtype=np.float64)

    def get_object_pos(self, obj_id):
        return self.objects.get(obj_id, None)

def main():
    radar = SpatialMap()
    
    # 1. Start the Background Logger
    logger = DataLogger()
    logger.start()
    
    drone_x, drone_y, drone_z = -4000.0, -4000.0, 150.0
    speed_mps = 100.0 
    
    print(f"--- SYSTEM START: Tracking at {TARGET_FPS} Hz ---")

    try:
        while True:
            start_time = time.perf_counter()

            # --- UPDATE ---
            radar.update_object("drone_1", drone_x, drone_y, drone_z)
            pos = radar.get_object_pos("drone_1")
            dist = np.linalg.norm(pos)

            # --- LOGGING (Now Concurrent) ---
            # We hand off the data to the logger thread and immediately move on.
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            logger.add_entry("drone_1", pos, dist, timestamp)
            
            # Print less frequently to keep terminal clean (every 20th frame)
            # But the CSV gets EVERY frame.
            print(f"PING: {timestamp} | Dist: {dist:.4f}m")

            # --- MOVEMENT ---
            move_step = speed_mps * FRAME_TIME
            drone_x += move_step
            drone_y += move_step

            # --- TIMING ---
            elapsed = time.perf_counter() - start_time
            sleep_time = FRAME_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
        logger.stop()
        logger.join() # Wait for logger to finish writing remaining data
        print("System Shutdown.")

if __name__ == "__main__":
    main()
