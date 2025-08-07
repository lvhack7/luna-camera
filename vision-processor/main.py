import os
import json
import time
import redis
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import supervision as sv
from multiprocessing import Process
from threading import Thread
from queue import Queue, Empty

# --- Configuration Constants ---
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11l.pt")
CONF_LOW = 0.1
CONF_HIGH_TO_CONFIRM = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
FRAMES_TO_CONFIRM = 15
FRAMES_TO_COAST = 90
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL = "vision-data-events"
ZONE_CHANGE_CHANNEL = "vision-zone-change-events"
ALERT_CHANNEL = "vision-alert-events"

# --- THE DEFINITIVE SOLUTION TO LATENCY: A THREADED VIDEO CAPTURE CLASS ---
class VideoCaptureThreaded:
    def __init__(self, src=0, name="VideoCaptureThread"):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        # A queue of maxsize=1 is the key. It automatically discards old frames.
        self.q = Queue(maxsize=1)
        self.is_running = True
        self.thread = Thread(target=self._reader, name=name, daemon=True)
        self.thread.start()

    def _reader(self):
        print(f"[{self.thread.name}] Reader thread started for {self.src}")
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.thread.name}] Stream connection lost. Reconnecting...")
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            
            # If the queue is full, the old frame is automatically dropped when we put the new one.
            if not self.q.empty():
                try: self.q.get_nowait()
                except Empty: pass
            self.q.put(frame)

    def read(self):
        return self.q.get() # This will block until a new frame is available

    def release(self):
        self.is_running = False
        self.thread.join()
        self.cap.release()

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def is_frame_corrupted(frame: np.ndarray, threshold: float = 10.0) -> bool:
    return frame.std() < threshold

def process_camera(camera_id: str, config: dict):
    try:
        # 1) Setup
        rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")
        if not rtsp_url: print(f"[{camera_id}] FATAL: no RTSP URL", flush=True); return
        
        # USE THE THREADED CAPTURE
        cap = VideoCaptureThreaded(rtsp_url, name=f"CamThread-{camera_id}")
        time.sleep(2) # Give the thread time to connect and get the first frame
        print(f"[{camera_id}] Opened threaded stream.", flush=True)

        device = get_device(); model = YOLO(MODEL_NAME)
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0); r.ping()
        print(f"[{camera_id}] Loaded model and connected to Redis.", flush=True)

        tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=FRAMES_TO_COAST + 10, match_thresh=0.8, frame_rate=20)
        resolution_wh = tuple(config.get("resolution_wh", (1920, 1080)))
        zones = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32), frame_resolution_wh=resolution_wh) for z in config["zones"]]
        zone_names = [z["name"] for z in config["zones"]]
        
        # 2) State Management
        tracked_objects = {}; prev_zone_per_id = {}; last_known_occupancy = {name: 0 for name in zone_names}
        
    except Exception as setup_e:
        print(f"[{camera_id}] FATAL error during setup: {setup_e}", flush=True)
        import traceback; traceback.print_exc(); return

    print(f"[{camera_id}] Entering main loop...", flush=True)
    while True:
        frame = cap.read() # This now gets the LATEST frame, no delay
        
        if is_frame_corrupted(frame):
            print(f"[{camera_id}] Corrupted frame! Preserving last count.", flush=True)
            data_payload = {"timestamp": time.time(), "camera_id": camera_id, "zone_counts": last_known_occupancy}
            r.publish(DATA_CHANNEL, json.dumps(data_payload))
            time.sleep(0.5); continue
        
        # --- AI Processing ---
        # (The rest of the stateful "Algorithm of Intent" is identical to the previous correct version)
        results = model(frame, conf=CONF_LOW, classes=[0], device=device, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(all_detections)

        current_frame_track_ids = set(tracks.tracker_id) if tracks.tracker_id is not None else set()
        for obj_id in list(tracked_objects.keys()):
            if obj_id in current_frame_track_ids:
                tracked_objects[obj_id]['misses'] = 0
                if tracked_objects[obj_id]['state'] == 'Candidate':
                    track_confidence = tracks[tracks.tracker_id == obj_id].confidence[0]
                    if track_confidence > CONF_HIGH_TO_CONFIRM: tracked_objects[obj_id]['high_conf_frames'] += 1
                    if tracked_objects[obj_id]['high_conf_frames'] >= FRAMES_TO_CONFIRM: tracked_objects[obj_id]['state'] = 'Confirmed'
            else:
                tracked_objects[obj_id]['misses'] += 1
                if tracked_objects[obj_id]['state'] in ['Confirmed', 'Coasting']:
                    if tracked_objects[obj_id]['misses'] <= FRAMES_TO_COAST: tracked_objects[obj_id]['state'] = 'Coasting'
                    else: tracked_objects[obj_id]['state'] = 'Lost'
        if tracks.tracker_id is not None:
            for i in range(len(tracks)):
                track_id = tracks.tracker_id[i]
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = {'state': 'Candidate', 'misses': 0, 'high_conf_frames': 1 if tracks.confidence[i] > CONF_HIGH_TO_CONFIRM else 0}
        lost_ids = [obj_id for obj_id, data in tracked_objects.items() if data['state'] == 'Lost']
        for obj_id in lost_ids: del tracked_objects[obj_id]
        
        stable_track_ids = {obj_id for obj_id, data in tracked_objects.items() if data['state'] in ['Confirmed', 'Coasting']}
        stable_tracks = sv.Detections.empty()
        if len(stable_track_ids) > 0 and tracks.tracker_id is not None:
            stable_mask = np.isin(tracks.tracker_id, list(stable_track_ids))
            stable_tracks = tracks[stable_mask]

        occupancy_counts = {}
        for name, zone in zip(zone_names, zones):
            mask = zone.trigger(detections=stable_tracks, anchor=sv.Position.CENTER)
            count = int(np.sum(mask))
            occupancy_counts[name] = occupancy_counts.get(name, 0) + count
        
        data_payload = {"timestamp": time.time(), "camera_id": camera_id, "zone_counts": occupancy_counts}
        r.publish(DATA_CHANNEL, json.dumps(data_payload))
        last_known_occupancy = occupancy_counts
        # We no longer need a sleep here as the cap.read() will naturally pace the loop
        
if __name__ == "__main__":
    # This block remains the same
    with open("configs/zones.json", "r") as f: full_config = json.load(f)
    procs = []; [procs.append(p) for cam_id, cam_cfg in full_config.items() if (p := Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True), p.start(), print(f"Started camera process {cam_id}"))]
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down..."); [p.terminate() for p in procs if p.is_alive()]; [p.join() for p in procs]; print("All processes terminated.")