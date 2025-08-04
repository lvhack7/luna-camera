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

# --- Configuration Constants ---
MODEL_NAME       = os.getenv("MODEL_NAME", "yolo11m.pt")
CONF_THRESHOLD   = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))
REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL     = "vision-data-events"
ZONE_CHANGE_CHANNEL = "vision-zone-change-events"
ALERT_CHANNEL    = "vision-alert-events"

def get_device():
    """Auto-select the best available hardware accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def process_camera(camera_id: str, config: dict):
    """Process loop for a single camera: detection, tracking, zone counting, and Redis publish."""
    # 1) Resolve RTSP URL and set TCP option
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    rtsp_url = config.get("rtsp_url") or os.getenv(config.get("rtsp_url_env", ""), "")
    if not rtsp_url:
        print(f"[{camera_id}] FATAL: no RTSP URL in config or env.", flush=True)
        return

    # 2) Open VideoCapture
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[{camera_id}] ERROR: cannot open stream {rtsp_url}", flush=True)
        return
    print(f"[{camera_id}] Opened stream.", flush=True)

    # 3) Load model & connect Redis
    device = get_device()
    print(f"[{camera_id}] Loading model {MODEL_NAME} on {device}...", flush=True)
    model = YOLO(MODEL_NAME)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[{camera_id}] ERROR: cannot connect to Redis – {e}", flush=True)
        return
    print(f"[{camera_id}] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}", flush=True)

    # 4) Initialize tracker and zones
    tracker = sv.ByteTrack(frame_rate=20)
    zones = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32)) for z in config["zones"]]
    zone_names = [z["name"] for z in config["zones"]]
    print(f"[{camera_id}] Initialized tracker and {len(zones)} zones.", flush=True)

    # 5) State for entry/exit events
    prev_zone_per_id = {}
    
    print(f"[{camera_id}] Entering main loop...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[{camera_id}] WARNING: empty frame, reconnecting in 5s...", flush=True)
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        
        # --- Detection & Tracking ---
        results = model(frame, conf=CONF_THRESHOLD, classes=[0], device=device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(detections)
        print(f"[{camera_id}] Detected {len(detections)} raw, Tracking {len(tracks)} people.", flush=True)

        # --- Zone Occupancy (for real-time dashboard) ---
        occupancy_counts = {}
        for name, zone in zip(zone_names, zones):
            mask = zone.trigger(detections=tracks)
            count = int(np.sum(mask))
            occupancy_counts[name] = occupancy_counts.get(name, 0) + count
            if count > 0:
                print(f"[{camera_id}]  → {count} in zone '{name}'", flush=True)

        # --- Entry/Exit Transitions (for historical reporting) ---
        for track_id, box in zip(tracks.tracker_id, tracks.xyxy):
            # FIX 1: Use the correct method to check zone containment
            # Create a temporary Detections object for the single track
            single_track_detection = sv.Detections(xyxy=np.array([box]))

            current_zone = None
            for name, zone in zip(zone_names, zones):
                # The .trigger() method is the modern way to check for intersection
                if zone.trigger(detections=single_track_detection)[0]:
                    current_zone = name
                    break # Assign the first zone found

            prev_zone = prev_zone_per_id.get(track_id)
            
            # If the zone has changed, fire an event
            if prev_zone != current_zone:
                event_payload = {
                    "timestamp": time.time(), "tracker_id": int(track_id), "camera_id": camera_id,
                    "from_zone": prev_zone or "outside",
                    "to_zone": current_zone or "outside"
                }
                if event_payload["from_zone"] != event_payload["to_zone"]:
                    r.publish(ZONE_CHANGE_CHANNEL, json.dumps(event_payload))
                    print(f"[{camera_id}] EVENT: Track {track_id} moved from {event_payload['from_zone']} to {event_payload['to_zone']}", flush=True)
                
                # Update the state for the next frame
                if current_zone:
                    prev_zone_per_id[track_id] = current_zone
                elif track_id in prev_zone_per_id:
                    # The track has left all zones, remove it from state
                    del prev_zone_per_id[track_id]
        
        # --- Publish Instantaneous Occupancy Data ---
        # FIX 2: Rename 'occupancy' key to 'zone_counts' to match API server
        data_payload = {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "zone_counts": occupancy_counts 
        }
        r.publish(DATA_CHANNEL, json.dumps(data_payload))

        time.sleep(0.05)


if __name__ == "__main__":
    # Load per-camera configuration
    with open("configs/zones.json", "r") as f:
        full_config = json.load(f)

    procs = []
    for cam_id, cam_cfg in full_config.items():
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start()
        procs.append(p)
        print(f"Started camera process {cam_id}")

    # Keep main alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        for p in procs:
            p.terminate()
        print("All camera processes terminated.")