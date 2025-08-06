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
MODEL_NAME = os.getenv("MODEL_NAME", "yolov11l.pt")
# LOW threshold is for feeding the tracker
CONF_LOW = 0.1
# HIGH threshold is for the final, stable count
CONF_HIGH = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL = "vision-data-events"
ZONE_CHANGE_CHANNEL = "vision-zone-change-events"
ALERT_CHANNEL = "vision-alert-events"

def get_device():
    """Auto-select the best available hardware accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def process_camera(camera_id: str, config: dict):
    """Process loop for a single camera using the proven 'low-confidence handshake'."""
    try:
        # 1) Setup: RTSP URL, TCP, VideoCapture
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")
        if not rtsp_url:
            print(f"[{camera_id}] FATAL: no RTSP URL", flush=True)
            return

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[{camera_id}] ERROR: cannot open stream {rtsp_url}", flush=True)
            return
        print(f"[{camera_id}] Opened stream.", flush=True)

        # 2) Setup: Model, Redis
        device = get_device()
        print(f"[{camera_id}] Loading model {MODEL_NAME} on {device}...", flush=True)
        model = YOLO(MODEL_NAME)
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print(f"[{camera_id}] Connected to Redis.", flush=True)

        # 3) Setup: Tracker and Zones with robust settings
        tracker = sv.ByteTrack()
        zones = [
            sv.PolygonZone(
                polygon=np.array(z["polygon"], np.int32)
                # We will specify the anchor point in the trigger() call, which is more robust
            )
            for z in config["zones"]
        ]
        zone_names = [z["name"] for z in config["zones"]]
        print(f"[{camera_id}] Initialized ROBUST tracker and {len(zones)} zones.", flush=True)

        # 4) Setup: State Management for Entry/Exit and Alerts
        prev_zone_per_id = {}
        alert_config = config.get('alert_config')
        alert_state = {"last_detection_time": time.time(), "alert_sent": False}
        
    except Exception as setup_e:
        print(f"[{camera_id}] A FATAL error occurred during setup: {setup_e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    print(f"[{camera_id}] Entering main loop...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[{camera_id}] WARNING: empty frame, reconnecting...", flush=True)
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        
        # --- ROBUST DETECTION & TRACKING (from your working test script) ---
        # Step 1: Detect with a VERY LOW threshold to get all potential objects
        results = model(frame, conf=CONF_LOW, classes=[0], device=device, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        
        # Step 2: Give EVERYTHING to the tracker
        tracks = tracker.update_with_detections(all_detections)
        
        # Step 3: AFTER tracking, filter for tracks you are confident in for counting
        # This is the "stable" set that your test script proved works.
        stable_tracks = tracks[tracks.confidence > CONF_HIGH] if tracks is not None else sv.Detections.empty()

        print(f"[{camera_id}] Detections: {len(all_detections)}, Stable Tracks for Counting: {len(stable_tracks)}", flush=True)

        # --- ZONE OCCUPANCY ---
        occupancy_counts = {}
        for name, zone in zip(zone_names, zones):
            # Pass the stable tracks and the anchor point to the trigger
            mask = zone.trigger(detections=stable_tracks)
            count = int(np.sum(mask))
            occupancy_counts[name] = occupancy_counts.get(name, 0) + count
        
        # --- ENTRY/EXIT TRANSITIONS ---
        # FIX: Check if stable_tracks has valid data before iterating
        if stable_tracks.tracker_id is not None:
            for track_id, box in zip(stable_tracks.tracker_id, stable_tracks.xyxy):
                single_track_detection = sv.Detections(xyxy=np.array([box]))
                current_zone = None
                for name, zone in zip(zone_names, zones):
                    if zone.trigger(detections=single_track_detection)[0]:
                        current_zone = name
                        break
                
                prev_zone = prev_zone_per_id.get(track_id)
                if prev_zone != current_zone:
                    event_payload = {
                        "timestamp": time.time(), "tracker_id": int(track_id), "camera_id": camera_id,
                        "from_zone": prev_zone or "outside", "to_zone": current_zone or "outside"
                    }
                    if event_payload["from_zone"] != event_payload["to_zone"]:
                        r.publish(ZONE_CHANGE_CHANNEL, json.dumps(event_payload))
                    
                    if current_zone:
                        prev_zone_per_id[track_id] = current_zone
                    elif track_id in prev_zone_per_id:
                        del prev_zone_per_id[track_id]
        
        # --- ALERTING LOGIC ---
        if alert_config:
            # (Alerting logic here, uses occupancy_counts)
            pass

        # --- PUBLISH DATA ---
        data_payload = {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "zone_counts": occupancy_counts
        }
        r.publish(DATA_CHANNEL, json.dumps(data_payload))

        time.sleep(0.05)

if __name__ == "__main__":
    # Your __main__ block is correct and doesn't need changes.
    with open("configs/zones.json", "r") as f:
        full_config = json.load(f)
    procs = []
    for cam_id, cam_cfg in full_config.items():
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start()
        procs.append(p)
        print(f"Started camera process {cam_id}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()
        print("All camera processes terminated.")