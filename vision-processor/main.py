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
    # 1) Resolve RTSP URL
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
    tracker = sv.ByteTrack()  # stable ID tracking
    zones      = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32))
                  for z in config["zones"]]
    zone_names = [z["name"] for z in config["zones"]]
    print(f"[{camera_id}] Initialized tracker and {len(zones)} zones.", flush=True)

    # 5) State for entry/exit events
    prev_zone_per_id = {}
    entry_count = 0
    exit_count  = 0

    print(f"[{camera_id}] Entering main loop...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[{camera_id}] WARNING: empty frame, reconnecting in 5s...", flush=True)
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        
        # --- Detection ---
        results    = model(frame, conf=CONF_THRESHOLD, classes=[0], device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        print(f"[{camera_id}] Detected {len(detections)} raw people.", flush=True)

        # --- Tracking ---
        tracks = tracker.update_with_detections(detections)
        print(f"[{camera_id}] Tracking {len(tracks)} people.", flush=True)

        # --- Zone Occupancy & Events ---
        occupancy = {}
        for name, zone in zip(zone_names, zones):
            mask = zone.trigger(detections=tracks)
            cnt  = int(np.sum(mask))
            occupancy[name] = cnt
            if cnt:
                print(f"[{camera_id}]  → {cnt} in zone '{name}'", flush=True)

        # Entry/exit transitions based on 'inside' zone
        for tid, box in zip(tracks.tracker_id, tracks.xyxy):
            # compute centroid
            x1, y1, x2, y2 = box
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            # detect current zone
            current = None
            for name, zone in zip(zone_names, zones):
                if zone.contains_point((cx, cy)):
                    current = name
                    break

            prev = prev_zone_per_id.get(tid)
            if prev != current:
                # example: count entry if coming from outside into 'inside_zone'
                if prev is None and current == "inside_zone":
                    entry_count += 1
                    print(f"[{camera_id}] ENTRY #{entry_count} by track {tid}", flush=True)
                # count exit if leaving 'inside_zone'
                if prev == "inside_zone" and current is None:
                    exit_count += 1
                    print(f"[{camera_id}] EXIT  #{exit_count} by track {tid}", flush=True)

                prev_zone_per_id[tid] = current

        # --- Publish counts ---
        data_payload = {
            "camera_id": camera_id,
            "timestamp": time.time(),
            "occupancy": occupancy,
            "entries": entry_count,
            "exits": exit_count
        }
        r.publish(DATA_CHANNEL, json.dumps(data_payload))

        # short sleep to match roughly camera FPS
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