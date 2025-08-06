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
# Use robust defaults that can be overridden by environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11l.pt")

# Hysteresis Thresholds for stable counting
# A track is ADDED if confidence > CONF_THRESHOLD_TO_ADD
# A track is MAINTAINED if confidence > CONF_THRESHOLD_TO_TRACK
CONF_THRESHOLD_TO_ADD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
CONF_THRESHOLD_TO_TRACK = 0.15 # Keep this low to maintain tracks even with flickering confidence

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL = "vision-data-events"
ZONE_CHANGE_CHANNEL = "vision-zone-change-events"
ALERT_CHANNEL = "vision-alert-events"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def process_camera(camera_id: str, config: dict):

    try:
        # 1) Setup: RTSP URL, TCP, VideoCapture
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")
        if not rtsp_url:
            print(f"[{camera_id}] FATAL: no RTSP URL in env var '{config.get('rtsp_url_env')}'", flush=True)
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
        print(f"[{camera_id}] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}", flush=True)

        # 3) Setup: Tracker and Zones with robust settings
        tracker = sv.ByteTrack(frame_rate=30)
        zones = [
            sv.PolygonZone(
                polygon=np.array(z["polygon"], np.int32),
            )
            for z in config["zones"]
        ]
        zone_names = [z["name"] for z in config["zones"]]
        print(f"[{camera_id}] Initialized ROBUST tracker and {len(zones)} zones.", flush=True)

        # 4) Setup: State Management for Hysteresis, Entry/Exit, and Alerts
        confirmed_track_ids = set()
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
            print(f"[{camera_id}] WARNING: empty frame, reconnecting in 5s...", flush=True)
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        
        # --- DETECTION & TRACKING ---
        results = model(frame, conf=CONF_THRESHOLD_TO_TRACK, classes=[0], device=device, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(all_detections)

        # --- HYSTERESIS LOGIC: "TRUST THE TRACKER" ---
        
        # Add new, high-confidence tracks to our confirmed set
        for track_id, confidence in zip(tracks.tracker_id, tracks.confidence):
            if track_id not in confirmed_track_ids and confidence > CONF_THRESHOLD_TO_ADD:
                confirmed_track_ids.add(track_id)
        
        # Prune tracks that the tracker has permanently lost
        current_frame_track_ids = set(tracks.tracker_id)
        confirmed_track_ids = confirmed_track_ids.intersection(current_frame_track_ids)
        
        # Create a Detections object containing ONLY our confirmed, active tracks
        confirmed_tracks = sv.Detections.empty()
        if len(confirmed_track_ids) > 0:
            confirmed_mask = np.isin(tracks.tracker_id, list(confirmed_track_ids))
            confirmed_tracks = tracks[confirmed_mask]
        
        print(f"[{camera_id}] Detections: {len(all_detections)}, Confirmed Stable Tracks: {len(confirmed_tracks)}", flush=True)

        # --- ZONE OCCUPANCY (using the hyper-stable confirmed_tracks) ---
        occupancy_counts = {}
        for name, zone in zip(zone_names, zones):
            mask = zone.trigger(detections=confirmed_tracks)
            count = int(np.sum(mask))
            occupancy_counts[name] = occupancy_counts.get(name, 0) + count
        
        # --- ENTRY/EXIT TRANSITIONS (also using stable confirmed_tracks) ---
        for track_id, box in zip(confirmed_tracks.tracker_id, confirmed_tracks.xyxy):
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
            alert_zone_name = alert_config['alert_zone_name']
            if occupancy_counts.get(alert_zone_name, 0) > 0:
                alert_state["last_detection_time"] = time.time()
                alert_state["alert_sent"] = False
            else:
                absence_duration = time.time() - alert_state["last_detection_time"]
                threshold = alert_config.get('absence_threshold_seconds', 30)
                if not alert_state["alert_sent"] and absence_duration > threshold:
                    alert_payload = {
                        "timestamp": time.time(), "camera_id": camera_id, "alert_type": "STAFF_ABSENCE",
                        "message": f"ALERT: No staff detected in '{alert_zone_name}' on {camera_id} for over {threshold} seconds."
                    }
                    r.publish(ALERT_CHANNEL, json.dumps(alert_payload))
                    alert_state["alert_sent"] = True

        # --- PUBLISH DATA ---
        data_payload = {
            "timestamp": time.time(),
            "camera_id": camera_id,
            "zone_counts": occupancy_counts
        }
        r.publish(DATA_CHANNEL, json.dumps(data_payload))

        time.sleep(0.05)

if __name__ == "__main__":
    with open("configs/zones.json", "r") as f:
        full_config = json.load(f)

    procs = []
    for cam_id, cam_cfg in full_config.items():
        # Use daemon=True to ensure child processes are terminated when the main script exits
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start()
        procs.append(p)
        print(f"Started camera process {cam_id}")

    try:
        # Keep the main process alive to manage the child processes
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        for p in procs:
            if p.is_alive():
                p.terminate() # Terminate child processes on exit
                p.join()      # Wait for them to finish
        print("All camera processes terminated.")