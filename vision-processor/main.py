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

# --- Configuration Constants (Mirrors your test script) ---
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11x.pt") # Defaulting to a standard model
CONF_LOW = 0.1
CONF_HIGH = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL = "vision-data-events"
ZONE_CHANGE_CHANNEL = "vision-zone-change-events"
ALERT_CHANNEL = "vision-alert-events"

def get_device():
    """Auto-select the best available hardware accelerator."""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def process_camera(camera_id: str, config: dict):
    """
    A direct, production-ready implementation of the proven test script logic,
    with entry/exit and alert features fully integrated.
    """
    try:
        # 1) Setup: RTSP URL, TCP, VideoCapture
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
        model = YOLO(MODEL_NAME)
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print(f"[{camera_id}] Loaded model '{MODEL_NAME}' and connected to Redis.", flush=True)

        # 3) Setup: Tracker and Zones (EXACTLY like your test script)
        tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=150, match_thresh=0.8, frame_rate=20)
        
        # CRITICAL: Using 'frame_resolution_wh' as proven necessary by your script
        resolution_wh = tuple(config.get("resolution_wh", (1920, 1080)))
        zones = [
            sv.PolygonZone(
                polygon=np.array(z["polygon"], np.int32),
                frame_resolution_wh=resolution_wh 
            ) for z in config["zones"]
        ]
        zone_names = [z["name"] for z in config["zones"]]
        print(f"[{camera_id}] Initialized tracker and {len(zones)} zones for resolution {resolution_wh}.", flush=True)

        # 4) State Management
        prev_zone_per_id = {}
        last_known_occupancy = {name: 0 for name in zone_names}
        alert_config = config.get('alert_config')
        alert_state = {"last_detection_time": time.time(), "alert_sent": False}
        
    except Exception as setup_e:
        print(f"[{camera_id}] A FATAL error during setup: {setup_e}", flush=True)
        import traceback; traceback.print_exc(); return

    print(f"[{camera_id}] Entering main loop...", flush=True)
    while True:
        ret, frame = cap.read()
        
        is_corrupt = (not ret or frame is None or frame.std() < 1)
        if is_corrupt:
            print(f"[{camera_id}] Corrupted frame! Preserving last count.", flush=True)
            data_payload = {"timestamp": time.time(), "camera_id": camera_id, "zone_counts": last_known_occupancy}
            r.publish(DATA_CHANNEL, json.dumps(data_payload))
            time.sleep(1);
            if not ret or frame is None:
                cap.release(); cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        
        # --- AI Processing (The Proven Logic) ---
        results = model(frame, conf=CONF_LOW, classes=[0], device=device, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(all_detections)
        stable_tracks = tracks[tracks.confidence > CONF_HIGH] if tracks is not None else sv.Detections.empty()

        print(f"[{camera_id}] Detections: {len(all_detections)}, Stable Tracks: {len(stable_tracks)}", flush=True)

        # --- Zone Occupancy ---
        occupancy_counts = {}
        for name, zone in zip(zone_names, zones):
            mask = zone.trigger(detections=stable_tracks) 
            count = int(np.sum(mask))
            occupancy_counts[name] = occupancy_counts.get(name, 0) + count
        
        # --- Entry/Exit and Alert Logic ---
        if stable_tracks.tracker_id is not None:
            for track_id, box in zip(stable_tracks.tracker_id, stable_tracks.xyxy):
                single_track_detection = sv.Detections(xyxy=np.array([box]))
                current_zone = None
                for name, zone in zip(zone_names, zones):
                    if zone.trigger(detections=single_track_detection)[0]:
                        current_zone = name; break
                
                prev_zone = prev_zone_per_id.get(track_id)
                if prev_zone != current_zone:
                    event = {"from": prev_zone or "outside", "to": current_zone or "outside"}
                    if event["from"] != event["to"]:
                        event_payload = {"timestamp": time.time(), "tracker_id": int(track_id), "camera_id": camera_id, "from_zone": event["from"], "to_zone": event["to"]}
                        r.publish(ZONE_CHANGE_CHANNEL, json.dumps(event_payload))
                    if current_zone: prev_zone_per_id[track_id] = current_zone
                    elif track_id in prev_zone_per_id: del prev_zone_per_id[track_id]
        
        if alert_config:
            alert_zone = alert_config['alert_zone_name']
            if occupancy_counts.get(alert_zone, 0) > 0:
                alert_state["last_detection_time"] = time.time(); alert_state["alert_sent"] = False
            else:
                absence_duration = time.time() - alert_state["last_detection_time"]
                threshold = alert_config.get('absence_threshold_seconds', 30)
                if not alert_state["alert_sent"] and absence_duration > threshold:
                    alert_payload = {"timestamp": time.time(), "camera_id": camera_id, "alert_type": "STAFF_ABSENCE", "message": f"ALERT: No staff in '{alert_zone}' for over {threshold}s."}
                    r.publish(ALERT_CHANNEL, json.dumps(alert_payload)); alert_state["alert_sent"] = True

        # --- PUBLISH DATA ---
        data_payload = {"timestamp": time.time(), "camera_id": camera_id, "zone_counts": occupancy_counts}
        r.publish(DATA_CHANNEL, json.dumps(data_payload))
        
        last_known_occupancy = occupancy_counts
        time.sleep(0.05)

if __name__ == "__main__":
    with open("configs/zones.json", "r") as f:
        full_config = json.load(f)
    procs = []
    for cam_id, cam_cfg in full_config.items():
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start()
        procs.append(p)
        print(f"Started camera process {cam_id}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down..."); [p.terminate() for p in procs]; print("All processes terminated.")