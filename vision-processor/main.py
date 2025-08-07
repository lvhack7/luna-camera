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
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11l.pt")
CONF_LOW = 0.1
CONF_HIGH_TO_CONFIRM = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
FRAMES_TO_CONFIRM = 10
FRAMES_TO_COAST = 60
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

def is_frame_corrupted(frame: np.ndarray, threshold: float = 2) -> bool:
    """Checks if a frame is likely corrupted (e.g., solid gray) by checking its pixel variance."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_frame)
    
    # --- ADD THIS DIAGNOSTIC PRINT ---
    # print(f"[Corruption Check] Frame Standard Deviation: {std_dev:.2f}")
    # ---------------------------------
    
    return std_dev < threshold

def process_camera(camera_id: str, config: dict):
    """Main processing loop with stateful tracking and resilience to corrupted frames."""
    try:
        # 1) Setup
        rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")

        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[{camera_id}] ERROR: cannot open stream {rtsp_url}", flush=True); return
        print(f"[{camera_id}] Opened stream.", flush=True)

        device = get_device()
        model = YOLO(MODEL_NAME)
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print(f"[{camera_id}] Loaded model and connected to Redis.", flush=True)

        tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=FRAMES_TO_COAST + 10, match_thresh=0.8, frame_rate=30)
        zones = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32), frame_resolution_wh=(1920, 1080)) for z in config["zones"]]
        zone_names = [z["name"] for z in config["zones"]]
        
        # 2) State Management
        tracked_objects = {}
        prev_zone_per_id = {}
        last_known_occupancy = {name: 0 for name in zone_names}
        alert_config = config.get('alert_config')
        alert_state = {"last_detection_time": time.time(), "alert_sent": False}
        
    except Exception as setup_e:
        print(f"[{camera_id}] FATAL error during setup: {setup_e}", flush=True)
        import traceback; traceback.print_exc(); return

    print(f"[{camera_id}] Entering main loop...", flush=True)
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"[{camera_id}] WARNING: empty frame, reconnecting...", flush=True)
            time.sleep(5); cap.release(); cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG); continue
        
        if frame.std() < 10:
            print(f"[{camera_id}] Corrupted frame detected! Preserving last count.", flush=True)
            data_payload = {"timestamp": time.time(), "camera_id": camera_id, "zone_counts": last_known_occupancy}
            r.publish(DATA_CHANNEL, json.dumps(data_payload))
            time.sleep(0.5); continue

        # --- AI Processing ---
        results = model(frame, conf=CONF_LOW, classes=[0], device=device, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(all_detections)

        # --- "Algorithm of Intent" - Stateful Logic ---
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
        for obj_id in lost_ids:
            del tracked_objects[obj_id]
            if obj_id in prev_zone_per_id: del prev_zone_per_id[obj_id]
        
        stable_track_ids = {obj_id for obj_id, data in tracked_objects.items() if data['state'] in ['Confirmed', 'Coasting']}
        stable_tracks = sv.Detections.empty()
        if len(stable_track_ids) > 0 and tracks.tracker_id is not None:
            stable_mask = np.isin(tracks.tracker_id, list(stable_track_ids))
            stable_tracks = tracks[stable_mask]

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

        # --- Publish Data ---
        data_payload = {"timestamp": time.time(), "camera_id": camera_id, "zone_counts": occupancy_counts}
        r.publish(DATA_CHANNEL, json.dumps(data_payload))
        last_known_occupancy = occupancy_counts
        time.sleep(0.05)

if __name__ == "__main__":
    with open("configs/zones.json", "r") as f: full_config = json.load(f)
    procs = []; [procs.append(p) for cam_id, cam_cfg in full_config.items() if (p := Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True), p.start(), print(f"Started camera process {cam_id}"))]
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down..."); [p.terminate() for p in procs if p.is_alive()]; [p.join() for p in procs]; print("All processes terminated.")