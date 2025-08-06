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
# Low confidence for initial detection to feed the tracker
CONF_LOW = 0.1
# High confidence to promote a 'Candidate' to 'Confirmed'
CONF_HIGH_TO_CONFIRM = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# --- STATEFUL TRACKING PARAMETERS (Tuning Knobs) ---
# How many consecutive frames a track must be seen with high confidence to be 'Confirmed'
FRAMES_TO_CONFIRM = 10
# How many frames a 'Confirmed' track can be 'Coasting' (undetected) before being 'Lost'
FRAMES_TO_COAST = 30 

# Redis Configuration
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
    """Process loop for a single camera with stateful, stable tracking (Algorithm of Intent)."""
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

        # 3) Setup: Tracker and Zones
        tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=FRAMES_TO_COAST + 10, match_thresh=0.8, frame_rate=30)
        zones = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32), frame_resolution_wh=tuple(config['resolution_wh'])) for z in config["zones"]]
        zone_names = [z["name"] for z in config["zones"]]
        print(f"[{camera_id}] Initialized ROBUST tracker and {len(zones)} zones.", flush=True)

        # 4) Setup: State Management
        tracked_objects = {} # Main dictionary to hold the state of each object
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
            time.sleep(5); cap.release(); cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG); continue
        
        # --- DETECTION & TRACKING ---
        results = model(frame, conf=CONF_LOW, classes=[0], device=device, verbose=False)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        tracks = tracker.update_with_detections(all_detections)

        # --- "ALGORITHM OF INTENT" - STATEFUL LOGIC ---
        
        # Get IDs from the current frame's tracks
        current_frame_track_ids = set(tracks.tracker_id) if tracks.tracker_id is not None else set()

        # --- A. Match and Update Existing Objects ---
        for obj_id in list(tracked_objects.keys()):
            if obj_id in current_frame_track_ids:
                # Object is still visible
                tracked_objects[obj_id]['misses'] = 0
                tracked_objects[obj_id]['seen_frames'] += 1
                
                # Promote 'Candidate' to 'Confirmed' if seen long enough with high confidence
                if tracked_objects[obj_id]['state'] == 'Candidate':
                    # Find the confidence for this specific track
                    track_confidence = tracks[tracks.tracker_id == obj_id].confidence[0]
                    if track_confidence > CONF_HIGH_TO_CONFIRM:
                        tracked_objects[obj_id]['high_conf_frames'] += 1
                    if tracked_objects[obj_id]['high_conf_frames'] >= FRAMES_TO_CONFIRM:
                        tracked_objects[obj_id]['state'] = 'Confirmed'
            else:
                # Object is NOT visible in this frame
                tracked_objects[obj_id]['misses'] += 1
                if tracked_objects[obj_id]['state'] in ['Confirmed', 'Coasting']:
                    if tracked_objects[obj_id]['misses'] <= FRAMES_TO_COAST:
                        tracked_objects[obj_id]['state'] = 'Coasting'
                    else:
                        tracked_objects[obj_id]['state'] = 'Lost'

        # --- B. Add New Candidate Objects ---
        if tracks.tracker_id is not None:
            for i in range(len(tracks)):
                track_id = tracks.tracker_id[i]
                if track_id not in tracked_objects:
                    tracked_objects[track_id] = {
                        'state': 'Candidate',
                        'misses': 0,
                        'seen_frames': 1,
                        'high_conf_frames': 1 if tracks.confidence[i] > CONF_HIGH_TO_CONFIRM else 0,
                    }

        # --- C. Prune Lost Objects ---
        lost_ids = [obj_id for obj_id, data in tracked_objects.items() if data['state'] == 'Lost']
        for obj_id in lost_ids:
            del tracked_objects[obj_id]
            if obj_id in prev_zone_per_id:
                del prev_zone_per_id[obj_id]
        
        # --- D. Final Counting and Zone Logic ---
        # The final set of people to count are those who are 'Confirmed' or 'Coasting'
        stable_track_ids = [obj_id for obj_id, data in tracked_objects.items() if data['state'] in ['Confirmed', 'Coasting']]
        stable_tracks = sv.Detections.empty()
        if len(stable_track_ids) > 0 and tracks.tracker_id is not None:
            stable_mask = np.isin(tracks.tracker_id, stable_track_ids)
            stable_tracks = tracks[stable_mask]

        print(f"[{camera_id}] Detections: {len(all_detections)}, Stable Count: {len(stable_track_ids)}", flush=True)

        # --- Zone Occupancy ---
        occupancy_counts = {}
        for name, zone in zip(zone_names, zones):
            mask = zone.trigger(detections=stable_tracks)
            count = int(np.sum(mask))
            occupancy_counts[name] = occupancy_counts.get(name, 0) + count

            
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