import os
import json
import time
import redis
import numpy as np
import torch
import cv2
from ultralytics import YOLO
import supervision as sv
from multiprocessing import Process

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11s.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL = "vision-data-events"
ALERT_CHANNEL = "vision-alert-events"

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def process_camera(camera_id, config):
    device = get_device()
    print(f"[{camera_id}] Starting process on device: {device}")

    rtsp_url = os.getenv(config['rtsp_url_env'])
    if not rtsp_url:
        print(f"[{camera_id}] ERROR: RTSP URL env var {config['rtsp_url_env']} not set.")
        return

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    # --- UPDATED INITIALIZATION ---
    model = YOLO(MODEL_NAME)
    
    # 1. Initialize tracker with a longer memory (track_buffer)
    tracker = sv.ByteTrack(
        track_buffer=60, # Frames to keep a track without detection
        track_thresh=0.25, # Detection confidence threshold to start a new track
        frame_rate=10      # Assumed frame rate
    )

    zones = [
        sv.PolygonZone(
            polygon=np.array(zone['polygon'], np.int32),
            frame_resolution_wh=tuple(config['resolution_wh'])
        ) for zone in config['zones']
    ]
    
    alert_config = config.get('alert_config')
    alert_state = { "last_detection_time": time.time(), "alert_sent": False }

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[{camera_id}] Error opening RTSP stream.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print(f"[{camera_id}] Reconnecting in 10s...")
            time.sleep(10)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        # --- REFINED DETECTION AND TRACKING LOGIC ---
        
        # 2. Detect with a LOW confidence threshold to capture all potential objects for the tracker
        results = model(frame, conf=0.1, classes=[0], device=device, verbose=False)[0]
        detections = sv.Detections.from_yolov8(results)

        # 3. Pass ALL detections to the tracker
        tracked_detections = tracker.update_with_detections(detections)
        
        # 4. NOW filter for high-confidence tracks to be used for counting
        # This ensures we count only "confirmed" objects, but tracking is stable
        high_confidence_tracks = tracked_detections[tracked_detections.confidence > CONFIDENCE]
        
        # ----------------------------------------------

        zone_counts = {}
        for i, zone in enumerate(zones):
            # Use the stable, high-confidence tracks for counting
            mask = zone.trigger(detections=high_confidence_tracks)
            count = int(np.sum(mask))
            
            zone_name = config['zones'][i]['name']
            zone_counts[zone_name] = zone_counts.get(zone_name, 0) + count

        # --- Alerting Logic (no changes needed) ---
        if alert_config:
            alert_zone_name = alert_config['alert_zone_name']
            if zone_counts.get(alert_zone_name, 0) > 0:
                alert_state["last_detection_time"] = time.time()
                alert_state["alert_sent"] = False
            else:
                absence_duration = time.time() - alert_state["last_detection_time"]
                threshold = alert_config['absence_threshold_seconds']
                
                if not alert_state["alert_sent"] and absence_duration > threshold:
                    alert_payload = { "timestamp": time.time(), "camera_id": camera_id, "alert_type": "STAFF_ABSENCE", "message": f"ALERT: No staff detected in '{alert_zone_name}' on {camera_id} for over {threshold} seconds." }
                    r.publish(ALERT_CHANNEL, json.dumps(alert_payload))
                    alert_state["alert_sent"] = True
                    print(f"[{camera_id}] ALERT PUBLISHED: {alert_payload['message']}")

        # --- Data Publishing (no changes needed) ---
        data_payload = { "timestamp": time.time(), "camera_id": camera_id, "zone_counts": zone_counts, }
        r.publish(DATA_CHANNEL, json.dumps(data_payload))
        
        time.sleep(0.1)

        
if __name__ == "__main__":
    print("--- Starting Vision Processor Service ---")
    with open('configs/zones.json', 'r') as f:
        full_config = json.load(f)

    processes = []
    for camera_id, camera_config in full_config.items():
        p = Process(target=process_camera, args=(camera_id, camera_config))
        p.start()
        processes.append(p)
        print(f"-> Started process for {camera_id}")

    for p in processes:
        p.join()