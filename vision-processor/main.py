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
from threading import Thread
from queue import Queue, Empty

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "yolo11m.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", 0.2))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL = "vision-data-events"
ALERT_CHANNEL = "vision-alert-events"
ZONE_CHANGE_CHANNEL = "vision-zone-change-events"


class VideoCaptureThreaded:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        # Use a queue to hold the latest frame. maxsize=1 ensures we only process the most recent frame.
        self.q = Queue(maxsize=1)
        self.is_running = True
        
        # Start the reader thread
        self.thread = Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        """Internal method to read frames from the stream and put them in a queue."""
        print(f"[VideoCaptureThread] Reader thread started for {self.src}")
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[VideoCaptureThread] Stream connection lost for {self.src}. Attempting to reconnect...")
                self.cap.release()
                time.sleep(5)  # Wait 5 seconds before trying to reconnect
                self.cap = cv2.VideoCapture(self.src)
                continue
            
            # If the queue is full, it means the main thread is slow. Discard the old frame.
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except Empty:
                    pass
            self.q.put(frame)

    def read(self):
        """Blocks until a frame is available and returns it."""
        return self.q.get()

    def release(self):
        """Stops the reader thread and releases the video capture object."""
        self.is_running = False
        self.thread.join()  # Wait for the thread to finish
        self.cap.release()
        print(f"[VideoCaptureThread] Released stream {self.src}")


def get_device():
    """Auto-selects the best available hardware accelerator."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available(): # For Apple Silicon
        return 'mps'
    return 'cpu'


def process_camera(camera_id, config):
    try:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        device = get_device()
        print(f"[{camera_id}] Starting process on device: {device}. Forcing RTSP over TCP.")

        rtsp_url = os.getenv(config['rtsp_url_env'])
        if not rtsp_url:
            print(f"[{camera_id}] FATAL: RTSP URL env var {config['rtsp_url_env']} not set. Exiting.")
            return

        print(f"[{camera_id}] Loading model: {MODEL_NAME}...")
        model = YOLO(MODEL_NAME)
        print(f"[{camera_id}] Model loaded.")

        print(f"[{camera_id}] Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print(f"[{camera_id}] Successfully connected to Redis.")

        tracker = sv.ByteTrack()
        zones = [sv.PolygonZone(polygon=np.array(zone['polygon'], np.int32)) for zone in config['zones']]
        zone_names = [zone['name'] for zone in config['zones']]
        print(f"[{camera_id}] Tracker and {len(zones)} zones initialized.")

        tracker_last_known_zone = {}
        alert_config = config.get('alert_config')
        alert_state = { "last_detection_time": time.time(), "alert_sent": False }
        
        # --- Use the new threaded capture ---
        print(f"[{camera_id}] Opening threaded RTSP stream...")
        cap = VideoCaptureThreaded(rtsp_url)
        time.sleep(2)  # Give the thread a moment to connect and grab the first frame
        print(f"[{camera_id}] Stream capture thread started. Entering main processing loop.")

    except Exception as setup_e:
        print(f"[{camera_id}] A FATAL error occurred during setup: {setup_e}")
        import traceback
        traceback.print_exc()
        return

    has_printed_resolution = False
    while True:
        print(f"[{camera_id}] Processing frame...")
        try:
            success, frame = cap.read()
            if not success:
                print(f"[{camera_id}] Failed to grab frame. Reconnecting in 10s...")
                time.sleep(10)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue

            if not has_printed_resolution:
                h, w, _ = frame.shape
                print(f"[{camera_id}] [DIAGNOSTIC] Actual Frame Resolution: {w}x{h}. Configured Resolution: {config['resolution_wh']}")
                has_printed_resolution = True

            # --- Refined Detection and Tracking Logic ---
            results = model(frame, conf=0.1, classes=[0], device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            if len(detections) > 0:
                print(f"[{camera_id}] [DIAGNOSTIC] Raw Detections: {len(detections)}. Confidence scores: {[f'{c:.2f}' for c in detections.confidence]}")

            tracked_detections = tracker.update_with_detections(detections)
            high_confidence_tracks = tracked_detections[tracked_detections.confidence > CONFIDENCE]
            
            # [DIAGNOSTIC] See if any tracks survive the confidence filter
            if len(high_confidence_tracks) > 0:
                print(f"[{camera_id}] [DIAGNOSTIC] High-Confidence Tracks Found: {len(high_confidence_tracks)}")
            
            # --- 1. Instantaneous Occupancy Counting ---
            occupancy_counts = {}
            for i, zone in enumerate(zones):
                mask = zone.trigger(detections=high_confidence_tracks)
                count = int(np.sum(mask))

                # [DIAGNOSTIC] Check if any track is ever in a zone
                if count > 0:
                    print(f"[{camera_id}] [DIAGNOSTIC] {count} track(s) detected in zone '{zone_names[i]}'")

                zone_name = zone_names[i]
                occupancy_counts[zone_name] = occupancy_counts.get(zone_name, 0) + int(np.sum(mask))
            
            occupancy_payload = { "timestamp": time.time(), "camera_id": camera_id, "zone_counts": occupancy_counts }
            r.publish(DATA_CHANNEL, json.dumps(occupancy_payload))

            # --- 2. Entry/Exit Event Detection ---
            for track_id, box in zip(high_confidence_tracks.tracker_id, high_confidence_tracks.xyxy):
                current_zone_for_track = None
                for i, zone in enumerate(zones):
                    detection_for_track = sv.Detections(xyxy=np.array([box]))
                    if zone.trigger(detections=detection_for_track)[0]:
                        current_zone_for_track = zone_names[i]
                        break
                
                previous_zone = tracker_last_known_zone.get(track_id)

                if current_zone_for_track != previous_zone:
                    event_payload = {
                        "timestamp": time.time(), "tracker_id": int(track_id), "camera_id": camera_id,
                        "from_zone": previous_zone or "outside",
                        "to_zone": current_zone_for_track or "outside"
                    }
                    if event_payload["from_zone"] != event_payload["to_zone"]:
                        r.publish(ZONE_CHANGE_CHANNEL, json.dumps(event_payload))
                
                if current_zone_for_track:
                    tracker_last_known_zone[track_id] = current_zone_for_track
                elif track_id in tracker_last_known_zone:
                    del tracker_last_known_zone[track_id]

            # --- 3. Alerting Logic ---
            if alert_config:
                alert_zone_name = alert_config['alert_zone_name']
                if occupancy_counts.get(alert_zone_name, 0) > 0:
                    alert_state["last_detection_time"] = time.time()
                    alert_state["alert_sent"] = False
                else:
                    absence_duration = time.time() - alert_state["last_detection_time"]
                    threshold = alert_config['absence_threshold_seconds']
                    
                    if not alert_state["alert_sent"] and absence_duration > threshold:
                        alert_payload = {
                            "timestamp": time.time(), "camera_id": camera_id, "alert_type": "STAFF_ABSENCE",
                            "message": f"ALERT: No staff detected in '{alert_zone_name}' on {camera_id} for over {threshold} seconds."
                        }
                        r.publish(ALERT_CHANNEL, json.dumps(alert_payload))
                        alert_state["alert_sent"] = True

            # Short sleep to prevent CPU overload if frame rate is very high
            time.sleep(0.05)
        
        except Exception as e:
            print(f"[{camera_id}] An error occurred in the main loop: {e}")
            time.sleep(10) # Wait before retrying

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