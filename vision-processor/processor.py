import os
import json
import time
from collections import defaultdict
from typing import Dict, Set

import cv2
import numpy as np
import redis
import torch
from ultralytics import YOLO
import supervision as sv
from multiprocessing import Process
from threading import Thread
from queue import Queue, Empty


# ===============================
# CONFIG (override via env)
# ===============================
MODEL_NAME        = os.getenv("MODEL_NAME", "yolo11l.pt")

# Detection / tracking
CONF_DETECT_LOW   = float(os.getenv("CONF_DETECT_LOW", 0.10))   # generous for tracker
CONF_STABLE       = float(os.getenv("CONF_STABLE", 0.35))       # for counting
IOU_NMS           = float(os.getenv("IOU_NMS", 0.30))           # separate close people

CONFIRM_THRESH = float(os.getenv("CONFIRM_THRESH", 0.30))  # confirm a track
KEEP_THRESH    = float(os.getenv("KEEP_THRESH", 0.15))

# Track stability
MIN_AGE_FRAMES    = int(os.getenv("MIN_AGE_FRAMES", 5))         # min age before affecting counts
ENTER_FRAMES      = int(os.getenv("ENTER_FRAMES", 3))           # hysteresis: frames inside to add
EXIT_FRAMES       = int(os.getenv("EXIT_FRAMES", 3))            # hysteresis: frames outside to remove
COAST_FRAMES      = int(os.getenv("COAST_FRAMES", 60))          # frames we allow ID to “miss” before cleanup

# RTSP guard
GRAY_STD_THRESH   = float(os.getenv("GRAY_STD_THRESH", 10.0))   # gray/corrupt frame std threshold

# Redis
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL      = "vision-data-events"
ALERT_CHANNEL     = "vision-alert-events"
ZONE_CHANGE_CH    = "vision-zone-change-events"

# Conversion windows (seconds)
ENTRY_TO_QUEUE_S  = int(os.getenv("ENTRY_TO_QUEUE_S", 60))
QUEUE_TO_HALL_S   = int(os.getenv("QUEUE_TO_HALL_S", 90))

# Unique visitor dedupe window in seconds (per entry)
UNIQUE_DEDUPE_S   = int(os.getenv("UNIQUE_DEDUPE_S", 300))      # 5 min

# Barista alert
BARISTA_ABSENCE_S = int(os.getenv("BARISTA_ABSENCE_S", 60))


# ===============================
# THREADED CAPTURE
# ===============================
class VideoCaptureThreaded:
    def __init__(self, src: str, name: str):
        self.src = src
        self.q = Queue(maxsize=1)
        self.is_running = True
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.t = Thread(target=self._reader, name=name, daemon=True)
        self.t.start()

    def _reader(self):
        print(f"[{self.t.name}] started for {self.src}", flush=True)
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(2.0)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except Empty: pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.is_running = False
        self.t.join()
        self.cap.release()


def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def is_corrupted(frame: np.ndarray, std_thresh: float = GRAY_STD_THRESH) -> bool:
    return frame is None or frame.size == 0 or frame.std() < std_thresh


# ===============================
# REDIS HELPERS
# ===============================
def seconds_until_midnight() -> int:
    now = time.localtime()
    midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday + 1, 0, 0, 0, 0, 0, -1))
    return int(max(60, midnight - time.mktime(now)))

def day_key(prefix: str) -> str:
    return f"{prefix}:{time.strftime('%Y-%m-%d', time.localtime())}"

# Sorted-set buffers for cross-camera conversion matching
def add_event_zset(r: redis.Redis, zset_key: str, member: str, ts: float, window_s: int):
    r.zadd(zset_key, {member: ts})
    r.zremrangebyscore(zset_key, 0, ts - window_s)   # drop stale

def pop_match(r: redis.Redis, zset_key: str, ts: float, window_s: int) -> bool:
    # get the earliest event within [ts-window, ts]
    candidates = r.zrangebyscore(zset_key, ts - window_s, ts, start=0, num=1)
    if candidates:
        r.zrem(zset_key, candidates[0])
        return True
    return False

# ===============================
# MAIN CAMERA LOOP
# ===============================
def process_camera(camera_id: str, config: dict):
    track_confirmed = {}

    rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")
    if not rtsp_url:
        print(f"[{camera_id}] FATAL: RTSP URL env var not set", flush=True)
        return

    cap = VideoCaptureThreaded(rtsp_url, name=f"RTSP-{camera_id}")
    time.sleep(1.5)

    device = get_device()
    print(f"[{camera_id}] Using device: {device}", flush=True)
    model = YOLO(MODEL_NAME)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try: r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"[{camera_id}] Redis error: {e}", flush=True); return

    tracker = sv.ByteTrack(
        track_thresh=0.05,                 # LOWER gate into the tracker
        track_buffer=COAST_FRAMES + 10,    # allow long coasting
        match_thresh=0.8,
        frame_rate=20
    )

    # zones
    zones_cfg = config["zones"]
    zone_names = [z["name"] for z in zones_cfg]
    zones = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32), frame_resolution_wh=(1920, 1080), trigger_anchors=[sv.Position.CENTER]) for z in zones_cfg]

    # derived names (if exist)
    has_queue  = any(n.lower() == "queue"  for n in zone_names)
    has_hall   = any(n.lower() == "hall"   for n in zone_names)
    has_entry  = any(n.lower() == "entry"  for n in zone_names)
    barista_zn = next((n for n in zone_names if "barista" in n.lower()), None)

    # state
    age_frames: Dict[int, int]  = defaultdict(int)
    miss_frames: Dict[int, int] = defaultdict(int)
    in_counts  = defaultdict(lambda: defaultdict(int))
    out_counts = defaultdict(lambda: defaultdict(int))
    zone_members: Dict[str, Set[int]] = { zn: set() for zn in zone_names }
    last_payload_counts = { zn: 0 for zn in zone_names }

    # barista alert state
    last_barista_seen = time.time()
    alert_sent = False

    print(f"[{camera_id}] Zones: {zone_names}", flush=True)

    while True:
        frame = cap.read()

        # corrupt frame → publish last good counts, do not mutate state
        if is_corrupted(frame):
            r.publish(DATA_CHANNEL, json.dumps({
                "timestamp": time.time(),
                "camera_id": camera_id,
                "zone_counts": last_payload_counts
            }))
            continue

        # DETECT & TRACK
        with torch.no_grad():
            res = model(
                frame,
                imgsz=int(os.getenv("IMG_SIZE", 1280)),   # was default 640 — 960/1280 helps a lot
                conf=CONF_DETECT_LOW,                     # e.g., 0.05–0.10, generous for tracker
                iou=IOU_NMS,                              # 0.25–0.35 separates close people
                classes=[0],
                device=device,
                verbose=False
            )[0]
        dets = sv.Detections.from_ultralytics(res)
        tracks = tracker.update_with_detections(dets)

        # update age/miss
        current_ids = set(tracks.tracker_id) if tracks.tracker_id is not None else set()
        for tid in current_ids:
            tid = int(tid)
            age_frames[tid] += 1
            miss_frames[tid] = 0
        for tid in list(age_frames.keys()):
            if tid not in current_ids:
                miss_frames[tid] += 1
                if miss_frames[tid] > COAST_FRAMES:
                    # cleanup all track state
                    age_frames.pop(tid, None)
                    miss_frames.pop(tid, None)
                    in_counts.pop(tid, None)
                    out_counts.pop(tid, None)
                    for zn in zone_names:
                        zone_members[zn].discard(tid)

        # filter stable tracks by age + conf
        stable_idx = []
        if tracks.tracker_id is not None:
            for i, tid in enumerate(tracks.tracker_id):
                tid = int(tid)
                conf_i = float(tracks.confidence[i])

                # must be alive long enough
                if age_frames[tid] < MIN_AGE_FRAMES:
                    continue

                # confirm if confidence is high enough once
                if not track_confirmed.get(tid, False):
                    if conf_i >= CONFIRM_THRESH:
                        track_confirmed[tid] = True
                    else:
                        # not confirmed yet; don't count it this frame
                        continue

                # already confirmed → allow it to stay even with lower confidence,
                # and also let coasting handle brief “no detection” gaps
                if conf_i >= KEEP_THRESH or miss_frames[tid] <= COAST_FRAMES:
                    stable_idx.append(i)
                else:
                    # very low confidence and already missed too long → drop confirmation
                    track_confirmed.pop(tid, None)

        stable = tracks[np.array(stable_idx, dtype=int)] if stable_idx else sv.Detections.empty()

        # HYSTERESIS MEMBERSHIP (anchor=center)
        # build masks once
        zone_masks = {}
        for zn, zone in zip(zone_names, zones):
            zone_masks[zn] = zone.trigger(detections=stable)

        # store whether a track just ENTERED a zone (for conversions)
        zone_enter_events = []  # tuples: (zone_name, track_id, ts)

        if stable.tracker_id is not None and len(stable) > 0:
            for idx, tid in enumerate(stable.tracker_id):
                tid = int(tid)
                for zn in zone_names:
                    inside = bool(zone_masks[zn][idx])
                    if inside:
                        in_counts[tid][zn]  += 1
                        out_counts[tid][zn]  = 0
                        if in_counts[tid][zn] == ENTER_FRAMES:
                            # first time we cross the enter threshold → enter event
                            zone_members[zn].add(tid)
                            zone_enter_events.append((zn, tid, time.time()))
                            # optional: publish enter event
                            r.publish(ZONE_CHANGE_CH, json.dumps({
                                "timestamp": time.time(), "camera_id": camera_id,
                                "tracker_id": tid, "event": "enter", "zone": zn
                            }))
                    else:
                        out_counts[tid][zn] += 1
                        in_counts[tid][zn]   = 0
                        if out_counts[tid][zn] == EXIT_FRAMES:
                            zone_members[zn].discard(tid)
                            r.publish(ZONE_CHANGE_CH, json.dumps({
                                "timestamp": time.time(), "camera_id": camera_id,
                                "tracker_id": tid, "event": "exit", "zone": zn
                            }))

        # Occupancy after updates
        occupancy = { zn: len(zone_members[zn]) for zn in zone_names }

        # ===============================
        # CONVERSIONS & UNIQUE VISITORS
        # ===============================
        now = time.time()
        day_visitors_key = day_key("visitors:count")             # counter
        dedupe_prefix    = day_key("visitors:seen")              # per-day dedupe namespace

        # ZSET buffers used across cameras
        z_entry = day_key("convbuf:entry")
        z_queue = day_key("convbuf:queue")

        for zn, tid, ts in zone_enter_events:
            # 1) UNIQUE VISITORS (count only on entry zone)
            if zn.lower() == "entry":
                # per-day dedupe key for this track (camera+id)
                entry_key = day_key("entry_count")
                r.incr(entry_key); r.expire(entry_key, seconds_until_midnight())

                dedupe_key = f"{dedupe_prefix}:{camera_id}:{tid}"
                # if new within window, count as a new unique visitor for the day
                added = r.set(dedupe_key, "1", ex=max(UNIQUE_DEDUPE_S, 60), nx=True)
                # also set TTL until midnight so we can recount tomorrow
                r.expire(dedupe_key, seconds_until_midnight())
                if added:
                    r.incr(day_visitors_key)
                    r.expire(day_visitors_key, seconds_until_midnight())

                # also add to entry buffer for entry->queue conversion
                add_event_zset(r, z_entry, f"{camera_id}:{tid}:{int(ts)}", ts, ENTRY_TO_QUEUE_S)

            # 2) Add queue enters to buffer and try to match with entry
            if zn.lower() == "queue":
                qent_key = day_key("queue_entries")
                r.incr(qent_key); r.expire(qent_key, seconds_until_midnight())
                # match any entry event within window (global, any camera)
                if pop_match(r, z_entry, ts, ENTRY_TO_QUEUE_S):
                    conv_key = day_key("conv:entry_to_queue")
                    r.incr(conv_key)
                    r.expire(conv_key, seconds_until_midnight())
                # Add this queue enter to buffer for queue->hall
                add_event_zset(r, z_queue, f"{camera_id}:{tid}:{int(ts)}", ts, QUEUE_TO_HALL_S)

            # 3) Hall enters try to match with queue within window
            if zn.lower() == "hall":
                if pop_match(r, z_queue, ts, QUEUE_TO_HALL_S):
                    conv_key = day_key("conv:queue_to_hall")
                    r.incr(conv_key)
                    r.expire(conv_key, seconds_until_midnight())

        # ===============================
        # BARISTA ALERT
        # ===============================
        if barista_zn:
            if occupancy.get(barista_zn, 0) > 0:
                last_barista_seen = now
                alert_sent = False
            else:
                if not alert_sent and (now - last_barista_seen) > BARISTA_ABSENCE_S:
                    r.publish(ALERT_CHANNEL, json.dumps({
                        "timestamp": now, "camera_id": camera_id,
                        "alert_type": "STAFF_ABSENCE",
                        "message": f"No barista in '{barista_zn}' > {BARISTA_ABSENCE_S}s"
                    }))
                    alert_sent = True

        # ===============================
        # PUBLISH CURRENT OCCUPANCY
        # ===============================
        payload = {
            "timestamp": now,
            "camera_id": camera_id,
            "zone_counts": occupancy
        }
        r.publish(DATA_CHANNEL, json.dumps(payload))
        last_payload_counts = occupancy


# ===============================
# BOOT
# ===============================
if __name__ == "__main__":
    with open("configs/zones.json", "r") as f:
        full_cfg = json.load(f)

    processes = []
    for cam_id, cam_cfg in full_cfg.items():
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start()
        processes.append(p)
        print(f"Started camera process {cam_id}", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down…", flush=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("All processes terminated.", flush=True)