import os, json, time, signal
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from collections import defaultdict
from typing import Dict, Set, List

import cv2, numpy as np, redis, torch
from ultralytics import YOLO
import supervision as sv
from multiprocessing import Process
from threading import Thread
from queue import Queue, Empty

# ==================== GLOBAL SETTINGS ====================
ALMATY_TZ        = ZoneInfo("Asia/Almaty")
OPEN_T           = dtime(8, 30)           # 08:30
CLOSE_T          = dtime(23, 59, 59)      # ~24:00
MODEL_NAME       = os.getenv("MODEL_NAME", "yolo11l.pt")

CONF_DETECT_LOW  = float(os.getenv("CONF_DETECT_LOW", 0.10))
IOU_NMS          = float(os.getenv("IOU_NMS", 0.30))
CONFIRM_THRESH   = float(os.getenv("CONFIRM_THRESH", 0.30))
KEEP_THRESH      = float(os.getenv("KEEP_THRESH", 0.15))
IMG_SIZE         = int(os.getenv("IMG_SIZE", 1280))

ENTER_SECONDS    = float(os.getenv("ENTER_SECONDS", 0.8))
EXIT_SECONDS     = float(os.getenv("EXIT_SECONDS", 2.5))
MIN_AGE_FRAMES   = int(os.getenv("MIN_AGE_FRAMES", 5))
COAST_FRAMES     = int(os.getenv("COAST_FRAMES", 60))
GRAY_STD_THRESH  = float(os.getenv("GRAY_STD_THRESH", 10.0))

REDIS_HOST       = os.getenv("REDIS_HOST", "redis")
REDIS_PORT       = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL     = "vision-data-events"
ZONE_CHANGE_CH   = "vision-zone-change-events"
ALERT_CHANNEL    = "vision-alert-events"

# Conversion windows (s)
ORDER_TO_PICKUP_S = int(os.getenv("ORDER_TO_PICKUP_S", 900))
PICKUP_TO_HALL_S  = int(os.getenv("PICKUP_TO_HALL_S", 900))
PICKUP_TO_EXIT_S  = int(os.getenv("PICKUP_TO_EXIT_S", 900))

def day_key(prefix: str, dt: datetime | None = None) -> str:
    if dt is None: dt = datetime.now(ALMATY_TZ)
    return f"{prefix}:{dt.strftime('%Y-%m-%d')}"

def seconds_until_midnight(dt: datetime | None = None) -> int:
    if dt is None: dt = datetime.now(ALMATY_TZ)
    next_midnight = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60, int((next_midnight - dt).total_seconds()))

# ==================== CAPTURE ====================
class VideoCaptureThreaded:
    def __init__(self, src: str, name: str):
        self.src = src
        self.q = Queue(maxsize=1)
        self.is_running = True
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.t = Thread(target=self._reader, name=name, daemon=True); self.t.start()
    def _reader(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release(); time.sleep(2.0)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG); continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except Empty: pass
            self.q.put(frame)
    def read(self): return self.q.get()
    def release(self): self.is_running = False; self.t.join(); self.cap.release()

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def is_corrupted(frame: np.ndarray) -> bool:
    return frame is None or frame.size == 0 or frame.std() < GRAY_STD_THRESH

# ==================== PER-CAMERA WORKER ====================
def process_camera(camera_id: str, config: dict):
    # Track-level state
    track_confirmed: Dict[int, bool] = {}
    # Per-track/per-polygon membership timers
    zone_state = defaultdict(lambda: defaultdict(lambda: {"first_in": None, "last_in": None, "is_member": False}))

    rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")
    if not rtsp_url:
        print(f"[{camera_id}] FATAL: RTSP URL not set"); return

    cap = VideoCaptureThreaded(rtsp_url, name=f"RTSP-{camera_id}")
    time.sleep(1.5)

    device = get_device()
    print("MODEL: ", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try: r.ping()
    except Exception as e: print(f"[{camera_id}] Redis error: {e}"); return

    tracker = sv.ByteTrack(track_thresh=0.05, track_buffer=COAST_FRAMES+10, match_thresh=0.8, frame_rate=20)

    zones_cfg = config["zones"]
    names = [z["name"] for z in zones_cfg]
    names_lc = [n.lower() for n in names]
    zones = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32), frame_resolution_wh=(1920, 1080), triggering_position=sv.Position.CENTER) for z in zones_cfg]

    # Index sets
    idx_hall   = [i for i,n in enumerate(names_lc) if n == "hall"]
    idx_queue  = [i for i,n in enumerate(names_lc) if n == "queue"]
    idx_order  = [i for i,n in enumerate(names_lc) if n == "order"]
    idx_pickup = [i for i,n in enumerate(names_lc) if n == "pickup"]
    idx_hall1  = [i for i,n in enumerate(names_lc) if n == "hall1"]
    idx_hall2  = [i for i,n in enumerate(names_lc) if n == "hall2"]
    idx_exit   = [i for i,n in enumerate(names_lc) if n == "exit"]
    idx_barista= [i for i,n in enumerate(names_lc) if n == "barista"]

    # Journey state (201 only)
    journey: Dict[int, dict] = defaultdict(dict)

    # Staff alert state (601)
    last_barista_seen = time.time()
    alert_sent = False
    barista_alert_cfg = config.get("alert_config")

    # Live counts cache
    last_payload_counts = {"hall": 0, "queue": 0, "barista": 0}

    # Local age/miss
    age_frames: Dict[int, int]  = defaultdict(int)
    miss_frames: Dict[int, int] = defaultdict(int)

    print(f"[{camera_id}] Zones: {names}")

    while True:
        frame = cap.read(); now = time.time()
        if is_corrupted(frame):
            r.publish(DATA_CHANNEL, json.dumps({"timestamp": now, "camera_id": camera_id, "zone_counts": last_payload_counts}))
            continue

        with torch.no_grad():
            res = model(frame, imgsz=IMG_SIZE, conf=CONF_DETECT_LOW, iou=IOU_NMS, classes=[0], device=device, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        tracks = tracker.update_with_detections(dets)

        # Age / miss bookkeeping
        current_ids = set(tracks.tracker_id) if tracks.tracker_id is not None else set()
        for tid in current_ids:
            tid = int(tid); age_frames[tid] += 1; miss_frames[tid] = 0
        for tid in list(age_frames.keys()):
            if tid not in current_ids:
                miss_frames[tid] += 1
                if miss_frames[tid] > COAST_FRAMES:
                    age_frames.pop(tid, None); miss_frames.pop(tid, None)
                    zone_state.pop(tid, None); journey.pop(tid, None); track_confirmed.pop(tid, None)

        # Confirm / keep
        stable_idx = []
        if tracks.tracker_id is not None:
            for i, tid in enumerate(tracks.tracker_id):
                tid = int(tid); conf_i = float(tracks.confidence[i])
                if age_frames[tid] < MIN_AGE_FRAMES: continue
                if not track_confirmed.get(tid, False):
                    if conf_i >= CONFIRM_THRESH: track_confirmed[tid] = True
                    else: continue
                if conf_i >= KEEP_THRESH or miss_frames[tid] <= COAST_FRAMES:
                    stable_idx.append(i)
                else:
                    track_confirmed.pop(tid, None)

        stable = tracks[np.array(stable_idx, dtype=int)] if stable_idx else sv.Detections.empty()

        # Zone masks (CENTER)
        zone_masks: List[np.ndarray] = [np.array([], dtype=bool)] * len(zones)
        if len(stable) > 0 and stable.tracker_id is not None:
            for i, zone in enumerate(zones):
                zone_masks[i] = zone.trigger(detections=stable)

        # Per-polygon time hysteresis → enter events
        enter_events: List[tuple] = []  # (zname, tid, ts)
        for di, detection_data in enumerate(stable):
            tracker_id = detection_data[4]

            if tracker_id is None:
                continue
            tid = int(tracker_id)

            # Now, check its membership in each zone
            for pi in range(len(zones)):
                inside = bool(zone_masks[pi][di])
                st = zone_state[tid][pi]
                zname = names_lc[pi]
                if inside:
                    if st["first_in"] is None: st["first_in"] = now
                    st["last_in"] = now
                    if not st["is_member"] and (now - st["first_in"]) >= ENTER_SECONDS:
                        st["is_member"] = True
                        enter_events.append((zname, tid, now))
                        r.publish(ZONE_CHANGE_CH, json.dumps({"timestamp": now,"camera_id": camera_id,"tracker_id": tid,"event":"enter","zone": zname}))
                else:
                    st["first_in"] = None
                    if st["is_member"] and st["last_in"] and (now - st["last_in"]) >= EXIT_SECONDS:
                        st["is_member"] = False
                        r.publish(ZONE_CHANGE_CH, json.dumps({"timestamp": now,"camera_id": camera_id,"tracker_id": tid,"event":"exit","zone": zname}))


        # ========== CONVERSIONS: camera_201 ONLY ==========
        if camera_id == "camera_201" and enter_events:
            rconn = r
            # Redis keys (daily)
            k_order_unique      = day_key("order_unique")
            k_pickup_unique     = day_key("pickup_unique")
            k_conv_o2p          = day_key("conv:order_to_pickup")
            k_conv_p2h          = day_key("conv:pickup_to_hall")
            k_conv_p2e          = day_key("conv:pickup_to_exit")
            k_sum_o2p           = day_key("sum:o2p_s");  k_cnt_o2p = day_key("cnt:o2p")
            k_sum_p2h           = day_key("sum:p2h_s");  k_cnt_p2h = day_key("cnt:p2h")
            k_sum_p2e           = day_key("sum:p2e_s");  k_cnt_p2e = day_key("cnt:p2e")

            for zname, tid, ts in enter_events:
                j = journey[tid]
                if zname == "order":
                    ded = day_key(f"seen:order:{camera_id}:{tid}")
                    if rconn.set(ded, "1", ex=seconds_until_midnight(), nx=True):
                        rconn.incr(k_order_unique); rconn.expire(k_order_unique, seconds_until_midnight())
                    j.setdefault("order_enter_ts", ts); j["last_ts"] = ts

                elif zname == "pickup":
                    ded = day_key(f"seen:pickup:{camera_id}:{tid}")
                    if rconn.set(ded, "1", ex=seconds_until_midnight(), nx=True):
                        rconn.incr(k_pickup_unique); rconn.expire(k_pickup_unique, seconds_until_midnight())
                    if "order_enter_ts" in j and not j.get("counted_o2p"):
                        if ts - j["order_enter_ts"] <= ORDER_TO_PICKUP_S:
                            rconn.incr(k_conv_o2p); rconn.expire(k_conv_o2p, seconds_until_midnight())
                            rconn.incrbyfloat(k_sum_o2p, float(ts - j["order_enter_ts"]))
                            rconn.incr(k_cnt_o2p); rconn.expire(k_sum_o2p, seconds_until_midnight()); rconn.expire(k_cnt_o2p, seconds_until_midnight())
                            j["counted_o2p"] = True
                    j.setdefault("pickup_enter_ts", ts); j["done_after_pickup"] = False; j["last_ts"] = ts

                elif zname in ("hall1", "hall2"):
                    if "pickup_enter_ts" in j and not j.get("done_after_pickup"):
                        if (ts - j["pickup_enter_ts"]) <= PICKUP_TO_HALL_S:
                            rconn.incr(k_conv_p2h); rconn.expire(k_conv_p2h, seconds_until_midnight())
                            rconn.incrbyfloat(k_sum_p2h, float(ts - j["pickup_enter_ts"]))
                            rconn.incr(k_cnt_p2h); rconn.expire(k_sum_p2h, seconds_until_midnight()); rconn.expire(k_cnt_p2h, seconds_until_midnight())
                        j["done_after_pickup"] = True

                elif zname == "exit":
                    if "pickup_enter_ts" in j and not j.get("done_after_pickup"):
                        if (ts - j["pickup_enter_ts"]) <= PICKUP_TO_EXIT_S:
                            rconn.incr(k_conv_p2e); rconn.expire(k_conv_p2e, seconds_until_midnight())
                            rconn.incrbyfloat(k_sum_p2e, float(ts - j["pickup_enter_ts"]))
                            rconn.incr(k_cnt_p2e); rconn.expire(k_sum_p2e, seconds_until_midnight()); rconn.expire(k_cnt_p2e, seconds_until_midnight())
                        j["done_after_pickup"] = True

        # ========== LIVE OCCUPANCY ==========
        def count_group(name: str) -> int:
            idxs = [i for i,n in enumerate(names_lc) if n == name]
            if not idxs: return 0
            ids: Set[int] = set()
            for tid, states in zone_state.items():
                for i in idxs:
                    st = states.get(i)
                    if st and st["is_member"]:
                        ids.add(tid); break
            return len(ids)

        occ = {"hall": count_group("hall"), "queue": count_group("queue")}
        # Barista presence (only on 601 if present)
        if "barista" in names_lc:
            present = 1 if count_group("barista") > 0 else 0
            occ["barista"] = present
            if present:
                last_barista_seen = now; alert_sent = False
            elif barista_alert_cfg and not alert_sent and (now - last_barista_seen) > barista_alert_cfg.get("absence_threshold_seconds", 30):
                r.publish(ALERT_CHANNEL, json.dumps({
                    "timestamp": now, "camera_id": camera_id,
                    "alert_type": "STAFF_ABSENCE",
                    "message": f"No barista in 'barista' > {barista_alert_cfg.get('absence_threshold_seconds', 30)}s"
                }))
                alert_sent = True

        r.publish(DATA_CHANNEL, json.dumps({"timestamp": now, "camera_id": camera_id, "zone_counts": occ}))
        last_payload_counts = occ

# ==================== SCHEDULER (08:30 → 24:00) ====================
def within_business_hours(dt: datetime) -> bool:
    t = dt.timetz()
    return OPEN_T <= t <= CLOSE_T

def seconds_until(target_dt: datetime, now: datetime) -> int:
    return max(1, int((target_dt - now).total_seconds()))

def next_open(now: datetime) -> datetime:
    start = now.replace(hour=OPEN_T.hour, minute=OPEN_T.minute, second=0, microsecond=0)
    return start if now <= start else (start + timedelta(days=1))

def next_close(now: datetime) -> datetime:
    end = now.replace(hour=CLOSE_T.hour, minute=CLOSE_T.minute, second=CLOSE_T.second, microsecond=0)
    return end if now <= end else (end + timedelta(days=1))

def spawn_all(full_cfg: dict) -> list[Process]:
    procs: list[Process] = []
    for cam_id, cam_cfg in full_cfg.items():
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start(); procs.append(p)
        print(f"[scheduler] started {cam_id} pid={p.pid}")
    return procs

def kill_all(procs: list[Process]):
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        if p.is_alive():
            p.join(timeout=5)
    print("[scheduler] all workers stopped")

if __name__ == "__main__":
    with open("configs/zones.json", "r") as f:
        full_config = json.load(f)

    procs: list[Process] = []
    try:
        while True:
            now = datetime.now(ALMATY_TZ)
            if within_business_hours(now):
                if not procs:
                    procs = spawn_all(full_config)
                # sleep until close (or small step)
                wait = min(30, seconds_until(next_close(now), now))
                time.sleep(wait)
            else:
                if procs:
                    kill_all(procs); procs = []
                # sleep until next open
                wait = min(60, seconds_until(next_open(now), now))
                print(f"[scheduler] sleeping {wait}s until open")
                time.sleep(wait)
    except KeyboardInterrupt:
        kill_all(procs)
        print("scheduler shutdown")