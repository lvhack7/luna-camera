# vision-processor/vision_processor.py
import os, json, time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional

import cv2, numpy as np, redis, torch
from ultralytics import YOLO
import supervision as sv
from multiprocessing import Process
from threading import Thread
from queue import Queue, Empty

# ==================== GLOBAL SETTINGS ====================
ALMATY_TZ        = ZoneInfo("Asia/Almaty")
OPEN_T           = dtime(8, 30)
CLOSE_T          = dtime(23, 59, 59)

MODEL_NAME       = os.getenv("MODEL_NAME", "yolo11l.pt")
IMG_SIZE         = int(os.getenv("IMG_SIZE", 1280))

CONF_DETECT_LOW  = float(os.getenv("CONF_DETECT_LOW", 0.10))
IOU_NMS          = float(os.getenv("IOU_NMS", 0.30))
CONFIRM_THRESH   = float(os.getenv("CONFIRM_THRESH", 0.30))
KEEP_THRESH      = float(os.getenv("KEEP_THRESH", 0.15))

ENTER_SECONDS    = float(os.getenv("ENTER_SECONDS", 0.8))   # hysteresis enter (sec)
EXIT_SECONDS     = float(os.getenv("EXIT_SECONDS", 2.5))    # hysteresis exit (sec)
MIN_AGE_FRAMES   = int(os.getenv("MIN_AGE_FRAMES", 5))
COAST_FRAMES     = int(os.getenv("COAST_FRAMES", 60))
GRAY_STD_THRESH  = float(os.getenv("GRAY_STD_THRESH", 10.0))

# Conversions (seconds) - camera_201
ORDER_TO_PICKUP_S = int(os.getenv("ORDER_TO_PICKUP_S", 900))
PICKUP_TO_HALL_S  = int(os.getenv("PICKUP_TO_HALL_S", 900))
PICKUP_TO_EXIT_S  = int(os.getenv("PICKUP_TO_EXIT_S", 900))

# Unique guests: count when dwell in ORDER ≥ 10s
MIN_ORDER_DWELL_S = int(os.getenv("MIN_ORDER_DWELL_S", 10))

# Association / dwell (ID bridging via spatial proximity)
MAX_ASSOC_DIST    = int(os.getenv("MAX_ASSOC_DIST", 220))     # px for O→P bridge
PICKUP_DWELL_S    = float(os.getenv("PICKUP_DWELL_S", 3.0))   # require ≥3s in pickup

# Unique guest cross-ID dedupe (re-entries)
UNIQUE_COOLDOWN_S = int(os.getenv("UNIQUE_COOLDOWN_S", 1200)) # 20 min
UNIQUE_ASSOC_DIST = int(os.getenv("UNIQUE_ASSOC_DIST", 220))   # px

# Redis / channels
REDIS_HOST       = os.getenv("REDIS_HOST", "redis")
REDIS_PORT       = int(os.getenv("REDIS_PORT", 6379))
DATA_CHANNEL     = "vision-data-events"
ZONE_CHANGE_CH   = "vision-zone-change-events"
ALERT_CHANNEL    = "vision-alert-events"

# ==================== DEBUG SETTINGS ====================
DEBUG = os.getenv("VP_DEBUG", "0") == "1"
DEBUG_INTERVAL = float(os.getenv("VP_DEBUG_INTERVAL", 5))         # seconds between console logs
SNAPSHOT_DIR = os.getenv("VP_SNAPSHOT_DIR", "/app/debug")          # where to store annotated frames
SNAPSHOT_EVERY_S = float(os.getenv("VP_SNAPSHOT_EVERY_S", 0))      # 0 disables snapshots
PUBLISH_DEBUG = os.getenv("VP_PUBLISH_DEBUG", "0") == "1"          # publish metrics to Redis
DEBUG_CH = "vision-debug-events"

def log(*a, **kw):
    if DEBUG:
        print(*a, **kw, flush=True)

# --- small utilities for FPS/latency smoothing ---
from collections import deque
class RateCounter:
    def __init__(self, horizon=300):
        self.t = deque(maxlen=horizon)
    def tick(self):
        self.t.append(time.time())
    def fps(self) -> float:
        if len(self.t) < 2: return 0.0
        dt = self.t[-1] - self.t[0]
        return (len(self.t) - 1) / dt if dt > 0 else 0.0

class EMA:
    def __init__(self, alpha=0.2):
        self.v = None
        self.alpha = alpha
    def update(self, x):
        self.v = x if self.v is None else (self.alpha * x + (1 - self.alpha) * self.v)
        return self.v

def day_key(prefix: str, dt: Optional[datetime] = None) -> str:
    if dt is None: dt = datetime.now(ALMATY_TZ)
    return f"{prefix}:{dt.strftime('%Y-%m-%d')}"

def seconds_until_midnight(dt: Optional[datetime] = None) -> int:
    if dt is None: dt = datetime.now(ALMATY_TZ)
    nxt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60, int((nxt - dt).total_seconds()))

# ==================== CAPTURE ====================
class VideoCaptureThreaded:
    """
    Threaded RTSP reader with:
      - drop-old queue (maxsize=1) to always process newest frame
      - capture FPS measurement
    """
    def __init__(self, src: str, name: str):
        self.src = src
        self.q = Queue(maxsize=1)
        self.is_running = True
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.t = Thread(target=self._reader, name=name, daemon=True)
        # debug counters
        self.cap_rate = RateCounter(horizon=300)
        self.t0 = time.time()
        self.frames_total = 0
        self.t.start()

    def _reader(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(2.0)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            self.cap_rate.tick()
            self.frames_total += 1
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

    # diagnostics
    def fps(self) -> float:
        return self.cap_rate.fps()
    def uptime_s(self) -> float:
        return time.time() - self.t0
    def frames(self) -> int:
        return self.frames_total

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def is_corrupted(frame: np.ndarray) -> bool:
    return frame is None or frame.size == 0 or frame.std() < GRAY_STD_THRESH

# ==================== PER-CAMERA WORKER ====================
def process_camera(camera_id: str, config: dict):
    # Track confirmation & hysteresis state
    track_confirmed: Dict[int, bool] = {}
    zone_state = defaultdict(lambda: defaultdict(lambda: {"first_in": None, "last_in": None, "is_member": False}))

    # Journeys + association buffers (camera_201 conversions + unique guests)
    journey: Dict[int, dict] = defaultdict(dict)
    recent_orders:  deque = deque(maxlen=2000)   # {"ts","cx","cy","used"}
    recent_pickups: deque = deque(maxlen=2000)   # {"ts","cx","cy","used"}
    recent_unique:  deque = deque(maxlen=2000)   # {"ts","cx","cy"}
    last_center: Dict[int, Tuple[int,int]] = {}

    rtsp_url = os.getenv(config.get("rtsp_url_env", ""), "")
    if not rtsp_url:
        print(f"[{camera_id}] FATAL: RTSP URL not set", flush=True); return

    cap = VideoCaptureThreaded(rtsp_url, name=f"RTSP-{camera_id}")
    time.sleep(1.5)

    device = get_device()
    model  = YOLO(MODEL_NAME)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try: r.ping()
    except Exception as e: print(f"[{camera_id}] Redis error: {e}"); return

    tracker = sv.ByteTrack(track_thresh=0.05, track_buffer=COAST_FRAMES+10, match_thresh=0.8, frame_rate=20)

    zones_cfg = config["zones"]
    names     = [z["name"] for z in zones_cfg]
    names_lc  = [n.lower() for n in names]

    # KEEP THIS EXACT LINE:
    zones     = [sv.PolygonZone(polygon=np.array(z["polygon"], np.int32), frame_resolution_wh=(1920, 1080), triggering_position=sv.Position.CENTER) for z in zones_cfg]

    # Barista alerting (camera_601)
    last_barista_seen = time.time()
    alert_sent = False
    barista_alert_cfg = config.get("alert_config")

    # Live counts cache
    last_payload_counts = {"hall": 0, "queue": 0, "barista": 0}

    # Local age/miss
    age_frames: Dict[int, int]  = defaultdict(int)
    miss_frames: Dict[int, int] = defaultdict(int)

    log(f"[{camera_id}] Zones: {names}")

    # --- performance meters ---
    loop_rate = RateCounter(horizon=300)
    ema_infer_ms = EMA(alpha=0.2)
    ema_track_ms = EMA(alpha=0.2)
    ema_zone_ms  = EMA(alpha=0.2)
    last_debug_print = 0.0
    last_snapshot = 0.0

    # helpers
    def _euclid(a: Optional[Tuple[int,int]], b: Optional[Tuple[int,int]]) -> float:
        if not a or not b: return 1e9
        ax, ay = a; bx, by = b
        dx = ax - bx; dy = ay - by
        return (dx*dx + dy*dy)**0.5

    def match_from_buffer(buf: deque, ts: float, center: Tuple[int,int], window_s: int) -> Optional[dict]:
        best, best_d = None, 1e9
        for ev in reversed(buf):
            if ev.get("used"): continue
            if ts - ev["ts"] > window_s: break
            d = _euclid(center, (ev["cx"], ev["cy"]))
            if d < best_d:
                best, best_d = ev, d
        if best and best_d <= MAX_ASSOC_DIST:
            best["used"] = True
            return best
        return None

    def exists_close(buf: deque, ts: float, center: Tuple[int,int], window_s: int, dist_px: int) -> bool:
        if not center: return False
        for ev in reversed(buf):
            if ts - ev["ts"] > window_s: break
            if _euclid(center, (ev["cx"], ev["cy"])) <= dist_px:
                return True
        return False

    while True:
        t_loop0 = time.time()
        frame = cap.read(); now = time.time()
        if is_corrupted(frame):
            r.publish(DATA_CHANNEL, json.dumps({"timestamp": now, "camera_id": camera_id, "zone_counts": last_payload_counts}))
            continue

        # --- DETECTION ---
        t0 = time.time()
        with torch.no_grad():
            res = model(frame, imgsz=IMG_SIZE, conf=CONF_DETECT_LOW, iou=IOU_NMS, classes=[0], device=device, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000.0
        infer_ms_smooth = ema_infer_ms.update(infer_ms)

        # --- TRACKING ---
        t1 = time.time()
        dets = sv.Detections.from_ultralytics(res)
        tracks = tracker.update_with_detections(dets)
        track_ms_smooth = ema_track_ms.update((time.time() - t1) * 1000.0)

        # Age/miss
        current_ids = set(tracks.tracker_id) if tracks.tracker_id is not None else set()
        for tid in current_ids:
            tid = int(tid); age_frames[tid] += 1; miss_frames[tid] = 0
        for tid in list(age_frames.keys()):
            if tid not in current_ids:
                miss_frames[tid] += 1
                if miss_frames[tid] > COAST_FRAMES:
                    age_frames.pop(tid, None); miss_frames.pop(tid, None)
                    zone_state.pop(tid, None); journey.pop(tid, None); track_confirmed.pop(tid, None)

        # Confirm/keep
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

        # Centers for stable detections
        last_center_local: Dict[int, Tuple[int,int]] = {}
        if len(stable) > 0 and stable.tracker_id is not None:
            for i in range(len(stable)):
                x1, y1, x2, y2 = map(float, stable.xyxy[i])
                cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                tid_i = int(stable.tracker_id[i])
                last_center[tid_i] = (cx, cy)
                last_center_local[tid_i] = (cx, cy)

        # --- ZONES & HYSTERESIS ---
        t2 = time.time()
        zone_masks: List[np.ndarray] = [np.array([], dtype=bool) for _ in zones]
        if len(stable) > 0 and stable.tracker_id is not None:
            for i, zone in enumerate(zones):
                zone_masks[i] = zone.trigger(detections=stable)

        enter_events: List[Tuple[str, int, float, Optional[Tuple[int,int]]]] = []
        if len(stable) > 0 and stable.tracker_id is not None:
            for di in range(len(stable)):
                tid = int(stable.tracker_id[di])
                for pi in range(len(zones)):
                    inside = bool(zone_masks[pi][di]) if zone_masks[pi].size > di else False
                    st     = zone_state[tid][pi]
                    zname  = names_lc[pi]
                    if inside:
                        if st["first_in"] is None: st["first_in"] = now
                        st["last_in"] = now
                        if not st["is_member"] and (now - st["first_in"]) >= ENTER_SECONDS:
                            st["is_member"] = True
                            center = last_center_local.get(tid, last_center.get(tid))
                            enter_events.append((zname, tid, now, center))
                            r.publish(ZONE_CHANGE_CH, json.dumps({
                                "timestamp": now, "camera_id": camera_id, "tracker_id": tid, "event":"enter", "zone": zname
                            }))
                    else:
                        st["first_in"] = None
                        if st["is_member"] and st["last_in"] and (now - st["last_in"]) >= EXIT_SECONDS:
                            st["is_member"] = False
                            r.publish(ZONE_CHANGE_CH, json.dumps({
                                "timestamp": now, "camera_id": camera_id, "tracker_id": tid, "event":"exit", "zone": zname
                            }))
        zone_ms_smooth = ema_zone_ms.update((time.time() - t2) * 1000.0)

        # ===== UNIQUE GUESTS: dwell in ORDER ≥ MIN_ORDER_DWELL_S with cross-ID dedupe (camera_201) =====
        if camera_id == "camera_201":
            k_unique = day_key("unique_guests")
            ttl = seconds_until_midnight()
            for tid, per_poly in zone_state.items():
                order_idxs = [i for i,n in enumerate(names_lc) if n == "order"]
                if not order_idxs: continue
                in_order = False; first_in = None
                for oi in order_idxs:
                    st = per_poly.get(oi)
                    if st and st["is_member"] and st["first_in"] is not None:
                        in_order = True
                        first_in = first_in if first_in is not None else st["first_in"]
                if not in_order or first_in is None: continue

                dwell = now - first_in
                if dwell < MIN_ORDER_DWELL_S: continue

                j = journey[tid]
                if j.get("unique_counted"): continue

                ded = day_key(f"seen:unique_guest:{camera_id}:{tid}")
                already_tid = not r.set(ded, "1", ex=ttl, nx=True)

                center_now = last_center.get(int(tid))
                already_nearby = exists_close(recent_unique, now, center_now, UNIQUE_COOLDOWN_S, UNIQUE_ASSOC_DIST)

                if not (already_tid or already_nearby):
                    r.incr(k_unique); r.expire(k_unique, ttl)
                    log(f"[{camera_id}] UNIQUE ++ tid={tid} dwell={dwell:.1f}s")
                    if center_now:
                        recent_unique.append({"ts": now, "cx": center_now[0], "cy": center_now[1]})
                j["unique_counted"] = True

        # ===== CONVERSIONS with ID bridging & pickup dwell gate (camera_201) =====
        if camera_id == "camera_201" and enter_events:
            k_conv_o2p      = day_key("conv:order_to_pickup")
            k_conv_p2h      = day_key("conv:pickup_to_hall")
            k_conv_p2e      = day_key("conv:pickup_to_exit")
            k_sum_o2p       = day_key("sum:o2p_s");  k_cnt_o2p = day_key("cnt:o2p")
            k_sum_p2h       = day_key("sum:p2h_s");  k_cnt_p2h = day_key("cnt:p2h")
            k_sum_p2e       = day_key("sum:p2e_s");  k_cnt_p2e = day_key("cnt:p2e")
            k_pickup_valid  = day_key("pickup_valid")   # denominator for conversion %
            ttl = seconds_until_midnight()

            # A) cache raw order enters
            for zname, tid, ts, center in enter_events:
                j = journey.setdefault(tid, {})
                if zname == "order":
                    recent_orders.append({"ts": ts, "cx": (center or (0,0))[0], "cy": (center or (0,0))[1], "used": False})
                    j.setdefault("order_enter_ts", ts); j["last_ts"] = ts

            # B) when pickup dwell ≥ PICKUP_DWELL_S, count pickup_valid once and emit order→pickup
            for tid_k, per_poly in zone_state.items():
                in_pick = False; first_in = None
                for pi, nm in enumerate(names_lc):
                    if nm != "pickup": continue
                    st = per_poly.get(pi)
                    if st and st["is_member"] and st["first_in"] is not None:
                        in_pick = True
                        first_in = first_in or st["first_in"]
                if not in_pick or first_in is None:
                    continue

                dwell = now - first_in
                if dwell < PICKUP_DWELL_S:
                    continue

                j = journey.setdefault(tid_k, {})

                # Denominator: pickup_valid (each track once)
                if not j.get("pickup_valid_counted"):
                    r.incr(k_pickup_valid); r.expire(k_pickup_valid, ttl)
                    j["pickup_valid_counted"] = True

                # order→pickup once (same ID, else bridge)
                if not j.get("counted_o2p"):
                    if "order_enter_ts" in j:
                        dt = now - j["order_enter_ts"]
                        if 0 <= dt <= ORDER_TO_PICKUP_S:
                            r.incr(k_conv_o2p); r.expire(k_conv_o2p, ttl)
                            r.incrbyfloat(k_sum_o2p, float(dt)); r.incr(k_cnt_o2p)
                            r.expire(k_sum_o2p, ttl); r.expire(k_cnt_o2p, ttl)
                            j["counted_o2p"] = True
                    if not j.get("counted_o2p"):
                        cnow = last_center.get(int(tid_k))
                        ev = match_from_buffer(recent_orders, now, cnow, ORDER_TO_PICKUP_S)
                        if ev:
                            dt = now - ev["ts"]
                            r.incr(k_conv_o2p); r.expire(k_conv_o2p, ttl)
                            r.incrbyfloat(k_sum_o2p, float(dt)); r.incr(k_cnt_o2p)
                            r.expire(k_sum_o2p, ttl); r.expire(k_cnt_o2p, ttl)
                            j["counted_o2p"] = True

                # buffer pickup point for next hops
                c = last_center.get(int(tid_k))
                if c and not any((now - ev["ts"] < 2 and ((ev["cx"]-c[0])**2 + (ev["cy"]-c[1])**2) < 400) for ev in recent_pickups):
                    recent_pickups.append({"ts": now, "cx": c[0], "cy": c[1], "used": False})

            # C) on hall/exit enters, match a recent pickup within window
            for zname, tid, ts, center in enter_events:
                if zname in ("hall1","hall2"):
                    ev = match_from_buffer(recent_pickups, ts, center, PICKUP_TO_HALL_S)
                    if ev:
                        dt = ts - ev["ts"]
                        r.incr(k_conv_p2h); r.expire(k_conv_p2h, ttl)
                        r.incrbyfloat(k_sum_p2h, float(dt)); r.incr(k_cnt_p2h)
                        r.expire(k_sum_p2h, ttl); r.expire(k_cnt_p2h, ttl)
                elif zname == "exit":
                    ev = match_from_buffer(recent_pickups, ts, center, PICKUP_TO_EXIT_S)
                    if ev:
                        dt = ts - ev["ts"]
                        r.incr(k_conv_p2e); r.expire(k_conv_p2e, ttl)
                        r.incrbyfloat(k_sum_p2e, float(dt)); r.incr(k_cnt_p2e)
                        r.expire(k_sum_p2e, ttl); r.expire(k_cnt_p2e, ttl)

        # ===== LIVE OCCUPANCY (for dashboard) =====
        def count_group(label: str) -> int:
            idxs = [i for i,n in enumerate(names_lc) if n == label]
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

        # publish live
        r.publish(DATA_CHANNEL, json.dumps({"timestamp": now, "camera_id": camera_id, "zone_counts": occ}))
        last_payload_counts = occ

        # --- LOOP FPS & DEBUG OUTPUT ---
        loop_rate.tick()
        if DEBUG and (now - last_debug_print) >= DEBUG_INTERVAL:
            last_debug_print = now
            cap_fps = cap.fps()
            print(
                f"[{camera_id}] cap_fps={cap_fps:.1f} | infer={ema_infer_ms.v or 0.0:.1f}ms "
                f"({(1000.0/(ema_infer_ms.v or 1e9)):.1f} fps) | track={ema_track_ms.v or 0.0:.1f}ms | "
                f"zones={ema_zone_ms.v or 0.0:.1f}ms | loop_fps={loop_rate.fps():.1f} | "
                f"hall={occ.get('hall',0)} queue={occ.get('queue',0)}",
                flush=True
            )
            if PUBLISH_DEBUG:
                dbg = {
                    "ts": now,
                    "camera_id": camera_id,
                    "capture_fps": round(cap_fps, 2),
                    "infer_ms": round(ema_infer_ms.v or 0.0, 1),
                    "infer_fps": round((1000.0 / (ema_infer_ms.v or 1e9)), 1),
                    "track_ms": round(ema_track_ms.v or 0.0, 1),
                    "zones_ms": round(ema_zone_ms.v or 0.0, 1),
                    "loop_fps": round(loop_rate.fps(), 1),
                    "occupancy": occ
                }
                r.publish(DEBUG_CH, json.dumps(dbg))

        # --- OPTIONAL SNAPSHOTS ---
        if SNAPSHOT_EVERY_S > 0 and (now - last_snapshot) >= SNAPSHOT_EVERY_S:
            last_snapshot = now
            try:
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                snap = frame.copy()

                # draw polygons
                for zc in config["zones"]:
                    poly = np.array(zc["polygon"], np.int32)
                    cv2.polylines(snap, [poly], isClosed=True, color=(0, 165, 255), thickness=2)
                    M = poly.mean(axis=0).astype(int)
                    cv2.putText(snap, zc["name"], tuple(M), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # draw stable tracks
                if len(stable) > 0 and stable.tracker_id is not None:
                    for i in range(len(stable)):
                        x1, y1, x2, y2 = map(int, stable.xyxy[i])
                        tid = int(stable.tracker_id[i])
                        cv2.rectangle(snap, (x1, y1), (x2, y2), (60, 180, 75), 2)
                        cv2.putText(snap, f"ID:{tid}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,180,75), 2)
                        cxcy = last_center.get(tid, ((x1+x2)//2, (y1+y2)//2))
                        cv2.circle(snap, (int(cxcy[0]), int(cxcy[1])), 3, (255, 255, 255), -1)

                # overlay perf text
                overlay = [
                    f"cap_fps: {cap.fps():.1f}",
                    f"infer: {(ema_infer_ms.v or 0.0):.1f} ms ({(1000.0/(ema_infer_ms.v or 1e9)):.1f} fps)",
                    f"track: {(ema_track_ms.v or 0.0):.1f} ms",
                    f"zones: {(ema_zone_ms.v or 0.0):.1f} ms",
                    f"loop_fps: {loop_rate.fps():.1f}",
                    f"hall={occ.get('hall',0)} queue={occ.get('queue',0)}"
                ]
                y = 24
                for line in overlay:
                    cv2.putText(snap, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 3)
                    cv2.putText(snap, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    y += 24

                out_path = os.path.join(SNAPSHOT_DIR, f"{camera_id}_latest.jpg")
                cv2.imwrite(out_path, snap)
            except Exception as e:
                log(f"[{camera_id}] snapshot error: {e}")

# ==================== SCHEDULER ====================
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

def spawn_all(full_cfg: dict) -> List[Process]:
    procs: List[Process] = []
    for cam_id, cam_cfg in full_cfg.items():
        p = Process(target=process_camera, args=(cam_id, cam_cfg), daemon=True)
        p.start(); procs.append(p)
        print(f"[scheduler] started {cam_id} pid={p.pid}")
    return procs

def kill_all(procs: List[Process]):
    for p in procs:
        if p.is_alive(): p.terminate()
    for p in procs:
        if p.is_alive(): p.join(timeout=5)
    print("[scheduler] all workers stopped")

if __name__ == "__main__":
    with open("configs/zones.json", "r") as f:
        full_config = json.load(f)

    procs: List[Process] = []
    try:
        while True:
            now = datetime.now(ALMATY_TZ)
            if within_business_hours(now):
                if not procs:
                    procs = spawn_all(full_config)
                time.sleep(min(30, seconds_until(next_close(now), now)))
            else:
                if procs:
                    kill_all(procs); procs = []
                time.sleep(min(60, seconds_until(next_open(now), now)))
    except KeyboardInterrupt:
        kill_all(procs)
        print("scheduler shutdown")