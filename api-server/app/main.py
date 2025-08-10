# api-server/app/main.py
import os
import json
import asyncio
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from sqlalchemy import select, and_, desc

from . import models, reporting
from .database import async_engine, AsyncSessionLocal
from .telegram_bot import send_telegram_message

# ----------------------------
# Globals & config
# ----------------------------
ALMATY_TZ = ZoneInfo("Asia/Almaty")
OPEN_T = dtime(8, 30)
CLOSE_T = dtime(23, 59, 59)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REPORTS_DIR = "reports"

app = FastAPI(title="Vision Analytics API")
templates = Jinja2Templates(directory="app/templates")
scheduler = AsyncIOScheduler()

# Shared realtime state for SSE/dashboard
realtime_state = {
    "consolidated_counts": defaultdict(int),  # zone -> total across cameras
    "total_occupancy": 0,                     # excludes 'barista'
    "per_camera_data": {}                     # camera_id -> last payload
}

# ----------------------------
# Helpers
# ----------------------------
def day_key(prefix: str, dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(ALMATY_TZ)
    return f"{prefix}:{dt.strftime('%Y-%m-%d')}"

def is_business_time(dt: datetime) -> bool:
    # Optional helper if you ever want to gate logging by business hours
    t = dt.timetz()
    return OPEN_T <= t <= CLOSE_T

# ----------------------------
# Redis Listeners
# ----------------------------
async def data_listener(r: redis.Redis):
    """
    Subscribes to 'vision-data-events' and maintains:
      - per-camera payloads
      - consolidated zone counts
      - total occupancy (excluding 'barista')
    """
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-data-events")
    print("[api] Listening for vision-data-events...")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not message:
                await asyncio.sleep(0.01)
                continue

            data = json.loads(message["data"])
            cam_id = data.get("camera_id")
            if not cam_id:
                continue

            # store last payload per camera
            realtime_state["per_camera_data"][cam_id] = data

            # recompute consolidated counts
            consolidated = defaultdict(int)
            for cam_payload in realtime_state["per_camera_data"].values():
                zc = cam_payload.get("zone_counts", {})
                for zone, cnt in zc.items():
                    consolidated[zone] += int(cnt or 0)

            realtime_state["consolidated_counts"] = consolidated

            # total occupancy: sum everything except 'barista'
            total = 0
            for zone, cnt in consolidated.items():
                if zone != "barista":
                    total += int(cnt or 0)
            realtime_state["total_occupancy"] = total

        except Exception as e:
            print(f"[api] data_listener error: {e}")
            await asyncio.sleep(1.0)

async def db_writer_listener(channel_name: str, model_class, r: redis.Redis):
    """
    Generic Redis->DB writer for:
      - 'vision-zone-change-events' -> TransitionEvent
      - 'vision-alert-events'       -> AlertLog (+ Telegram)
    """
    pubsub = r.pubsub()
    await pubsub.subscribe(channel_name)
    print(f"[api] Listening for {channel_name}...")
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            data = json.loads(message["data"])
            if channel_name == "vision-alert-events":
                # Send Telegram
                try:
                    await send_telegram_message(data.get("message", ""))
                except Exception as te:
                    print(f"[api] Telegram send error: {te}")
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    db_row = model_class(**data)
                    session.add(db_row)
        except Exception as e:
            print(f"[api] db_writer_listener({channel_name}) error: {e}")

async def log_metrics_periodically():
    """
    Every minute: write a snapshot of consolidated occupancy to DB.
    This powers peaks & peak-period analytics in the nightly report.
    """
    print("[api] Periodic occupancy logger started.")
    while True:
        await asyncio.sleep(60)
        try:
            counts = realtime_state["consolidated_counts"]
            total_occupancy = int(realtime_state["total_occupancy"] or 0)
            queue_count = int(counts.get("queue", 0))
            hall_count = int(counts.get("hall", 0))

            async with AsyncSessionLocal() as session:
                async with session.begin():
                    row = models.OccupancyLog(
                        ts=datetime.now(ALMATY_TZ),
                        total_occupancy=total_occupancy,
                        queue_count=queue_count,
                        hall_count=hall_count
                    )
                    session.add(row)
        except Exception as e:
            print(f"[api] Periodic logging error: {e}")

# ----------------------------
# FastAPI lifecycle
# ----------------------------
@app.on_event("startup")
async def startup_event():
    # Create DB tables (safe to run repeatedly)
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    # Redis connection (async)
    r = redis.from_url(f"redis://{REDIS_HOST}")

    # Kick off background tasks
    asyncio.create_task(data_listener(r))
    asyncio.create_task(db_writer_listener("vision-zone-change-events", models.TransitionEvent, r))
    asyncio.create_task(db_writer_listener("vision-alert-events", models.AlertLog, r))
    asyncio.create_task(log_metrics_periodically())

    # Schedule daily report BEFORE midnight Almaty so Redis daily keys are still there
    scheduler.add_job(
        reporting.generate_daily_report,
        trigger=CronTrigger(hour=23, minute=59, timezone=ALMATY_TZ),
        id="daily_report_job",
        replace_existing=True
    )
    scheduler.start()
    print("[api] Startup complete. Scheduler running.")

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    # Your redesigned template should be placed at app/templates/index.html
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    """
    Server-Sent Events stream with:
      - realtime consolidated occupancy
      - KPIs (counts/averages/%) for today from Redis
    """
    r = redis.from_url(f"redis://{REDIS_HOST}")

    # Cache barista alert count for 30s to avoid DB spam
    last_alerts_check = 0.0
    cached_alerts = 0

    async def event_generator():
        nonlocal last_alerts_check, cached_alerts

        while True:
            if await request.is_disconnected():
                break

            # Base realtime snapshot
            payload = {
                "consolidated_counts": dict(realtime_state["consolidated_counts"]),
                "total_occupancy": int(realtime_state["total_occupancy"] or 0),
                "per_camera_data": realtime_state["per_camera_data"],  # optional; keep if useful
            }

            # Pull today's KPI counters from Redis
            now = datetime.now(ALMATY_TZ)
            keys = {
                "unique_guests": day_key("unique_guests", now),
                "pickup_valid":  day_key("pickup_valid",  now),
                "o2p":           day_key("conv:order_to_pickup", now),
                "p2h":           day_key("conv:pickup_to_hall",  now),
                "p2e":           day_key("conv:pickup_to_exit",  now),
                "sum_o2p":       day_key("sum:o2p_s", now),
                "cnt_o2p":       day_key("cnt:o2p",   now),
                "sum_p2h":       day_key("sum:p2h_s", now),
                "cnt_p2h":       day_key("cnt:p2h",   now),
                "sum_p2e":       day_key("sum:p2e_s", now),
                "cnt_p2e":       day_key("cnt:p2e",   now),
            }
            pipe = r.pipeline()
            for k in keys.values():
                pipe.get(k)
            vals = await pipe.execute()

            def to_int(x):
                try:
                    return int(x) if x is not None else 0
                except Exception:
                    return 0

            def to_float(x):
                try:
                    return float(x) if x is not None else 0.0
                except Exception:
                    return 0.0

            unique_guests = to_int(vals[0])
            pickup_valid  = to_int(vals[1])
            o2p = to_int(vals[2]);   p2h = to_int(vals[3]);   p2e = to_int(vals[4])
            sum_o2p = to_float(vals[5]); cnt_o2p = to_int(vals[6])
            sum_p2h = to_float(vals[7]); cnt_p2h = to_int(vals[8])
            sum_p2e = to_float(vals[9]); cnt_p2e = to_int(vals[10])

            avg_o2p = 0.0 if cnt_o2p == 0 else round(sum_o2p / cnt_o2p, 1)
            avg_p2h = 0.0 if cnt_p2h == 0 else round(sum_p2h / cnt_p2h, 1)
            avg_p2e = 0.0 if cnt_p2e == 0 else round(sum_p2e / cnt_p2e, 1)

            p2h_pct = 0.0 if pickup_valid == 0 else round(100.0 * p2h / pickup_valid, 1)
            p2e_pct = 0.0 if pickup_valid == 0 else round(100.0 * p2e / pickup_valid, 1)

            # Optional: live count of today's barista alerts (cached ~30s)
            # If you prefer not to hit DB here, set cached_alerts = 0 and remove this block.
            if (asyncio.get_event_loop().time() - last_alerts_check) > 30.0:
                try:
                    async with AsyncSessionLocal() as session:
                        start_today = now.replace(hour=OPEN_T.hour, minute=OPEN_T.minute, second=0, microsecond=0)
                        q = select(models.AlertLog).where(
                            and_(
                                models.AlertLog.timestamp >= start_today,
                                models.AlertLog.timestamp <= now,
                                models.AlertLog.alert_type == "STAFF_ABSENCE",
                            )
                        )
                        res = await session.execute(q)
                        cached_alerts = len(list(res.scalars()))
                except Exception:
                    cached_alerts = 0
                finally:
                    last_alerts_check = asyncio.get_event_loop().time()

            payload["kpis"] = {
                "unique_guests": unique_guests,
                "o2p": {"count": o2p, "avg_s": avg_o2p},
                "p2h": {"count": p2h, "avg_s": avg_p2h, "pct": p2h_pct},
                "p2e": {"count": p2e, "avg_s": avg_p2e, "pct": p2e_pct},
                "barista_alerts": cached_alerts,
            }

            yield {"event": "message", "data": json.dumps(payload)}
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

@app.get("/reports/latest")
async def get_latest_report():
    try:
        if not os.path.exists(REPORTS_DIR):
            return JSONResponse({"error": "Отчёты ещё не сформированы."}, status_code=404)
        files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".csv")]
        if not files:
            return JSONResponse({"error": "Отчёты ещё не сформированы."}, status_code=404)
        files.sort(reverse=True)
        return {"latest_report_filename": files[0]}
    except Exception as e:
        return JSONResponse({"error": f"Ошибка: {e}"}, status_code=500)

@app.get("/reports/download/{filename}")
async def download_report(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "Файл не найден."}, status_code=404)
    return FileResponse(path=path, media_type="text/csv", filename=filename)