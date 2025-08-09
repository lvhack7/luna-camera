# app/main.py
import os
import json
import asyncio
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sse_starlette.sse import EventSourceResponse

from . import models, reporting
from .database import async_engine, AsyncSessionLocal
from .telegram_bot import send_telegram_message

# ------------------ Settings ------------------
ALMATY_TZ = ZoneInfo("Asia/Almaty")
OPEN_T = dtime(8, 30)     # 08:30
CLOSE_T = dtime(23, 59, 59)

REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}"
REPORTS_DIR = "reports"

# Metric keys emitted by the vision-processor (per local calendar day)
R_KEYS = [
    "order_unique",
    "pickup_unique",
    "conv:order_to_pickup",
    "conv:pickup_to_hall",
    "conv:pickup_to_exit",
    "sum:o2p_s", "cnt:o2p",
    "sum:p2h_s", "cnt:p2h",
    "sum:p2e_s", "cnt:p2e",
    "alerts:barista"
]

# Runtime state for SSE
realtime_state = {
    "consolidated_counts": defaultdict(int),  # live union of zone counts from cameras
    "total_occupancy": 0,                     # sum of non-barista zones
    "per_camera_data": {},                    # last payload per camera
    "kpis": {}                                # live conversions etc.
}

app = FastAPI(title="Vision Analytics API")
templates = Jinja2Templates(directory="app/templates")
scheduler = AsyncIOScheduler(timezone=ALMATY_TZ)

# ------------------ Helpers ------------------
def today_str_tz(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(ALMATY_TZ)
    return dt.strftime("%Y-%m-%d")

def day_key(prefix: str, dt: datetime | None = None) -> str:
    return f"{prefix}:{today_str_tz(dt)}"

def is_within_business_hours(now: datetime) -> bool:
    t = now.timetz()
    return OPEN_T <= t <= CLOSE_T

async def redis_mget_map(r: redis.Redis, keys: list[str]) -> dict[str, int | float]:
    values = await r.mget([day_key(k) for k in keys])
    out = {}
    for k, v in zip(keys, values):
        if v is None:
            out[k] = 0
        else:
            try:
                # numbers may be float (sums) or int (counters)
                out[k] = float(v) if k.startswith("sum:") else int(v)
            except Exception:
                out[k] = 0
    return out

# ------------------ Live occupancy listener ------------------
async def data_listener(r: redis.Redis):
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-data-events")
    print("Listening for data events...")
    while True:
        try:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg:
                data = json.loads(msg["data"])
                # 1) keep per-camera
                realtime_state["per_camera_data"][data["camera_id"]] = data

                # 2) recompute consolidated zone counts (sum by zone name)
                consolidated = defaultdict(int)
                for cam_data in realtime_state["per_camera_data"].values():
                    for zone, count in cam_data.get("zone_counts", {}).items():
                        consolidated[zone] += count
                realtime_state["consolidated_counts"] = consolidated

                # 3) total occupancy = everyone except barista
                realtime_state["total_occupancy"] = sum(
                    v for k, v in consolidated.items() if k != "barista"
                )
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"data_listener error: {e}")
            await asyncio.sleep(3)

# ------------------ DB writers ------------------
async def db_writer_listener(channel_name: str, model_class, r: redis.Redis):
    pubsub = r.pubsub()
    await pubsub.subscribe(channel_name)
    print(f"Listening for {channel_name} events...")
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            data = json.loads(message["data"])
            # Extra: count barista alerts in Redis for quick KPIs
            if channel_name == "vision-alert-events":
                await send_telegram_message(data["message"])
                # increment per-day barista alerts
                key = day_key("alerts:barista")
                await r.incr(key)
                await r.expire(key, 48 * 3600)

            async with AsyncSessionLocal() as session:
                async with session.begin():
                    session.add(model_class(**data))
        except Exception as e:
            print(f"DB writer error on {channel_name}: {e}")

# ------------------ Baseline at open (08:30) ------------------
async def snapshot_open_baseline(r: redis.Redis):
    """At 08:30 local, snapshot baselines so KPI = current - baseline within business day."""
    now = datetime.now(ALMATY_TZ)
    bkey = f"baseline:{today_str_tz(now)}"
    # Build {field: value}
    current = await redis_mget_map(r, R_KEYS)
    payload = {k: current.get(k, 0) for k in R_KEYS}
    if payload:
        await r.hset(bkey, mapping={k: str(v) for k, v in payload.items()})
        await r.expire(bkey, 3 * 24 * 3600)
    print(f"[baseline] snapshotted at {now.isoformat()} -> {bkey}: {payload}")

# ------------------ KPI poller (every 2s) ------------------
async def kpi_poller(r: redis.Redis):
    print("Starting KPI poller...")
    while True:
        try:
            now = datetime.now(ALMATY_TZ)
            # Fetch current values + baseline
            current = await redis_mget_map(r, R_KEYS)
            bkey = f"baseline:{today_str_tz(now)}"
            h = await r.hgetall(bkey)  # bytes
            baseline = {}
            for k in R_KEYS:
                v = h.get(k.encode()) if h else None
                baseline[k] = float(v.decode()) if v and k.startswith("sum:") else int(v.decode()) if v else 0

            # Use deltas within business window, otherwise show zeros
            if is_within_business_hours(now):
                delta = {k: (current[k] - baseline.get(k, 0)) for k in R_KEYS}
            else:
                delta = {k: 0 for k in R_KEYS}

            # Safely compute rates and averages
            def rate(num, den): return (num / den) if den > 0 else 0.0
            def avg(sum_s, cnt): return (sum_s / cnt) if cnt > 0 else 0.0

            kpis = {
                "order_to_pickup": {
                    "num": delta["conv:order_to_pickup"],
                    "den": delta["order_unique"],
                    "rate": rate(delta["conv:order_to_pickup"], delta["order_unique"]),
                    "avg_seconds": avg(delta["sum:o2p_s"], delta["cnt:o2p"])
                },
                "pickup_to_hall": {
                    "num": delta["conv:pickup_to_hall"],
                    "den": delta["pickup_unique"],
                    "rate": rate(delta["conv:pickup_to_hall"], delta["pickup_unique"]),
                    "avg_seconds": avg(delta["sum:p2h_s"], delta["cnt:p2h"])
                },
                "pickup_to_exit": {
                    "num": delta["conv:pickup_to_exit"],
                    "den": delta["pickup_unique"],
                    "rate": rate(delta["conv:pickup_to_exit"], delta["pickup_unique"]),
                    "avg_seconds": avg(delta["sum:p2e_s"], delta["cnt:p2e"])
                },
                "barista_alerts": int(delta["alerts:barista"]),
            }

            # Optional “customers who ordered so far” as a proxy for visitors
            kpis["orders_today"] = int(delta["order_unique"])

            realtime_state["kpis"] = kpis
        except Exception as e:
            print(f"kpi_poller error: {e}")
        await asyncio.sleep(2)

# ------------------ Periodic occupancy logging ------------------
async def log_metrics_periodically():
    """Persist total occupancy + zone slices every minute for peak analysis (local time)."""
    print("Periodic occupancy logging started.")
    while True:
        await asyncio.sleep(60)
        try:
            now = datetime.now(ALMATY_TZ)
            if not is_within_business_hours(now):
                continue
            counts = realtime_state["consolidated_counts"]
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    db_log = models.OccupancyLog(
                        ts=now,  # ensure your model has a ts column; otherwise store epoch + tz
                        total_occupancy=realtime_state["total_occupancy"],
                        queue_count=counts.get("queue", 0),
                        hall_count=counts.get("hall", 0),
                    )
                    session.add(db_log)
        except Exception as e:
            print(f"Periodic logging error: {e}")

# ------------------ Startup ------------------
@app.on_event("startup")
async def startup_event():
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    r = redis.from_url(REDIS_URL, decode_responses=False)

    # Background tasks
    asyncio.create_task(data_listener(r))
    asyncio.create_task(db_writer_listener("vision-zone-change-events", models.TransitionEvent, r))
    asyncio.create_task(db_writer_listener("vision-alert-events", models.AlertLog, r))
    asyncio.create_task(kpi_poller(r))
    asyncio.create_task(log_metrics_periodically())

    # Schedules in Asia/Almaty
    # 1) Open-day baseline at 08:30
    scheduler.add_job(snapshot_open_baseline, trigger=CronTrigger(hour=8, minute=30, timezone=ALMATY_TZ), args=[r], id="baseline_open", replace_existing=True)
    # 2) Daily CSV at 00:05 next day (covers previous business day 08:30–24:00)
    scheduler.add_job(reporting.generate_daily_report, trigger=CronTrigger(hour=0, minute=5, timezone=ALMATY_TZ),
                      args=[AsyncSessionLocal, r, REPORTS_DIR, ALMATY_TZ, OPEN_T, CLOSE_T],
                      id="daily_report_job", replace_existing=True)
    scheduler.start()
    print("API startup complete.")

# ------------------ HTTP endpoints ------------------
@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            yield {"event": "message", "data": json.dumps(dict(realtime_state))}
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())

@app.get("/reports/latest")
async def get_latest_report_meta():
    if not os.path.exists(REPORTS_DIR) or not os.listdir(REPORTS_DIR):
        return {"error": "Отчётов пока нет."}
    latest = sorted(os.listdir(REPORTS_DIR), reverse=True)[0]
    return {"latest_report_filename": latest}

@app.get("/reports/download/{filename}")
async def download_report(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return {"error": "Файл не найден."}
    return FileResponse(path=path, media_type="text/csv", filename=filename)

# Optional: on-demand report for a specific date (YYYY-MM-DD)
@app.get("/reports/daily")
async def on_demand_daily(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    await reporting.generate_daily_report(AsyncSessionLocal, redis.from_url(REDIS_URL), REPORTS_DIR, ALMATY_TZ, OPEN_T, CLOSE_T, date_override=date)
    return {"ok": True}