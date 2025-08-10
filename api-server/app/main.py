import os, json, asyncio
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sse_starlette.sse import EventSourceResponse

from . import models, reporting
from .database import async_engine, AsyncSessionLocal
from sqlalchemy import select, func

ALMATY = ZoneInfo("Asia/Almaty")
OPEN_T  = time(8, 30)
CLOSE_T = time(23, 59, 59)

app = FastAPI(title="Vision Analytics API")
templates = Jinja2Templates(directory="app/templates")
scheduler = AsyncIOScheduler()

realtime_state = {
    "consolidated_counts": defaultdict(int),
    "total_occupancy": 0,
    "per_camera_data": {},
    "kpis": {
        "unique_guests": 0,
        "o2p": {"count": 0, "avg_s": 0.0},
        "p2h": {"count": 0, "avg_s": 0.0},
        "p2e": {"count": 0, "avg_s": 0.0},
        "barista_alerts": 0
    }
}

def day_str(dt: datetime | None = None) -> str:
    if dt is None: dt = datetime.now(ALMATY)
    return dt.strftime("%Y-%m-%d")

def redis_key(prefix: str, dt: datetime | None = None) -> str:
    return f"{prefix}:{day_str(dt)}"

def today_bounds_almaty():
    now = datetime.now(ALMATY)
    start = datetime.combine(now.date(), OPEN_T, tzinfo=ALMATY)
    end   = datetime.combine(now.date(), CLOSE_T, tzinfo=ALMATY)
    return start, end

async def data_listener(r: redis.Redis):
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-data-events")
    print("[api] Listening for vision-data-events…")
    while True:
        try:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if msg:
                data = json.loads(msg["data"])
                realtime_state["per_camera_data"][data["camera_id"]] = data

                consolidated = defaultdict(int)
                for cam_data in realtime_state["per_camera_data"].values():
                    for zone, count in cam_data.get("zone_counts", {}).items():
                        consolidated[zone] += count

                realtime_state["consolidated_counts"] = consolidated
                realtime_state["total_occupancy"] = sum(v for k,v in consolidated.items() if k != "barista")
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"[api] data_listener error: {e}"); await asyncio.sleep(2)

async def poll_kpis(r: redis.Redis):
    print("[api] KPI poller started")
    while True:
        try:
            k_unique = redis_key("unique_guests")
            k_o2p    = redis_key("conv:order_to_pickup")
            k_p2h    = redis_key("conv:pickup_to_hall")
            k_p2e    = redis_key("conv:pickup_to_exit")

            k_sum_o2p = redis_key("sum:o2p_s"); k_cnt_o2p = redis_key("cnt:o2p")
            k_sum_p2h = redis_key("sum:p2h_s"); k_cnt_p2h = redis_key("cnt:p2h")
            k_sum_p2e = redis_key("sum:p2e_s"); k_cnt_p2e = redis_key("cnt:p2e")

            pipe = r.pipeline()
            for k in [k_unique, k_o2p, k_p2h, k_p2e, k_sum_o2p, k_cnt_o2p, k_sum_p2h, k_cnt_p2h, k_sum_p2e, k_cnt_p2e]:
                pipe.get(k)
            vals = await pipe.execute()
            (v_unique, v_o2p, v_p2h, v_p2e, sum_o2p, cnt_o2p, sum_p2h, cnt_p2h, sum_p2e, cnt_p2e) = vals

            to_int   = lambda x: int(x) if x is not None else 0
            to_float = lambda x: float(x) if x is not None else 0.0

            unique_guests = to_int(v_unique)
            o2p_count = to_int(v_o2p); o2p_avg = to_float(sum_o2p)/max(1,to_int(cnt_o2p))
            p2h_count = to_int(v_p2h); p2h_avg = to_float(sum_p2h)/max(1,to_int(cnt_p2h))
            p2e_count = to_int(v_p2e); p2e_avg = to_float(sum_p2e)/max(1,to_int(cnt_p2e))

            start, end = today_bounds_almaty()
            async with AsyncSessionLocal() as sess:
                q = select(func.count(models.AlertLog.id)).where(
                    models.AlertLog.timestamp >= start.timestamp(),
                    models.AlertLog.timestamp <= end.timestamp(),
                    models.AlertLog.alert_type == "STAFF_ABSENCE"
                )
                res = await sess.execute(q)
                barista_alerts = int(res.scalar_one())

            realtime_state["kpis"] = {
                "unique_guests": unique_guests,
                "o2p": {"count": o2p_count, "avg_s": round(o2p_avg,1)},
                "p2h": {"count": p2h_count, "avg_s": round(p2h_avg,1)},
                "p2e": {"count": p2e_count, "avg_s": round(p2e_avg,1)},
                "barista_alerts": barista_alerts
            }
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[api] KPI poller error: {e}"); await asyncio.sleep(5)

async def db_listener(channel_name, model_class, r: redis.Redis):
    pubsub = r.pubsub()
    await pubsub.subscribe(channel_name)
    print(f"[api] Listening for {channel_name} → DB")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not message:
                await asyncio.sleep(0.1); continue
            data = json.loads(message["data"])
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    db_log = model_class(**data)
                    session.add(db_log)
        except Exception as e:
            print(f"[api] db_listener({channel_name}) error: {e}")
            await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)

    r = redis.from_url(f"redis://{os.getenv('REDIS_HOST','redis')}:6379/0")
    asyncio.create_task(data_listener(r))
    asyncio.create_task(db_listener("vision-zone-change-events", models.TransitionEvent, r))
    asyncio.create_task(db_listener("vision-alert-events",       models.AlertLog,        r))
    asyncio.create_task(poll_kpis(r))

    # generate previous-day CSV at 00:05 ALMT
    scheduler.add_job(reporting.generate_daily_report,
                      trigger=CronTrigger(hour=0, minute=5, timezone=ALMATY),
                      id="daily_report_job", replace_existing=True)
    scheduler.start()
    print("[api] Startup complete.")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected(): break
            yield {"event": "message", "data": json.dumps(realtime_state, default=str)}
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())

@app.get("/reports/latest")
async def get_latest_report():
    reports_dir = "app/reports"
    os.makedirs(reports_dir, exist_ok=True)
    files = [f for f in os.listdir(reports_dir) if f.endswith(".csv")]
    if not files:
        return {"error": "Нет доступных отчётов."}
    files.sort(reverse=True)
    return {"latest_report_filename": files[0]}

@app.get("/reports/download/{filename}")
async def download_report(filename: str):
    path = os.path.join("app/reports", filename)
    if not os.path.isfile(path):
        return {"error": "Файл не найден."}
    return FileResponse(path=path, media_type="text/csv", filename=filename)