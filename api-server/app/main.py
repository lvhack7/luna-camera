import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from collections import defaultdict
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sse_starlette.sse import EventSourceResponse

from . import models, reporting
from .database import async_engine, AsyncSessionLocal
from .telegram_bot import send_telegram_message

realtime_state = {"consolidated_counts": defaultdict(int), "total_occupancy": 0, "per_camera_data": {}}
app = FastAPI(title="Vision Analytics API")
templates = Jinja2Templates(directory="app/templates")
scheduler = AsyncIOScheduler()

def _day_key(prefix: str) -> str:
    # today in server’s local time; change to UTC if you prefer
    from time import localtime, strftime
    return f"{prefix}:{strftime('%Y-%m-%d', localtime())}"

def _parse_int(x) -> int:
    if x is None: return 0
    if isinstance(x, (bytes, bytearray)): 
        try: return int(x.decode())
        except: return 0
    try: return int(x)
    except: return 0


async def kpi_poller(r: redis.Redis):
    """Poll daily KPI counters from Redis and merge into realtime_state."""
    print("Starting KPI poller...")
    while True:
        try:
            # keys for today
            k_vis  = _day_key("visitors:count")
            k_e2q  = _day_key("conv:entry_to_queue")
            k_q2h  = _day_key("conv:queue_to_hall")
            k_ent  = _day_key("entry_count")
            k_qent = _day_key("queue_entries")

            # fetch
            vis, e2q, q2h, ent, qent = await r.mget(k_vis, k_e2q, k_q2h, k_ent, k_qent)

            vis  = _parse_int(vis)
            e2q  = _parse_int(e2q)
            q2h  = _parse_int(q2h)
            ent  = _parse_int(ent)
            qent = _parse_int(qent)

            # safe denominators
            e2q_rate = (e2q / ent) if ent > 0 else 0.0
            q2h_rate = (q2h / qent) if qent > 0 else 0.0

            # store in realtime_state
            realtime_state["kpis"] = {
                "unique_visitors": vis,
                "entry_to_queue": {"num": e2q, "den": ent, "rate": e2q_rate},
                "queue_to_hall":  {"num": q2h, "den": qent, "rate": q2h_rate},
            }
        except Exception as e:
            print(f"KPI poller error: {e}")

        await asyncio.sleep(2)  # update every 2s


async def data_listener(r: redis.Redis):
    """Listens for occupancy data and aggregates it for the live dashboard."""
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-data-events")
    print("Listening for data events...")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                data = json.loads(message['data'])
                realtime_state['per_camera_data'][data['camera_id']] = data
                
                consolidated_counts = defaultdict(int)
                for cam_data in realtime_state['per_camera_data'].values():
                    for zone, count in cam_data.get('zone_counts', {}).items():
                        consolidated_counts[zone] += count
                
                realtime_state['consolidated_counts'] = consolidated_counts
                realtime_state['total_occupancy'] = sum(v for k, v in consolidated_counts.items() if k != 'barista')
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Data listener error: {e}"); await asyncio.sleep(5)

async def db_writer_listener(channel_name, model_class, r: redis.Redis):
    """Generic listener to write events from a Redis channel to a DB table."""
    pubsub = r.pubsub()
    await pubsub.subscribe(channel_name)
    print(f"Listening for {channel_name} events to log to DB...")
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                if channel_name == "vision-alert-events": # Also send Telegram alert
                    await send_telegram_message(data['message'])
                
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        db_log = model_class(**data)
                        session.add(db_log)
            except Exception as e:
                print(f"Error processing {channel_name}: {e}")

async def log_metrics_periodically():
    """Logs the current aggregated occupancy metrics to the database every minute."""
    print("Periodic occupancy logging task started.")
    while True:
        await asyncio.sleep(60)
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    counts = realtime_state['consolidated_counts']
                    db_log = models.OccupancyLog(
                        total_occupancy=realtime_state['total_occupancy'],
                        queue_count=counts.get('queue', 0),
                        hall_count=counts.get('hall', 0)
                    )
                    session.add(db_log)
        except Exception as e:
            print(f"Periodic logging error: {e}")

@app.on_event("startup")
async def startup_event():
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    
    r = redis.from_url(f"redis://{os.getenv('REDIS_HOST')}")
    asyncio.create_task(data_listener(r))
    asyncio.create_task(kpi_poller(r))
    asyncio.create_task(db_writer_listener("vision-zone-change-events", models.TransitionEvent, r))
    asyncio.create_task(db_writer_listener("vision-alert-events", models.AlertLog, r))
    asyncio.create_task(log_metrics_periodically())
    
    scheduler.add_job(reporting.generate_daily_report, trigger=CronTrigger(hour=1, minute=5, timezone='UTC'),
                      id="daily_report_job", replace_existing=True)
    scheduler.start()
    print("Application startup complete. Reporting scheduler started.")

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected(): break
            yield {"event": "message", "data": json.dumps(dict(realtime_state))}
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())

@app.get("/reports/latest")
async def get_latest_report_meta():
    reports_dir = "reports"
    if not os.path.exists(reports_dir) or not os.listdir(reports_dir):
        return {"error": "Отчётов пока нет."}
    latest = sorted(os.listdir(reports_dir), reverse=True)[0]
    return {"latest_report_filename": latest}

@app.get("/reports/download/{filename}")
async def download_report(filename: str):
    path = os.path.join("reports", filename)
    if not os.path.exists(path):
        return {"error": "Файл не найден."}
    return FileResponse(path=path, media_type="text/csv", filename=filename)