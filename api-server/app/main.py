import os
import json
import asyncio
from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from collections import defaultdict
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sse_starlette.sse import EventSourceResponse

# Local module imports
from . import models, reporting
from .database import async_engine, AsyncSessionLocal
from .telegram_bot import send_telegram_alert

# --- In-Memory State for Real-Time Metrics ---
realtime_state = {
    "consolidated_counts": defaultdict(int),
    "total_occupancy": 0,
    "per_camera_data": {}
}

# --- FastAPI App, Scheduler, and Template Engine ---
app = FastAPI(title="Vision Analytics API")
templates = Jinja2Templates(directory="app/templates")
scheduler = AsyncIOScheduler()

# --- Database Dependency ---
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# --- Background Task Listeners for Redis ---

async def data_listener(r: redis.Redis):
    """Listens for instantaneous occupancy data and aggregates it for the live dashboard."""
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-data-events")
    print("Listening for data events...")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                data = json.loads(message['data'])
                camera_id = data['camera_id']
                realtime_state['per_camera_data'][camera_id] = data
                
                consolidated_counts = defaultdict(int)
                for cam_data in realtime_state['per_camera_data'].values():
                    for zone, count in cam_data.get('zone_counts', {}).items():
                        consolidated_counts[zone] += count
                
                total_occupancy = sum(v for k, v in consolidated_counts.items() if k != 'barista')
                
                realtime_state['consolidated_counts'] = consolidated_counts
                realtime_state['total_occupancy'] = total_occupancy
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Data listener error: {e}")
            await asyncio.sleep(5)

async def alert_listener(r: redis.Redis):
    """Listens for alerts, triggers notifications, and logs to DB."""
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-alert-events")
    print("Listening for alert events...")
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                alert_data = json.loads(message['data'])
                await send_telegram_alert(alert_data['message'])
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        db_alert = models.AlertLog(
                            camera_id=alert_data['camera_id'],
                            alert_type=alert_data['alert_type'],
                            message=alert_data['message']
                        )
                        session.add(db_alert)
            except Exception as e:
                print(f"Alert processing error: {e}")

async def zone_change_listener(r: redis.Redis):
    """Listens for entry/exit events and logs them to the database for reporting."""
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-zone-change-events")
    print("Listening for zone change events...")
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                event_data = json.loads(message['data'])
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        db_event = models.TransitionEvent(
                            tracker_id=event_data['tracker_id'],
                            camera_id=event_data['camera_id'],
                            from_zone=event_data['from_zone'],
                            to_zone=event_data['to_zone']
                        )
                        session.add(db_event)
            except Exception as e:
                print(f"Zone change processing error: {e}")

async def log_metrics_periodically():
    """Logs the current aggregated occupancy metrics to the database every minute."""
    print("Periodic occupancy logging task started.")
    while True:
        await asyncio.sleep(60)
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    current_counts = realtime_state['consolidated_counts']
                    db_log = models.OccupancyLog(
                        total_occupancy=realtime_state['total_occupancy'],
                        queue_count=current_counts.get('queue', 0),
                        hall1_count=current_counts.get('hall1', 0),
                        hall2_count=current_counts.get('hall2', 0),
                    )
                    session.add(db_log)
        except Exception as e:
            print(f"Periodic logging error: {e}")

# --- Application Lifecycle Events (Startup/Shutdown) ---
@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    async with async_engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    
    # Initialize Redis connection and start background listeners
    r = redis.from_url(f"redis://{os.getenv('REDIS_HOST')}")
    asyncio.create_task(data_listener(r))
    asyncio.create_task(alert_listener(r))
    asyncio.create_task(zone_change_listener(r))
    asyncio.create_task(log_metrics_periodically())
    
    # Schedule the daily report job
    scheduler.add_job(
        reporting.generate_daily_report, 
        trigger=CronTrigger(hour=1, minute=5, timezone='UTC'),
        id="daily_report_job",
        name="Daily Summary Report",
        replace_existing=True
    )
    scheduler.start()
    print("Application startup complete. Reporting scheduler started.")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serves the main HTML dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    """Streams real-time metrics using Server-Sent Events."""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            yield {"event": "message", "data": json.dumps(dict(realtime_state))}
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())

@app.get("/reports/latest")
async def get_latest_report_path():
    """Returns the filename of the most recently generated report."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir) or not os.listdir(reports_dir):
        return {"error": "No reports generated yet."}
    latest_report = sorted(os.listdir(reports_dir), reverse=True)[0]
    return {"latest_report_filename": latest_report}

@app.get("/reports/download/{filename}")
async def download_report(filename: str):
    """Downloads a specific report file by its name."""
    filepath = os.path.join("reports", filename)
    if os.path.exists(filepath):
        return FileResponse(path=filepath, media_type='text/csv', filename=filename)
    return {"error": "File not found."}