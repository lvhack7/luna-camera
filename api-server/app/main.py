import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from collections import defaultdict
import redis.asyncio as redis
from sse_starlette.sse import EventSourceResponse

from .telegram_bot import send_telegram_alert

# --- In-Memory State for Real-Time Metrics ---
# This is our single source of truth for the current state
realtime_state = {
    "consolidated_counts": defaultdict(int),
    "total_occupancy": 0,
    "per_camera_data": {}
}

# --- FastAPI App, Scheduler, and Template Engine ---
app = FastAPI(title="Vision Analytics API")
templates = Jinja2Templates(directory="app/templates") # Point to the templates directory

# --- Background Tasks to Listen to Redis ---
# (The data_listener and alert_listener functions remain exactly the same as before)
async def data_listener(r: redis.Redis):
    """Listens to vision data and aggregates it."""
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
    """Listens for alerts and triggers notifications."""
    pubsub = r.pubsub()
    await pubsub.subscribe("vision-alert-events")
    print("Listening for alert events...")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                alert_data = json.loads(message['data'])
                print(f"API received alert: {alert_data['message']}")
                await send_telegram_alert(alert_data['message'])
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Alert listener error: {e}")
            await asyncio.sleep(5)

# (The periodic logging task also remains the same)
async def log_metrics_periodically():
    """Logs the current metrics to the database every minute for reporting."""
    print("Periodic logging task started. Logging metrics every 60 seconds.")
    while True:
        await asyncio.sleep(60)
        print(f"DB LOG (simulation): {realtime_state}")

@app.on_event("startup")
async def startup_event():
    r = redis.from_url(f"redis://{os.getenv('REDIS_HOST')}")
    asyncio.create_task(data_listener(r))
    asyncio.create_task(alert_listener(r))
    asyncio.create_task(log_metrics_periodically())


# --- API Endpoints ---

# NEW: Endpoint to serve the HTML dashboard
@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serves the main HTML dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})

# NEW: The Server-Sent Events (SSE) endpoint
@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    """Streams real-time metrics using Server-Sent Events."""
    async def event_generator():
        while True:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                print("Client disconnected.")
                break

            # Yield the current state as a JSON string
            yield {
                "event": "message",
                "data": json.dumps(dict(realtime_state))
            }
            # Wait for a short time before sending the next update
            await asyncio.sleep(1)
            
    return EventSourceResponse(event_generator())


# The old endpoint can be kept for direct API access or removed
@app.get("/api/metrics/current")
async def get_current_metrics():
    """Provides the latest aggregated metrics as a one-time JSON response."""
    return realtime_state