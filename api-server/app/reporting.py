# app/reporting.py
import os
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sqlalchemy import select, func
import redis.asyncio as redis

from . import models

def _dates_for_report(tz: ZoneInfo, open_t, close_t, date_override: str | None = None):
    if date_override:
        day = datetime.strptime(date_override, "%Y-%m-%d").replace(tzinfo=tz)
    else:
        now = datetime.now(tz)
        # run at 00:05 for previous day
        day = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = day.replace(hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0)
    end = day.replace(hour=23, minute=59, second=59, microsecond=0)  # close at ~24:00
    return day, start, end

def _day_key(prefix: str, day: datetime):
    return f"{prefix}:{day.strftime('%Y-%m-%d')}"

async def generate_daily_report(AsyncSessionLocal, r: redis.Redis, reports_dir: str,
                                tz: ZoneInfo, open_t, close_t, date_override: str | None = None):
    day, start, end = _dates_for_report(tz, open_t, close_t, date_override)

    # 1) Pull baseline deltas from Redis (baseline taken at 08:30, so "current - baseline" at 24:00 equals day totals)
    async def _get(name): 
        v = await r.get(_day_key(name, day)); 
        if v is None: return 0
        try: return float(v) if name.startswith("sum:") else int(v)
        except: return 0

    keys = ["order_unique","pickup_unique","conv:order_to_pickup","conv:pickup_to_hall","conv:pickup_to_exit",
            "sum:o2p_s","cnt:o2p","sum:p2h_s","cnt:p2h","sum:p2e_s","cnt:p2e","alerts:barista"]
    vals = {k: await _get(k) for k in keys}

    def rate(num, den): return (num / den) if den > 0 else 0.0
    def avg(sum_s, cnt): return (sum_s / cnt) if cnt > 0 else 0.0

    # 2) Peak occupancy from DB
    async with AsyncSessionLocal() as session:
        # total occupancy peak and when
        q = (
            select(models.OccupancyLog.ts, models.OccupancyLog.total_occupancy)
            .where(models.OccupancyLog.ts >= start, models.OccupancyLog.ts <= end)
            .order_by(models.OccupancyLog.total_occupancy.desc())
            .limit(1)
        )
        res = (await session.execute(q)).first()
        peak_ts, peak_val = (res[0], res[1]) if res else (None, 0)

        # total visitors (unique tracker ids touching order or hall1/2 on camera_201)
        tv = (
            select(func.count(func.distinct(models.TransitionEvent.tracker_id)))
            .where(
                models.TransitionEvent.camera_id == "camera_201",
                models.TransitionEvent.timestamp >= start.timestamp(),
                models.TransitionEvent.timestamp <= end.timestamp(),
                models.TransitionEvent.event == "enter",
                models.TransitionEvent.zone.in_(["order","hall1","hall2"])
            )
        )
        visitors_total = (await session.execute(tv)).scalar_one() or 0

        # total barista alerts (also in Redis, but DB is authoritative)
        tb = (
            select(func.count())
            .select_from(models.AlertLog)
            .where(
                models.AlertLog.timestamp >= start.timestamp(),
                models.AlertLog.timestamp <= end.timestamp(),
                models.AlertLog.alert_type == "STAFF_ABSENCE"
            )
        )
        barista_alerts_db = (await session.execute(tb)).scalar_one() or 0

    # Prefer DB count for alerts; fall back to Redis if DB is empty
    alerts_barista = barista_alerts_db if barista_alerts_db else int(vals.get("alerts:barista", 0))

    # 3) Compute KPIs
    order_unique       = int(vals["order_unique"])
    pickup_unique      = int(vals["pickup_unique"])
    conv_o2p           = int(vals["conv:order_to_pickup"])
    conv_p2h           = int(vals["conv:pickup_to_hall"])
    conv_p2e           = int(vals["conv:pickup_to_exit"])
    avg_o2p_s          = avg(vals["sum:o2p_s"], vals["cnt:o2p"])
    avg_p2h_s          = avg(vals["sum:p2h_s"], vals["cnt:p2h"])
    avg_p2e_s          = avg(vals["sum:p2e_s"], vals["cnt:p2e"])
    rate_o2p           = rate(conv_o2p, order_unique)
    rate_p2h           = rate(conv_p2h, pickup_unique)
    rate_p2e           = rate(conv_p2e, pickup_unique)

    # 4) Write CSV
    os.makedirs(reports_dir, exist_ok=True)
    fname = f"{day.strftime('%Y-%m-%d')}_daily_report.csv"
    fpath = os.path.join(reports_dir, fname)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Дата", day.strftime("%Y-%m-%d")])
        w.writerow(["Окно анализа (Алматы)", start.strftime("%H:%M"), end.strftime("%H:%M")])
        w.writerow([])
        w.writerow(["Пиковая загрузка (чел.)", peak_val])
        w.writerow(["Время пика (локально)", peak_ts.astimezone(tz).strftime("%H:%M") if peak_ts else "—"])
        w.writerow([])
        w.writerow(["Всего посетителей (уникальные треки 201: order | hall1 | hall2)", visitors_total])
        w.writerow(["Предупреждений по бариста", alerts_barista])
        w.writerow([])
        w.writerow(["Показатель", "Числитель", "Знаменатель", "Конверсия", "Среднее время (сек)"])
        w.writerow(["Order → Pickup", conv_o2p, order_unique, f"{rate_o2p:.2%}", f"{avg_o2p_s:.1f}"])
        w.writerow(["Pickup → Hall",  conv_p2h, pickup_unique, f"{rate_p2h:.2%}", f"{avg_p2h_s:.1f}"])
        w.writerow(["Pickup → Exit",  conv_p2e, pickup_unique, f"{rate_p2e:.2%}", f"{avg_p2e_s:.1f}"])

    print(f"[report] saved {fpath}")
    return fpath