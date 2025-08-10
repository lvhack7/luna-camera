import os, csv
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from sqlalchemy import select, func

from .database import AsyncSessionLocal
from . import models
import redis.asyncio as redis

ALMATY = ZoneInfo("Asia/Almaty")

def dstr(dt: datetime): return dt.strftime("%Y-%m-%d")

async def _read_day_redis(r: redis.Redis, day: datetime):
    key = lambda k: f"{k}:{dstr(day)}"
    wants = [
        "unique_guests",
        "conv:order_to_pickup", "conv:pickup_to_hall", "conv:pickup_to_exit",
        "sum:o2p_s","cnt:o2p", "sum:p2h_s","cnt:p2h", "sum:p2e_s","cnt:p2e"
    ]
    pipe = r.pipeline()
    for k in wants: pipe.get(key(k))
    vals = await pipe.execute()
    m = dict(zip(wants, vals))

    to_i = lambda x: int(x) if x is not None else 0
    def to_f(x):
        try: return float(x)
        except: return 0.0

    o2p_avg = (to_f(m["sum:o2p_s"]) / max(1, to_i(m["cnt:o2p"])))
    p2h_avg = (to_f(m["sum:p2h_s"]) / max(1, to_i(m["cnt:p2h"])))
    p2e_avg = (to_f(m["sum:p2e_s"]) / max(1, to_i(m["cnt:p2e"])))

    return {
        "unique_guests": to_i(m["unique_guests"]),
        "o2p": {"count": to_i(m["conv:order_to_pickup"]), "avg_s": round(o2p_avg,1)},
        "p2h": {"count": to_i(m["conv:pickup_to_hall"]),  "avg_s": round(p2h_avg,1)},
        "p2e": {"count": to_i(m["conv:pickup_to_exit"]),  "avg_s": round(p2e_avg,1)},
    }

async def generate_daily_report():
    now = datetime.now(ALMATY)
    day = now - timedelta(days=1)

    day_start = datetime.combine(day.date(), dtime(8,30), tzinfo=ALMATY)
    day_end   = datetime.combine(day.date(), dtime(23,59,59), tzinfo=ALMATY)

    r = redis.from_url(f"redis://{os.getenv('REDIS_HOST','redis')}:6379/0")
    kpis = await _read_day_redis(r, day)

    async with AsyncSessionLocal() as sess:
        q = select(models.OccupancyLog.ts, models.OccupancyLog.total_occupancy).where(
            models.OccupancyLog.ts >= day_start,
            models.OccupancyLog.ts <= day_end
        ).order_by(models.OccupancyLog.total_occupancy.desc()).limit(1)
        res = await sess.execute(q)
        peak_ts, peak_occ = None, 0
        if (row := res.first()):
            peak_ts, peak_occ = row[0], row[1]

        q2 = select(func.count(models.AlertLog.id)).where(
            models.AlertLog.timestamp >= day_start.timestamp(),
            models.AlertLog.timestamp <= day_end.timestamp(),
            models.AlertLog.alert_type == "STAFF_ABSENCE"
        )
        res2 = await sess.execute(q2)
        barista_alerts = int(res2.scalar_one())

    reports_dir = "app/reports"
    os.makedirs(reports_dir, exist_ok=True)
    fname = f"report_{dstr(day)}.csv"
    path = os.path.join(reports_dir, fname)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Дата", dstr(day)])
        w.writerow(["Пиковая загрузка (чел.)", peak_occ])
        w.writerow(["Время пика (ALMT)", peak_ts.isoformat() if peak_ts else "—"])
        w.writerow([])
        w.writerow(["Уникальные гости (всего за день)", kpis["unique_guests"]])
        w.writerow([])
        w.writerow(["Конверсия", "Количество", "Среднее время, сек"])
        w.writerow(["Order → Pickup", kpis["o2p"]["count"], kpis["o2p"]["avg_s"]])
        w.writerow(["Pickup → Hall",  kpis["p2h"]["count"], kpis["p2h"]["avg_s"]])
        w.writerow(["Pickup → Exit",  kpis["p2e"]["count"], kpis["p2e"]["avg_s"]])
        w.writerow([])
        w.writerow(["Оповещения (бариста отсутствует)", barista_alerts])

    print(f"[reporting] Saved {path}")
    return path