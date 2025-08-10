# api-server/app/reporting.py
import os, csv, asyncio
from datetime import datetime, date, timedelta, time as dtime
from zoneinfo import ZoneInfo
import redis.asyncio as redis
from sqlalchemy import select, desc, and_
from .database import AsyncSessionLocal
from .models import DailyKPI, OccupancyLog, AlertLog

ALMATY_TZ = ZoneInfo("Asia/Almaty")
OPEN_T = dtime(8, 30)
CLOSE_T = dtime(23, 59, 59)

BUCKET_MIN = int(os.getenv("PEAK_BUCKET_MIN", 30))  # ← configurable period

def day_key(prefix: str, the_dt: datetime) -> str:
    return f"{prefix}:{the_dt.strftime('%Y-%m-%d')}"

def day_bounds_almaty(the_day: date):
    start = datetime.combine(the_day, OPEN_T, tzinfo=ALMATY_TZ)
    end   = datetime.combine(the_day, CLOSE_T, tzinfo=ALMATY_TZ)
    return start, end

async def _read_today_redis(r: redis.Redis, the_dt: datetime):
    keys = {
        "unique_guests": day_key("unique_guests", the_dt),
        "pickup_valid":  day_key("pickup_valid", the_dt),
        "o2p":           day_key("conv:order_to_pickup", the_dt),
        "p2h":           day_key("conv:pickup_to_hall", the_dt),
        "p2e":           day_key("conv:pickup_to_exit", the_dt),
        "sum_o2p":       day_key("sum:o2p_s", the_dt),
        "cnt_o2p":       day_key("cnt:o2p", the_dt),
        "sum_p2h":       day_key("sum:p2h_s", the_dt),
        "cnt_p2h":       day_key("cnt:p2h", the_dt),
        "sum_p2e":       day_key("sum:p2e_s", the_dt),
        "cnt_p2e":       day_key("cnt:p2e", the_dt),
    }
    p = r.pipeline()
    for k in keys.values(): p.get(k)
    vals = await p.execute()

    def to_int(x): 
        try: return int(x) if x is not None else 0
        except: return 0
    def to_float(x):
        try: return float(x) if x is not None else 0.0
        except: return 0.0

    return {
        "unique_guests": to_int(vals[0]),
        "pickup_valid":  to_int(vals[1]),
        "o2p": to_int(vals[2]), "p2h": to_int(vals[3]), "p2e": to_int(vals[4]),
        "sum_o2p": to_float(vals[5]), "cnt_o2p": to_int(vals[6]),
        "sum_p2h": to_float(vals[7]), "cnt_p2h": to_int(vals[8]),
        "sum_p2e": to_float(vals[9]), "cnt_p2e": to_int(vals[10]),
    }

def _bucket_start(ts: datetime, size_min: int) -> datetime:
    # Align to bucket boundary in local tz
    minute = (ts.minute // size_min) * size_min
    return ts.replace(minute=minute, second=0, microsecond=0)

async def generate_daily_report():
    """
    Run at 23:59 Asia/Almaty.
    Copies counters from Redis, computes conversion %s + averages,
    finds absolute peak and the peak BUCKET_MIN-minute period by average,
    writes DB row + CSV.
    """
    now = datetime.now(ALMATY_TZ)
    the_day = now.date()

    # 1) Read Redis counters (before midnight)
    r = redis.from_url(f"redis://{os.getenv('REDIS_HOST')}")
    counters = await _read_today_redis(r, now)

    pickup_valid = counters["pickup_valid"]
    pickup_to_hall_pct = 0.0 if pickup_valid == 0 else round(100.0 * counters["p2h"] / pickup_valid, 1)
    pickup_to_exit_pct = 0.0 if pickup_valid == 0 else round(100.0 * counters["p2e"] / pickup_valid, 1)

    avg_o2p = 0.0 if counters["cnt_o2p"] == 0 else round(counters["sum_o2p"] / counters["cnt_o2p"], 1)
    avg_p2h = 0.0 if counters["cnt_p2h"] == 0 else round(counters["sum_p2h"] / counters["cnt_p2h"], 1)
    avg_p2e = 0.0 if counters["cnt_p2e"] == 0 else round(counters["sum_p2e"] / counters["cnt_p2e"], 1)

    start_dt, end_dt = day_bounds_almaty(the_day)

    # 2) Query occupancy logs for the day and compute peaks + bucketed peak
    async with AsyncSessionLocal() as session:
        # absolute peak
        q_peak = (
            select(OccupancyLog)
            .where(and_(OccupancyLog.ts >= start_dt, OccupancyLog.ts <= end_dt))
            .order_by(desc(OccupancyLog.total_occupancy))
            .limit(1)
        )
        res = await session.execute(q_peak)
        row = res.scalar_one_or_none()

        total_peak = row.total_occupancy if row else 0
        total_peak_ts = row.ts if row else None
        hall_peak = row.hall_count if row else 0
        queue_peak = row.queue_count if row else 0
        hall_peak_ts = total_peak_ts
        queue_peak_ts = total_peak_ts

        # fetch all rows to bucket (minute-level is fine)
        q_all = (
            select(OccupancyLog.ts, OccupancyLog.total_occupancy)
            .where(and_(OccupancyLog.ts >= start_dt, OccupancyLog.ts <= end_dt))
            .order_by(OccupancyLog.ts.asc())
        )
        rows = (await session.execute(q_all)).all()

        # bucket by BUCKET_MIN, compute avg per bucket
        bucket_sum: dict[datetime, int] = {}
        bucket_cnt: dict[datetime, int] = {}
        for ts, tot in rows:
            bstart = _bucket_start(ts.astimezone(ALMATY_TZ), BUCKET_MIN)
            bucket_sum[bstart] = bucket_sum.get(bstart, 0) + int(tot or 0)
            bucket_cnt[bstart] = bucket_cnt.get(bstart, 0) + 1

        peak_period_start = None
        peak_period_end = None
        peak_period_avg = 0.0
        for bstart, s in bucket_sum.items():
            c = bucket_cnt.get(bstart, 1)
            avg = s / c
            if avg > peak_period_avg:
                peak_period_avg = avg
                peak_period_start = bstart
                peak_period_end   = bstart + timedelta(minutes=BUCKET_MIN)

        # 3) Barista alerts
        q_alerts = (
            select(AlertLog)
            .where(
                and_(
                    AlertLog.timestamp >= start_dt,
                    AlertLog.timestamp <= end_dt,
                    AlertLog.alert_type == "STAFF_ABSENCE",
                )
            )
        )
        alerts_res = await session.execute(q_alerts)
        barista_alerts = len(list(alerts_res.scalars()))

        # 4) Upsert DailyKPI
        existing = await session.get(DailyKPI, the_day)
        if not existing:
            existing = DailyKPI(day=the_day)
            session.add(existing)

        existing.unique_guests = counters["unique_guests"]
        existing.pickup_valid  = pickup_valid
        existing.order_to_pickup = counters["o2p"]
        existing.pickup_to_hall  = counters["p2h"]
        existing.pickup_to_exit  = counters["p2e"]
        existing.pickup_to_hall_pct = pickup_to_hall_pct
        existing.pickup_to_exit_pct = pickup_to_exit_pct
        existing.avg_o2p_s = avg_o2p
        existing.avg_p2h_s = avg_p2h
        existing.avg_p2e_s = avg_p2e

        existing.total_occupancy_peak = total_peak
        existing.total_occupancy_peak_time = total_peak_ts
        existing.hall_peak = hall_peak
        existing.hall_peak_time = hall_peak_ts
        existing.queue_peak = queue_peak
        existing.queue_peak_time = queue_peak_ts

        existing.barista_alerts = barista_alerts

        # NEW: store bucketed peak period
        existing.peak_period_start = peak_period_start
        existing.peak_period_end   = peak_period_end
        existing.peak_period_avg_occupancy = round(float(peak_period_avg), 2)

        existing.note = f"snapshot@23:59, bucket={BUCKET_MIN}min"

        await session.commit()

    # 5) CSV
    os.makedirs("reports", exist_ok=True)
    csv_name = f"reports/{the_day.isoformat()}_kpi.csv"
    with open(csv_name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # ---- Заголовки на русском ----
        w.writerow([
            "Дата",
            "Уникальные посетители",
            "Количество выдач заказов",
            "Конверсия: Заказ → Выдача",
            "Конверсия: Выдача → Зал",
            "Конверсия: Выдача → Выход",
            "Доля ушедших в зал, %",
            "Доля ушедших на выход, %",
            "Среднее время (Заказ → Выдача), сек",
            "Среднее время (Выдача → Зал), сек",
            "Среднее время (Выдача → Выход), сек",
            "Пиковое число посетителей",
            "Время пиковой нагрузки",
            "Пиковая загрузка зала",
            "Время пика в зале",
            "Пиковая длина очереди",
            "Время пика в очереди",
            "Оповещения об отсутствии бариста",
            "Пиковый период: Начало",
            "Пиковый период: Конец",
            "Средняя загруженность в пик. период",
            "Интервал анализа пиков, мин"
        ])
        # ---- Данные ----
        w.writerow([
            the_day.isoformat(),
            counters["unique_guests"],
            pickup_valid,
            counters["o2p"],                      # order_to_pickup
            counters["p2h"],                      # pickup_to_hall
            counters["p2e"],                      # pickup_to_exit
            pickup_to_hall_pct,
            pickup_to_exit_pct,
            avg_o2p,
            avg_p2h,
            avg_p2e,
            total_peak,
            (total_peak_ts.isoformat() if total_peak_ts else ""),
            hall_peak,
            (hall_peak_ts.isoformat() if hall_peak_ts else ""),
            queue_peak,
            (queue_peak_ts.isoformat() if queue_peak_ts else ""),
            barista_alerts,
            (peak_period_start.isoformat() if peak_period_start else ""),
            (peak_period_end.isoformat() if peak_period_end else ""),
            round(float(peak_period_avg), 2),
            BUCKET_MIN
        ])
    print(f"[reporting] wrote {csv_name}")