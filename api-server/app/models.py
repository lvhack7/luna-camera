# app/models.py
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, Float, Date, DateTime, String
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from .database import Base

# ---------- Live occupancy rollups (every minute) ----------
class OccupancyLog(Base):
    __tablename__ = "occupancy_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Store with timezone; you already use Asia/Almaty in the app
    ts: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), index=True, nullable=False)

    total_occupancy: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    queue_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    hall_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)

# ---------- Zone transitions (enter/exit), from Redis listener ----------
class TransitionEvent(Base):
    __tablename__ = "transition_event"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # epoch seconds from vision-processor for easy numeric queries in reports
    timestamp: Mapped[float] = mapped_column(sa.Float, index=True, nullable=False)

    camera_id: Mapped[str] = mapped_column(sa.String(64), index=True, nullable=False)
    event: Mapped[str]     = mapped_column(sa.String(16), nullable=False)  # "enter" | "exit"
    zone: Mapped[str]      = mapped_column(sa.String(64), index=True, nullable=False)

    # YOLO/ByteTrack track id; cast to int in the processor
    tracker_id: Mapped[int] = mapped_column(sa.Integer, index=True, nullable=False)

    # Optional raw payload for debugging / future needs
    payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        sa.Index("ix_transition_event_day_zone", "zone", "camera_id"),
    )

# ---------- Alerts (e.g., barista absence), from Redis listener ----------
class AlertLog(Base):
    __tablename__ = "alert_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # epoch seconds when alert was emitted
    timestamp: Mapped[float] = mapped_column(sa.Float, index=True, nullable=False)

    camera_id: Mapped[Optional[str]] = mapped_column(sa.String(64), index=True, nullable=True)
    alert_type: Mapped[str]   = mapped_column(sa.String(64), index=True, nullable=False)
    message: Mapped[str]      = mapped_column(sa.Text, nullable=False)

    payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

# ---------- Optional: daily snapshot row (nice for historical queries) ----------

class DailyKPI(Base):
    __tablename__ = "daily_kpi"

    day = Column(Date, primary_key=True)

    unique_guests = Column(Integer, default=0)
    pickup_valid  = Column(Integer, default=0)

    order_to_pickup = Column(Integer, default=0)
    pickup_to_hall  = Column(Integer, default=0)
    pickup_to_exit  = Column(Integer, default=0)

    pickup_to_hall_pct = Column(Float, default=0.0)
    pickup_to_exit_pct = Column(Float, default=0.0)

    avg_o2p_s = Column(Float, default=0.0)
    avg_p2h_s = Column(Float, default=0.0)
    avg_p2e_s = Column(Float, default=0.0)

    total_occupancy_peak = Column(Integer, default=0)
    total_occupancy_peak_time = Column(DateTime(timezone=True))

    hall_peak = Column(Integer, default=0)
    hall_peak_time = Column(DateTime(timezone=True))
    queue_peak = Column(Integer, default=0)
    queue_peak_time = Column(DateTime(timezone=True))

    barista_alerts = Column(Integer, default=0)

    # NEW: peak period of day (bucketed average)
    peak_period_start = Column(DateTime(timezone=True))
    peak_period_end   = Column(DateTime(timezone=True))
    peak_period_avg_occupancy = Column(Float, default=0.0)

    note = Column(String, default="")