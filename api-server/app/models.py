# app/models.py
from datetime import datetime
from typing import Optional
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

    __table_args__ = (
        sa.Index("ix_occupancy_log_ts", "ts"),
    )

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
class DailySummary(Base):
    __tablename__ = "daily_summary"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(sa.Date, unique=True, index=True, nullable=False)

    # Denominators
    order_unique: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    pickup_unique: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)

    # Conversions
    conv_order_to_pickup: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    conv_pickup_to_hall: Mapped[int]  = mapped_column(sa.Integer, default=0, nullable=False)
    conv_pickup_to_exit: Mapped[int]  = mapped_column(sa.Integer, default=0, nullable=False)

    # Durations (seconds)
    avg_o2p_s: Mapped[float] = mapped_column(sa.Float, default=0.0, nullable=False)
    avg_p2h_s: Mapped[float] = mapped_column(sa.Float, default=0.0, nullable=False)
    avg_p2e_s: Mapped[float] = mapped_column(sa.Float, default=0.0, nullable=False)

    # Peak occupancy
    peak_ts: Mapped[Optional[datetime]] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    peak_occupancy: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )