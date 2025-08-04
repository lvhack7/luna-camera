# in api-server/app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    total_occupancy = Column(Integer)
    queue_count = Column(Integer)
    hall1_count = Column(Integer)
    hall2_count = Column(Integer)

class TransitionEvent(Base):
    __tablename__ = "transition_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    tracker_id = Column(BigInteger)
    camera_id = Column(String)
    from_zone = Column(String)
    to_zone = Column(String)

class AlertLog(Base):
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    camera_id = Column(String)
    alert_type = Column(String)
    message = Column(String)