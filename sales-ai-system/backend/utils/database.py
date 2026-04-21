"""
Database configuration using SQLAlchemy with async support.
Supports SQLite (dev) and PostgreSQL (production).
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./sales_ai.db"
)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ─── ORM Models ────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String(255), unique=True, index=True, nullable=False)
    name         = Column(String(255), nullable=False)
    password_hash= Column(String(255), nullable=False)
    role         = Column(String(50), default="viewer")   # admin | analyst | viewer
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)


class SalesRecord(Base):
    __tablename__ = "sales_records"
    id           = Column(Integer, primary_key=True, index=True)
    date         = Column(DateTime, nullable=False, index=True)
    product_id   = Column(String(100), nullable=False, index=True)
    product_name = Column(String(255))
    category     = Column(String(100))
    region       = Column(String(100))
    city         = Column(String(100))
    quantity     = Column(Integer, default=0)
    unit_price   = Column(Float, default=0.0)
    revenue      = Column(Float, default=0.0)
    profit       = Column(Float, default=0.0)
    discount     = Column(Float, default=0.0)


class CustomerReview(Base):
    __tablename__ = "customer_reviews"
    id           = Column(Integer, primary_key=True, index=True)
    product_id   = Column(String(100), index=True)
    product_name = Column(String(255))
    review_text  = Column(Text)
    rating       = Column(Float)
    sentiment    = Column(String(20))     # positive | negative | neutral
    emotion      = Column(String(50))     # happy | angry | frustrated | excited
    created_at   = Column(DateTime, default=datetime.utcnow)


class InventoryItem(Base):
    __tablename__ = "inventory"
    id           = Column(Integer, primary_key=True, index=True)
    product_id   = Column(String(100), unique=True, index=True)
    product_name = Column(String(255))
    category     = Column(String(100))
    stock_qty    = Column(Integer, default=0)
    reorder_point= Column(Integer, default=10)
    reorder_qty  = Column(Integer, default=50)
    unit_cost    = Column(Float, default=0.0)
    is_active    = Column(Boolean, default=True)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ForecastResult(Base):
    __tablename__ = "forecast_results"
    id           = Column(Integer, primary_key=True, index=True)
    forecast_date= Column(DateTime, nullable=False)
    product_id   = Column(String(100))
    predicted_revenue  = Column(Float)
    predicted_quantity = Column(Integer)
    model_used   = Column(String(50))
    confidence   = Column(Float)
    created_at   = Column(DateTime, default=datetime.utcnow)


class AlertLog(Base):
    __tablename__ = "alert_logs"
    id           = Column(Integer, primary_key=True, index=True)
    alert_type   = Column(String(100))   # low_stock | fraud | sales_drop
    severity     = Column(String(20))    # critical | warning | info
    message      = Column(Text)
    is_resolved  = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)


# ─── Helpers ───────────────────────────────────────────────────────────────────

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
