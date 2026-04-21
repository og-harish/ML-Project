"""
Sales Forecast Router
Endpoints for training, predicting, and comparing ML models.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
import io

from utils.database import get_db, SalesRecord
from ml.forecasting import run_forecast
from routers.auth import get_current_user

router = APIRouter()


@router.post("/train")
async def train_forecast(
    file: UploadFile = File(None),
    days: int = Query(30, ge=7, le=365),
    granularity: str = Query("daily"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Train all forecasting models on uploaded CSV or stored sales data.
    Returns best model selection, forecast, and model comparison metrics.
    """
    if file:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        required = {"date", "revenue"}
        if not required.issubset(df.columns):
            raise HTTPException(400, f"CSV must contain columns: {required}")
    else:
        # Load from database
        result = await db.execute(select(SalesRecord).order_by(SalesRecord.date))
        records = result.scalars().all()
        if not records:
            # Generate demo data
            df = _generate_demo_sales()
        else:
            df = pd.DataFrame([{
                "date": r.date, "revenue": r.revenue
            } for r in records])

    result = run_forecast(df, forecast_days=days, granularity=granularity)
    return result


@router.get("/demo")
async def forecast_demo(days: int = Query(30)):
    """Run forecast on built-in demo dataset (no auth required for demo)."""
    df = _generate_demo_sales()
    return run_forecast(df, forecast_days=days)


@router.get("/top-products")
async def top_products(db: AsyncSession = Depends(get_db), current_user=Depends(get_current_user)):
    """Return top selling products by revenue."""
    result = await db.execute(select(SalesRecord))
    records = result.scalars().all()
    if not records:
        return _demo_top_products()

    df = pd.DataFrame([{"product_name": r.product_name, "revenue": r.revenue, "quantity": r.quantity} for r in records])
    grouped = df.groupby("product_name").agg({"revenue": "sum", "quantity": "sum"}).reset_index()
    top = grouped.sort_values("revenue", ascending=False).head(10)
    return {"top_products": top.to_dict(orient="records")}


def _generate_demo_sales(n_days: int = 365) -> pd.DataFrame:
    """Generate synthetic daily sales data with trend + seasonality."""
    import numpy as np
    from datetime import datetime, timedelta
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    t = np.arange(n_days)
    # Trend + weekly + monthly seasonality + noise
    trend = 5000 + t * 12
    weekly = 800 * np.sin(2 * np.pi * t / 7)
    monthly = 1500 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 400, n_days)
    revenue = trend + weekly + monthly + noise
    revenue = np.clip(revenue, 1000, None)
    return pd.DataFrame({"date": dates, "revenue": revenue.round(2)})


def _demo_top_products():
    return {"top_products": [
        {"product_name": "Laptop Pro X1",    "revenue": 284000, "quantity": 142},
        {"product_name": "Wireless Earbuds", "revenue": 156000, "quantity": 780},
        {"product_name": "Smart Watch S3",   "revenue": 134000, "quantity": 268},
        {"product_name": "USB-C Hub 7-in-1", "revenue": 89000,  "quantity": 1112},
        {"product_name": "Mechanical Keyboard", "revenue": 76000, "quantity": 304},
    ]}
