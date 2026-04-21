"""Dashboard KPI Router"""
from fastapi import APIRouter, Depends
from routers.auth import get_current_user
import random

router = APIRouter()

@router.get("/kpis")
async def get_kpis(current_user=Depends(get_current_user)):
    return {
        "revenue": {"value": 4280000, "change": 12.4, "unit": "INR"},
        "orders": {"value": 8432, "change": 8.2, "unit": "count"},
        "avg_order_value": {"value": 507, "change": 3.8, "unit": "INR"},
        "return_rate": {"value": 4.2, "change": -1.1, "unit": "%"},
        "profit_margin": {"value": 28.6, "change": 2.3, "unit": "%"},
        "customer_sentiment": {"value": 74.2, "change": 5.1, "unit": "score"},
        "repeat_customers": {"value": 34.1, "change": 4.7, "unit": "%"},
        "inventory_turnover": {"value": 8.3, "change": 0.9, "unit": "times"},
        "forecast_confidence": {"value": 88.4, "change": 0.0, "unit": "%"},
    }

@router.get("/revenue-trend")
async def revenue_trend(current_user=Depends(get_current_user)):
    import random
    from datetime import datetime, timedelta
    base = datetime(2024, 1, 1)
    data = []
    revenue = 300000
    for i in range(365):
        date = base + timedelta(days=i)
        revenue = revenue * (1 + random.uniform(-0.03, 0.05))
        data.append({"date": date.strftime("%Y-%m-%d"), "revenue": round(revenue), "profit": round(revenue * 0.28)})
    return {"data": data}

@router.get("/region-performance")
async def region_performance(current_user=Depends(get_current_user)):
    return {"regions": [
        {"city": "Mumbai",    "revenue": 4200000, "orders": 1820, "growth": 8.4},
        {"city": "Bangalore", "revenue": 3800000, "orders": 1640, "growth": 14.2},
        {"city": "Delhi",     "revenue": 3100000, "orders": 1380, "growth": 6.1},
        {"city": "Chennai",   "revenue": 2400000, "orders": 1050, "growth": 23.4},
        {"city": "Hyderabad", "revenue": 1900000, "orders": 840,  "growth": 18.7},
        {"city": "Pune",      "revenue": 1600000, "orders": 720,  "growth": 11.3},
    ]}
