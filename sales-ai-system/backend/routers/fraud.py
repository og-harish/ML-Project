"""Fraud Detection Router"""
from fastapi import APIRouter, Depends
from routers.auth import get_current_user
router = APIRouter()

@router.get("/alerts")
async def fraud_alerts(current_user=Depends(get_current_user)):
    return {"alerts": [
        {"type":"refund_spike","severity":"critical","message":"Refund rate for Smart Watch jumped from 2% to 18% in 48 hours (156 refunds).","timestamp":"2024-01-15T14:32:00","is_resolved":False},
        {"type":"sales_crash","severity":"warning","message":"Revenue in Delhi dropped 67% vs last Tuesday — possible listing issue.","timestamp":"2024-01-15T09:00:00","is_resolved":False},
        {"type":"fake_orders","severity":"warning","message":"22 orders from same device ID in 2 hours — possible bot activity.","timestamp":"2024-01-14T22:10:00","is_resolved":True},
        {"type":"unusual_spike","severity":"info","message":"Laptop Pro X1 orders up 340% in 3 hours — verify stock availability.","timestamp":"2024-01-14T16:45:00","is_resolved":True},
    ]}

@router.get("/risk-score")
async def risk_score(current_user=Depends(get_current_user)):
    return {"overall_risk": "medium", "score": 62, "factors": [
        {"factor":"Refund rate elevated","weight":35,"status":"critical"},
        {"factor":"Order patterns normal","weight":10,"status":"ok"},
        {"factor":"Revenue trends stable","weight":15,"status":"ok"},
        {"factor":"Regional anomaly detected","weight":25,"status":"warning"},
    ]}
