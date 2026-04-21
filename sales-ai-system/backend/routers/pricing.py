"""Dynamic Pricing Router"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from routers.auth import get_current_user
router = APIRouter()

class PricingScenario(BaseModel):
    product_id: str
    current_price: float
    demand_index: float = 1.0
    inventory_qty: int = 50
    competitor_price: Optional[float] = None
    is_seasonal: bool = False

@router.post("/suggest")
async def suggest_price(scenario: PricingScenario, current_user=Depends(get_current_user)):
    """Suggest optimal price based on demand, inventory, and competition."""
    price = scenario.current_price
    demand_multiplier = 1 + (scenario.demand_index - 1) * 0.3
    if scenario.inventory_qty < 10:
        demand_multiplier *= 1.1  # scarcity premium
    elif scenario.inventory_qty > 200:
        demand_multiplier *= 0.93  # overstock discount
    if scenario.is_seasonal:
        demand_multiplier *= 1.08
    if scenario.competitor_price:
        comp_diff = (scenario.competitor_price - price) / price
        demand_multiplier *= 1 + comp_diff * 0.3
    suggested = round(price * demand_multiplier, -1)
    change_pct = round((suggested - price) / price * 100, 1)
    return {
        "current_price": price,
        "suggested_price": suggested,
        "change_pct": change_pct,
        "recommendation": "increase" if change_pct > 0 else "decrease" if change_pct < 0 else "maintain",
        "reason": _price_reason(scenario, change_pct),
        "estimated_revenue_impact": round(suggested * scenario.demand_index * 30, 0),
    }

def _price_reason(s: PricingScenario, change: float) -> str:
    if s.inventory_qty < 10: return "Low inventory — premium pricing justified"
    if s.inventory_qty > 200: return "High inventory — discounting to increase velocity"
    if s.is_seasonal: return "Seasonal demand spike — price increase recommended"
    if change > 0: return "High demand index supports price increase"
    if change < 0: return "Demand softening — strategic price reduction"
    return "Price is optimal for current conditions"

@router.get("/overview")
async def pricing_overview(current_user=Depends(get_current_user)):
    return {"products": [
        {"product_name":"Laptop Pro X1","current_price":74999,"suggested_price":79999,"change_pct":6.7,"recommendation":"increase"},
        {"product_name":"Smart Watch S3","current_price":24999,"suggested_price":22999,"change_pct":-8.0,"recommendation":"decrease"},
        {"product_name":"Wireless Earbuds","current_price":4999,"suggested_price":5499,"change_pct":10.0,"recommendation":"increase"},
        {"product_name":"USB-C Hub","current_price":1999,"suggested_price":1799,"change_pct":-10.0,"recommendation":"decrease"},
        {"product_name":"Mechanical Keyboard","current_price":8999,"suggested_price":8999,"change_pct":0.0,"recommendation":"maintain"},
    ]}
