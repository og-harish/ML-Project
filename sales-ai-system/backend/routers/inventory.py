"""Inventory Management Router"""
from fastapi import APIRouter, Depends
from routers.auth import get_current_user
router = APIRouter()

@router.get("/status")
async def inventory_status(current_user=Depends(get_current_user)):
    return {"inventory": [
        {"product_id":"P001","product_name":"Laptop Pro X1","stock_qty":8,"reorder_point":15,"daily_sales":15,"risk":"critical","reorder_qty":100,"days_until_stockout":0.5},
        {"product_id":"P002","product_name":"Smart Watch S3","stock_qty":24,"reorder_point":20,"daily_sales":12,"risk":"warning","reorder_qty":80,"days_until_stockout":2},
        {"product_id":"P003","product_name":"Wireless Earbuds","stock_qty":12,"reorder_point":25,"daily_sales":20,"risk":"critical","reorder_qty":150,"days_until_stockout":0.6},
        {"product_id":"P004","product_name":"USB-C Hub 7-in-1","stock_qty":145,"reorder_point":30,"daily_sales":18,"risk":"ok","reorder_qty":0,"days_until_stockout":8},
        {"product_id":"P005","product_name":"Mechanical Keyboard","stock_qty":320,"reorder_point":20,"daily_sales":8,"risk":"overstock","reorder_qty":0,"days_until_stockout":40},
        {"product_id":"P006","product_name":"Monitor 27inch 4K","stock_qty":35,"reorder_point":10,"daily_sales":5,"risk":"ok","reorder_qty":0,"days_until_stockout":7},
    ]}

@router.get("/dead-stock")
async def dead_stock(current_user=Depends(get_current_user)):
    return {"dead_stock": [
        {"product_name":"Bluetooth Speaker Mini","last_sold_days_ago":45,"stock_qty":87,"cost_locked":43500},
        {"product_name":"USB 2.0 Hub (old model)","last_sold_days_ago":62,"stock_qty":124,"cost_locked":12400},
    ]}
