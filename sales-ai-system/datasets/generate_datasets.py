"""
Demo Dataset Generator
Creates synthetic but realistic sales, review, and inventory data.
Run: python generate_datasets.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

PRODUCTS = [
    {"id": "P001", "name": "Laptop Pro X1",       "category": "Electronics",  "price": 74999, "cost": 52000},
    {"id": "P002", "name": "Smart Watch S3",       "category": "Wearables",    "price": 24999, "cost": 15000},
    {"id": "P003", "name": "Wireless Earbuds",     "category": "Audio",        "price": 4999,  "cost": 2500},
    {"id": "P004", "name": "USB-C Hub 7-in-1",     "category": "Accessories",  "price": 1999,  "cost": 800},
    {"id": "P005", "name": "Mechanical Keyboard",  "category": "Peripherals",  "price": 8999,  "cost": 4500},
    {"id": "P006", "name": "Monitor 27inch 4K",    "category": "Electronics",  "price": 34999, "cost": 24000},
    {"id": "P007", "name": "Gaming Mouse RGB",     "category": "Peripherals",  "price": 2999,  "cost": 1200},
    {"id": "P008", "name": "Laptop Stand Aluminum","category": "Accessories",  "price": 1499,  "cost": 600},
]

CITIES = [
    ("Mumbai", "Maharashtra"), ("Bangalore", "Karnataka"), ("Delhi", "Delhi"),
    ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"), ("Pune", "Maharashtra"),
    ("Ahmedabad", "Gujarat"), ("Kolkata", "West Bengal"),
]

POSITIVE_REVIEWS = [
    "Amazing product! Works perfectly. Highly recommend.",
    "Great quality, fast delivery. Exceeded my expectations.",
    "Excellent build quality. Worth every rupee!",
    "Love this product. Best purchase I have made this year.",
    "Fantastic! The features are exactly what I needed.",
    "Outstanding quality. Delivery was super fast.",
    "Brilliant product. Setup was easy, works flawlessly.",
    "Superb value for money. Very happy with this purchase.",
]
NEGATIVE_REVIEWS = [
    "Terrible product. Stopped working after 2 days.",
    "Disappointed. The build quality is very poor.",
    "Arrived damaged. Support was completely unhelpful.",
    "Worst purchase ever. Do not buy this product.",
    "Battery drains in 2 hours. Very frustrating.",
    "Fake product. Nothing like the pictures shown.",
    "Broke immediately. Waste of money. Requesting refund.",
    "Charging issues from day one. Horrible experience.",
]
NEUTRAL_REVIEWS = [
    "Okay product. Does the job but nothing special.",
    "Decent quality for the price. Average experience.",
    "Works fine. Delivery took longer than expected.",
    "Acceptable product. Not amazing, not terrible.",
    "It works as described. Basic quality.",
]


def generate_sales(n_days: int = 365, output_path: str = "datasets/sales_data.csv"):
    records = []
    base_date = datetime(2023, 1, 1)

    for day in range(n_days):
        date = base_date + timedelta(days=day)
        is_weekend = date.weekday() >= 5
        is_festival = date.month in [10, 11] and date.day >= 15  # festival season

        for product in PRODUCTS:
            city, state = random.choice(CITIES)
            # Base demand
            base_qty = max(1, int(np.random.poisson(8)))
            if is_weekend: base_qty = int(base_qty * 1.4)
            if is_festival: base_qty = int(base_qty * 1.8)

            qty = base_qty
            discount = random.choice([0, 0, 0, 0.05, 0.1, 0.15])
            unit_price = product["price"] * (1 - discount)
            revenue = qty * unit_price
            profit  = qty * (unit_price - product["cost"])

            records.append({
                "date":         date.strftime("%Y-%m-%d"),
                "product_id":   product["id"],
                "product_name": product["name"],
                "category":     product["category"],
                "city":         city,
                "state":        state,
                "quantity":     qty,
                "unit_price":   round(unit_price, 2),
                "discount_pct": int(discount * 100),
                "revenue":      round(revenue, 2),
                "profit":       round(profit, 2),
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Sales data: {len(df):,} rows → {output_path}")
    return df


def generate_reviews(n: int = 2000, output_path: str = "datasets/customer_reviews.csv"):
    records = []
    for _ in range(n):
        product = random.choice(PRODUCTS)
        rating = random.choices([1, 2, 3, 4, 5], weights=[10, 8, 15, 25, 42])[0]
        if rating >= 4:
            text = random.choice(POSITIVE_REVIEWS)
            sentiment = "positive"
        elif rating <= 2:
            text = random.choice(NEGATIVE_REVIEWS)
            sentiment = "negative"
        else:
            text = random.choice(NEUTRAL_REVIEWS)
            sentiment = "neutral"

        records.append({
            "product_id":   product["id"],
            "product_name": product["name"],
            "category":     product["category"],
            "rating":       rating,
            "review_text":  text,
            "sentiment":    sentiment,
            "date":         (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Reviews data: {len(df):,} rows → {output_path}")
    return df


def generate_inventory(output_path: str = "datasets/inventory.csv"):
    records = []
    for p in PRODUCTS:
        stock = random.randint(5, 400)
        records.append({
            "product_id":   p["id"],
            "product_name": p["name"],
            "category":     p["category"],
            "stock_qty":    stock,
            "reorder_point": random.randint(10, 30),
            "reorder_qty":  random.randint(50, 200),
            "unit_cost":    p["cost"],
            "unit_price":   p["price"],
        })
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Inventory data: {len(df)} products → {output_path}")
    return df


if __name__ == "__main__":
    print("Generating demo datasets...")
    generate_sales()
    generate_reviews()
    generate_inventory()
    print("\nAll datasets generated successfully!")
    print("Next: cd backend && uvicorn main:app --reload")
