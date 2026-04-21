"""
NLP Intelligence Router
Endpoints for customer review analysis.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io

from nlp.analyzer import analyze_reviews
from routers.auth import get_current_user

router = APIRouter()


class ReviewItem(BaseModel):
    text: str
    product_id: str = "unknown"
    product_name: str = "Unknown Product"
    rating: Optional[float] = None


class ReviewBatch(BaseModel):
    reviews: List[ReviewItem]


@router.post("/analyze")
async def analyze(batch: ReviewBatch, current_user=Depends(get_current_user)):
    """Analyze a batch of customer reviews."""
    return analyze_reviews([r.dict() for r in batch.reviews])


@router.post("/analyze-csv")
async def analyze_csv(
    file: UploadFile = File(...),
    text_col: str = "review_text",
    product_col: str = "product_name",
    rating_col: str = "rating",
    current_user=Depends(get_current_user),
):
    """Analyze reviews from uploaded CSV file."""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if text_col not in df.columns:
        raise HTTPException(400, f"Column '{text_col}' not found in CSV.")
    reviews = []
    for _, row in df.iterrows():
        reviews.append({
            "text": str(row.get(text_col, "")),
            "product_name": str(row.get(product_col, "Unknown")),
            "product_id": str(row.get("product_id", "unknown")),
            "rating": float(row[rating_col]) if rating_col in row else None,
        })
    return analyze_reviews(reviews)


@router.get("/demo")
async def nlp_demo():
    """Run NLP analysis on demo review data."""
    demo_reviews = [
        {"text": "Amazing product! Fast delivery and great quality.", "product_id": "P001", "product_name": "Laptop Pro", "rating": 5},
        {"text": "Terrible experience. Product broke after 2 days.", "product_id": "P002", "product_name": "Smart Watch", "rating": 1},
        {"text": "Okay product, nothing special. Does the job.", "product_id": "P001", "product_name": "Laptop Pro", "rating": 3},
        {"text": "I love this! Best purchase I've made this year!", "product_id": "P003", "product_name": "Earbuds", "rating": 5},
        {"text": "Disappointed. Arrived damaged and support was unhelpful.", "product_id": "P002", "product_name": "Smart Watch", "rating": 2},
        {"text": "Excellent build quality. Worth every rupee.", "product_id": "P003", "product_name": "Earbuds", "rating": 5},
        {"text": "Slow delivery, packaging was damaged.", "product_id": "P004", "product_name": "USB Hub", "rating": 2},
        {"text": "Absolutely fantastic! Exceeded my expectations.", "product_id": "P001", "product_name": "Laptop Pro", "rating": 5},
        {"text": "Frustrating. The product stopped working in a week.", "product_id": "P002", "product_name": "Smart Watch", "rating": 1},
        {"text": "Good value for money. Happy with the purchase.", "product_id": "P004", "product_name": "USB Hub", "rating": 4},
    ]
    return analyze_reviews(demo_reviews)
