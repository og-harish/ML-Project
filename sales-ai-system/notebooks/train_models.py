"""
Model Training & Evaluation Script
Run this standalone to train, evaluate, and save all ML models.

Usage:
    python notebooks/train_models.py

Outputs saved to: models/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "../datasets/sales_data.csv"
MODELS_DIR  = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        daily = df.groupby("date")["revenue"].sum().reset_index()
        print(f"Loaded {len(daily)} days of sales data.")
    else:
        print("No data file found — generating synthetic data...")
        sys.path.insert(0, "../datasets")
        from generate_datasets import generate_sales
        df = generate_sales()
        daily = df.groupby("date")["revenue"].sum().reset_index()
    return daily


def train_all_models(df):
    from ml.forecasting import run_forecast
    print("\nTraining all forecasting models...")
    result = run_forecast(df, forecast_days=30)

    print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE':>8}")
    print("-" * 60)
    for m in result["model_comparison"]:
        best_mark = " ← BEST" if m["is_best"] else ""
        print(f"{m['model']:<20} {m['MAE']:>10,.0f} {m['RMSE']:>10,.0f} {m['R2']:>8.4f} {m['MAPE']:>7.1f}%{best_mark}")

    print(f"\n30-Day Revenue Forecast: ₹{result['total_predicted_revenue']:,.0f}")
    print(f"Avg Daily Revenue:       ₹{result['avg_daily_revenue']:,.0f}")

    # Save results
    with open(MODELS_DIR / "forecast_results.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "historical"}, f, indent=2, default=str)
    print(f"\nResults saved to {MODELS_DIR}/forecast_results.json")
    return result


def train_nlp_model():
    """Train TF-IDF + Logistic Regression sentiment classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report

    reviews_path = "../datasets/customer_reviews.csv"
    if not Path(reviews_path).exists():
        print("\nNo review data found — skipping NLP training.")
        return

    df = pd.read_csv(reviews_path)
    print(f"\nTraining NLP sentiment classifier on {len(df)} reviews...")

    X = df["review_text"].fillna("").tolist()
    y = df["sentiment"].tolist()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=200, C=1.0, random_state=42)),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")
    print(f"Cross-val F1: {scores.mean():.4f} ± {scores.std():.4f}")

    pipeline.fit(X, y)
    with open(MODELS_DIR / "sentiment_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Sentiment model saved to {MODELS_DIR}/sentiment_model.pkl")

    # Quick test
    samples = [
        "Excellent product! Fast delivery and great quality.",
        "Terrible. Broke after 2 days. Very disappointed.",
        "Okay product, nothing special.",
    ]
    for s in samples:
        pred = pipeline.predict([s])[0]
        prob = pipeline.predict_proba([s]).max()
        print(f"  '{s[:45]}...' → {pred} ({prob:.2f})")


def evaluate_inventory():
    """Run inventory risk scoring."""
    inv_path = "../datasets/inventory.csv"
    if not Path(inv_path).exists():
        print("\nNo inventory data found — skipping.")
        return

    df = pd.read_csv(inv_path)
    print(f"\nInventory Risk Analysis:")
    print(f"{'Product':<30} {'Stock':>8} {'Reorder Pt':>12} {'Status':>10}")
    print("-" * 65)
    for _, row in df.iterrows():
        status = "CRITICAL" if row["stock_qty"] < row["reorder_point"] else "OK"
        print(f"{row['product_name']:<30} {row['stock_qty']:>8} {row['reorder_point']:>12} {status:>10}")


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Sales Prediction System — Model Training")
    print("=" * 60)

    df = load_data()
    train_all_models(df)
    train_nlp_model()
    evaluate_inventory()

    print("\n✅ Training complete. Models saved to /models/")
