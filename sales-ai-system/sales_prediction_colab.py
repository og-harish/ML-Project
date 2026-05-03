"""
Colab-ready Sales Prediction + NLP Insight pipeline.

Run in Google Colab or locally:
    pip install -r requirements.txt
    python sales_prediction_colab.py --sales-csv datasets/sales_data.csv --reviews-csv datasets/customer_reviews.csv
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional Colab dependency
    XGBRegressor = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional Gemini dependency
    genai = None

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - optional report dependency
    FPDF = None


ROOT = Path(__file__).resolve().parent
DEFAULT_SALES_CSV = ROOT / "datasets" / "sales_data.csv"
DEFAULT_REVIEWS_CSV = ROOT / "datasets" / "customer_reviews.csv"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "models"

SYSTEM_PROMPT = """
You are an expert sales analyst AI embedded in a Sales Prediction Dashboard.
Return valid JSON with these keys: sentiment, entities, narrative, recommendations, anomalies.
Be concise, data-driven, actionable, and do not hallucinate numbers.
"""


@dataclass
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    feature_columns: list[str]
    forecast: list[dict[str, float | str]]
    total_predicted_revenue: float
    avg_daily_revenue: float


def generate_demo_sales(n_days: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    products = [
        ("Laptop Pro X1", "Electronics", 74999, 52000),
        ("Smart Watch S3", "Wearables", 24999, 15000),
        ("Wireless Earbuds", "Audio", 4999, 2500),
        ("USB-C Hub 7-in-1", "Accessories", 1999, 800),
        ("Mechanical Keyboard", "Peripherals", 8999, 4500),
        ("Monitor 27inch 4K", "Electronics", 34999, 24000),
    ]
    regions = [
        ("Mumbai", "India", 19.0760, 72.8777),
        ("Bangalore", "India", 12.9716, 77.5946),
        ("Delhi", "India", 28.7041, 77.1025),
        ("Chennai", "India", 13.0827, 80.2707),
        ("Hyderabad", "India", 17.3850, 78.4867),
        ("Pune", "India", 18.5204, 73.8567),
    ]
    records = []
    start = pd.Timestamp("2024-01-01")

    for day in range(n_days):
        date = start + pd.Timedelta(days=day)
        weekend_boost = 1.25 if date.dayofweek >= 5 else 1.0
        festival_boost = 1.55 if date.month in {10, 11} else 1.0
        trend = 1 + (day / n_days) * 0.18

        for product_name, category, price, cost in products:
            city, country, lat, lon = regions[int(rng.integers(0, len(regions)))]
            discount_pct = int(rng.choice([0, 0, 5, 10, 15]))
            demand = max(1, int(rng.poisson(7) * weekend_boost * festival_boost * trend))
            unit_price = price * (1 - discount_pct / 100)
            revenue = demand * unit_price
            records.append(
                {
                    "date": date.date().isoformat(),
                    "region": city,
                    "country": country,
                    "latitude": lat,
                    "longitude": lon,
                    "product_category": category,
                    "product_name": product_name,
                    "units_sold": demand,
                    "discount_pct": discount_pct,
                    "revenue": round(revenue, 2),
                    "profit": round(demand * (unit_price - cost), 2),
                    "customer_reviews": rng.choice(
                        [
                            "Excellent quality and fast delivery.",
                            "Good product but delivery was late.",
                            "Poor packaging and support was slow.",
                            "Great value for money.",
                            "Average experience, needs better warranty.",
                        ]
                    ),
                }
            )

    return pd.DataFrame(records)


def load_sales_data(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else DEFAULT_SALES_CSV
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return generate_demo_sales()


def normalize_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "city": "region",
        "state": "country",
        "category": "product_category",
        "quantity": "units_sold",
        "review_text": "customer_reviews",
        "text_reviews": "customer_reviews",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns and new not in df.columns})

    required = {"date", "region", "product_category", "units_sold", "revenue"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Sales CSV is missing required columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["region"] = df["region"].fillna("Unknown")
    df["product_category"] = df["product_category"].fillna("Unknown")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["discount_pct"] = pd.to_numeric(df.get("discount_pct", 0), errors="coerce").fillna(0)
    df["profit"] = pd.to_numeric(df.get("profit", df["revenue"] * 0.22), errors="coerce").fillna(0)

    if "customer_reviews" not in df.columns:
        df["customer_reviews"] = ""
    if "country" not in df.columns:
        df["country"] = "India"

    return df.sort_values("date").reset_index(drop=True)


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            units_sold=("units_sold", "sum"),
            discount_pct=("discount_pct", "mean"),
            profit=("profit", "sum"),
        )
        .sort_values("date")
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_month"] = out["date"].dt.day
    out["week_number"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["revenue_lag_7"] = out["revenue"].shift(7)
    out["revenue_lag_30"] = out["revenue"].shift(30)
    out["rolling_mean_7"] = out["revenue"].rolling(7, min_periods=1).mean()
    out["rolling_mean_30"] = out["revenue"].rolling(30, min_periods=1).mean()
    return out.bfill().fillna(0)


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100), 2),
    }


def train_sales_model(df: pd.DataFrame, forecast_days: int = 90) -> tuple[TrainingResult, object]:
    daily = add_time_features(aggregate_daily_sales(df))
    feature_columns = [
        "units_sold",
        "discount_pct",
        "day_of_week",
        "day_of_month",
        "week_number",
        "month",
        "quarter",
        "is_weekend",
        "revenue_lag_7",
        "revenue_lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
    ]

    split = max(int(len(daily) * 0.8), 30)
    train_df = daily.iloc[:split]
    test_df = daily.iloc[split:]
    if test_df.empty:
        train_df = daily.iloc[:-14]
        test_df = daily.iloc[-14:]

    model_candidates = [
        ("Random Forest", RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)),
    ]
    if XGBRegressor is not None:
        model_candidates.insert(
            0,
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    objective="reg:squarederror",
                ),
            ),
        )

    best_name = ""
    best_model = None
    best_metrics: dict[str, float] | None = None

    for name, model in model_candidates:
        model.fit(train_df[feature_columns], train_df["revenue"])
        predictions = model.predict(test_df[feature_columns])
        metrics = compute_metrics(test_df["revenue"], predictions)
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name = name
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    forecast = forecast_future_revenue(daily, best_model, feature_columns, forecast_days)
    result = TrainingResult(
        model_name=best_name,
        metrics=best_metrics,
        feature_columns=feature_columns,
        forecast=forecast,
        total_predicted_revenue=round(float(sum(item["predicted_revenue"] for item in forecast)), 2),
        avg_daily_revenue=round(float(np.mean([item["predicted_revenue"] for item in forecast])), 2),
    )
    return result, best_model


def forecast_future_revenue(
    daily: pd.DataFrame,
    model: object,
    feature_columns: list[str],
    forecast_days: int,
) -> list[dict[str, float | str]]:
    history = daily[["date", "revenue", "units_sold", "discount_pct", "profit"]].copy()
    avg_units = float(history["units_sold"].tail(30).mean())
    avg_discount = float(history["discount_pct"].tail(30).mean())
    avg_profit_margin = float((history["profit"].sum() / max(history["revenue"].sum(), 1)) or 0.22)
    rows = []

    for step in range(1, forecast_days + 1):
        next_date = pd.to_datetime(history["date"].max()) + pd.Timedelta(days=1)
        probe = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "date": [next_date],
                        "revenue": [0.0],
                        "units_sold": [avg_units],
                        "discount_pct": [avg_discount],
                        "profit": [0.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        featured = add_time_features(probe)
        prediction = max(float(model.predict(featured.tail(1)[feature_columns])[0]), 0.0)
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "date": [next_date],
                        "revenue": [prediction],
                        "units_sold": [avg_units],
                        "discount_pct": [avg_discount],
                        "profit": [prediction * avg_profit_margin],
                    }
                ),
            ],
            ignore_index=True,
        )
        rows.append(
            {
                "date": next_date.date().isoformat(),
                "predicted_revenue": round(prediction, 2),
                "lower_bound": round(prediction * 0.92, 2),
                "upper_bound": round(prediction * 1.08, 2),
            }
        )

    return rows


def analyze_local_sentiment(texts: Iterable[str]) -> dict[str, object]:
    positive = {"excellent", "great", "amazing", "fast", "love", "perfect", "value", "recommend"}
    negative = {"poor", "bad", "late", "slow", "damaged", "terrible", "refund", "disappointed", "broken"}
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    keywords: dict[str, int] = {}

    for text in texts:
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in str(text).split()]
        score = sum(word in positive for word in words) - sum(word in negative for word in words)
        if score > 0:
            counts["positive"] += 1
        elif score < 0:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

        for word in words:
            if len(word) >= 5 and word not in {"product", "quality", "delivery"}:
                keywords[word] = keywords.get(word, 0) + 1

    total = max(sum(counts.values()), 1)
    sentiment = {key: round(value / total * 100, 2) for key, value in counts.items()}
    top_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)[:12]
    return {"sentiment": sentiment, "keywords": [{"term": term, "count": count} for term, count in top_keywords]}


def generate_nlp_insights(df: pd.DataFrame, forecast: TrainingResult, api_key: str | None = None) -> dict[str, object]:
    text_summary = " ".join(df["customer_reviews"].dropna().astype(str).head(80))
    local = analyze_local_sentiment(df["customer_reviews"].dropna().astype(str))
    top_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False).head(1)
    weak_region = df.groupby("region")["profit"].sum().sort_values().head(1)
    best_region = top_region.index[0] if not top_region.empty else "Unknown"
    lowest_profit_region = weak_region.index[0] if not weak_region.empty else "Unknown"

    fallback = {
        "sentiment": local["sentiment"],
        "entities": {
            "top_region": best_region,
            "lowest_profit_region": lowest_profit_region,
            "top_category": df.groupby("product_category")["revenue"].sum().idxmax(),
        },
        "narrative": (
            f"{best_region} is the strongest sales region while {lowest_profit_region} needs profit improvement. "
            f"The selected model predicts ₹{forecast.total_predicted_revenue:,.0f} revenue across the forecast window. "
            "Discount, weekend, and recent revenue trends are the strongest operational signals."
        ),
        "recommendations": [
            f"Increase marketing spend in {best_region} for high-performing categories.",
            f"Review pricing, stock, and delivery issues in {lowest_profit_region}.",
            "Use weekly retraining so the model adapts to new sales and review data.",
        ],
        "anomalies": detect_anomalies(df),
        "keywords": local["keywords"],
    }

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_api_key or genai is None:
        return fallback

    genai.configure(api_key=resolved_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    payload = {
        "sales_summary": {
            "forecast_revenue": forecast.total_predicted_revenue,
            "model": forecast.model_name,
            "metrics": forecast.metrics,
            "best_region": best_region,
            "lowest_profit_region": lowest_profit_region,
        },
        "sample_reviews": text_summary[:5000],
    }
    try:
        response = model.generate_content([SYSTEM_PROMPT, json.dumps(payload, default=str)])
        return json.loads(response.text)
    except json.JSONDecodeError:
        fallback["gemini_narrative"] = response.text
        return fallback
    except Exception as exc:
        fallback["gemini_error"] = f"{exc.__class__.__name__}: Gemini insight generation unavailable"
        return fallback


def generate_realtime_prediction_explanation(
    prediction_payload: dict[str, object],
    api_key: str | None = None,
) -> str:
    region = prediction_payload["region"]
    category = prediction_payload["category"]
    predicted_revenue = float(prediction_payload["predicted_revenue"])
    expected_profit = float(prediction_payload["expected_profit"])
    fallback = (
        f"{category} in {region} is predicted to generate INR {predicted_revenue:,.0f} "
        f"with about INR {expected_profit:,.0f} expected profit. Compare this against the "
        "selected segment baseline before committing inventory or promotion spend."
    )
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_api_key or genai is None:
        return fallback

    genai.configure(api_key=resolved_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Explain this real-time sales prediction in 2 concise business sentences. "
        "Do not invent numbers beyond the JSON payload."
    )
    try:
        response = model.generate_content([prompt, json.dumps(prediction_payload, default=str)])
    except Exception:
        return fallback
    return response.text.strip() or fallback


def detect_anomalies(df: pd.DataFrame) -> list[str]:
    regional = df.groupby("region")["revenue"].mean()
    overall = float(regional.mean())
    alerts = []
    for region, avg_revenue in regional.items():
        if avg_revenue > overall * 1.6:
            alerts.append(f"[ALERT] Region: {region} | Revenue is {avg_revenue / overall:.1f}x above average")
        elif avg_revenue < overall * 0.55:
            alerts.append(f"[ALERT] Region: {region} | Revenue is {avg_revenue / overall:.1f}x below average")
    return alerts[:8]


def export_pdf_report(result: TrainingResult, insights: dict[str, object], output_path: Path) -> None:
    if FPDF is None:
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(190, 10, "Sales Prediction Report")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(190, 8, f"Best model: {result.model_name}")
    pdf.multi_cell(190, 8, f"Metrics: {json.dumps(result.metrics)}")
    pdf.multi_cell(190, 8, f"Forecast revenue: INR {result.total_predicted_revenue:,.0f}")
    narrative = str(insights.get("narrative", "")).replace("₹", "INR ")
    pdf.multi_cell(190, 8, narrative)
    pdf.output(str(output_path))


def save_outputs(df: pd.DataFrame, result: TrainingResult, model: object, insights: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(result.forecast).to_csv(OUTPUT_DIR / "sales_forecast.csv", index=False)
    with open(OUTPUT_DIR / "insights.json", "w", encoding="utf-8") as file:
        json.dump(insights, file, indent=2, default=str)
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(result.__dict__, file, indent=2, default=str)
    with open(ARTIFACTS_DIR / "sales_forecast_model.pkl", "wb") as file:
        pickle.dump({"model": model, "features": result.feature_columns}, file)
    export_pdf_report(result, insights, OUTPUT_DIR / "sales_prediction_report.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sales forecast model and generate NLP insights.")
    parser.add_argument("--sales-csv", default=str(DEFAULT_SALES_CSV), help="Path to sales CSV")
    parser.add_argument("--forecast-days", type=int, default=90, help="Number of future days to forecast")
    args = parser.parse_args()

    raw_sales = load_sales_data(args.sales_csv)
    sales = normalize_sales_columns(raw_sales)
    result, model = train_sales_model(sales, forecast_days=args.forecast_days)
    insights = generate_nlp_insights(sales, result)
    save_outputs(sales, result, model, insights)

    print("Training complete")
    print(f"Best model: {result.model_name}")
    print(f"Metrics: {json.dumps(result.metrics)}")
    print(f"Forecast revenue: ₹{result.total_predicted_revenue:,.0f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
