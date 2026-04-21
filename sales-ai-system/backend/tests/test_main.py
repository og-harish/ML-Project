"""
Backend Test Suite
Run: cd backend && pytest tests/ -v
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set test env
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_salesai.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-ci"

from main import app


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session")
async def auth_token(client):
    # Create user
    await client.post("/api/auth/signup", json={
        "name": "Test User", "email": "test@test.com",
        "password": "test1234", "role": "admin"
    })
    # Login
    resp = await client.post("/api/auth/login", data={
        "username": "test@test.com", "password": "test1234"
    })
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ── Auth Tests ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_signup(client):
    resp = await client.post("/api/auth/signup", json={
        "name": "Alice", "email": "alice@test.com",
        "password": "password123", "role": "analyst"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == "alice@test.com"


@pytest.mark.asyncio
async def test_login(client):
    resp = await client.post("/api/auth/login", data={
        "username": "alice@test.com", "password": "password123"
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_me_requires_auth(client):
    resp = await client.get("/api/auth/me")
    assert resp.status_code == 401


# ── Forecast Tests ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_forecast_demo(client):
    resp = await client.get("/api/forecast/demo?days=14")
    assert resp.status_code == 200
    data = resp.json()
    assert "forecast" in data
    assert "best_model" in data
    assert len(data["forecast"]) == 14
    assert "total_predicted_revenue" in data


@pytest.mark.asyncio
async def test_forecast_structure(client):
    resp = await client.get("/api/forecast/demo")
    data = resp.json()
    assert "model_comparison" in data
    assert len(data["model_comparison"]) >= 2
    assert any(m["is_best"] for m in data["model_comparison"])


# ── NLP Tests ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_nlp_demo(client):
    resp = await client.get("/api/nlp/demo")
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment_distribution" in data
    assert "total_reviews" in data
    assert "keywords" in data


@pytest.mark.asyncio
async def test_nlp_analyze_requires_auth(client):
    resp = await client.post("/api/nlp/analyze", json={"reviews": []})
    assert resp.status_code == 401


# ── Dashboard Tests ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_dashboard_kpis_requires_auth(client):
    resp = await client.get("/api/dashboard/kpis")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_dashboard_kpis_authenticated(client, auth_token):
    resp = await client.get("/api/dashboard/kpis",
                            headers={"Authorization": f"Bearer {auth_token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "revenue" in data
    assert "orders" in data


# ── Inventory Tests ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_inventory_status(client, auth_token):
    resp = await client.get("/api/inventory/status",
                            headers={"Authorization": f"Bearer {auth_token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "inventory" in data
    assert len(data["inventory"]) > 0


# ── Fraud Tests ───────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fraud_alerts(client, auth_token):
    resp = await client.get("/api/fraud/alerts",
                            headers={"Authorization": f"Bearer {auth_token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data


# ── NLP Unit Tests ────────────────────────────────────────────────────────────
def test_nlp_sentiment_positive():
    from nlp.analyzer import classify_sentiment
    result = classify_sentiment("Amazing product! I love it. Excellent quality.")
    assert result["sentiment"] == "positive"
    assert result["score"] > 0.5


def test_nlp_sentiment_negative():
    from nlp.analyzer import classify_sentiment
    result = classify_sentiment("Terrible product. Broke immediately. Very disappointed.")
    assert result["sentiment"] == "negative"
    assert result["score"] > 0.5


def test_nlp_emotion_happy():
    from nlp.analyzer import detect_emotion
    emotion = detect_emotion("I love this! Excellent and fantastic!")
    assert emotion == "happy"


def test_nlp_keyword_extraction():
    from nlp.analyzer import extract_keywords
    texts = ["battery issue charging problem", "battery drain slow charging", "great battery life"]
    keywords = extract_keywords(texts, top_n=5)
    assert len(keywords) > 0
    assert any("battery" in k["keyword"] for k in keywords)


def test_nlp_analyze_reviews():
    from nlp.analyzer import analyze_reviews
    reviews = [
        {"text": "Amazing product! Love it.", "product_id": "P1", "product_name": "Widget", "rating": 5},
        {"text": "Terrible. Broke immediately.", "product_id": "P2", "product_name": "Gadget", "rating": 1},
    ]
    result = analyze_reviews(reviews)
    assert result["total_reviews"] == 2
    assert result["sentiment_distribution"]["positive"] >= 1
    assert result["sentiment_distribution"]["negative"] >= 1


# ── ML Unit Tests ─────────────────────────────────────────────────────────────
def test_forecast_build_features():
    import pandas as pd
    from datetime import datetime, timedelta
    from ml.forecasting import build_features
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({"date": dates, "revenue": range(10)})
    result = build_features(df)
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns


def test_compute_metrics():
    import numpy as np
    from ml.forecasting import compute_metrics
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 290])
    metrics = compute_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] > 0
