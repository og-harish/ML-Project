"""
AI-Powered Sales Prediction System
FastAPI Backend - Main Application Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import os

from routers import forecast, nlp, inventory, pricing, chatbot, reports, fraud, auth, dashboard
from utils.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Sales AI System ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Sales Prediction System",
    description="Intelligent business analytics with sales forecasting, NLP insights, and AI recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router,      prefix="/api/auth",      tags=["Authentication"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(forecast.router,  prefix="/api/forecast",  tags=["Sales Forecast"])
app.include_router(nlp.router,       prefix="/api/nlp",       tags=["NLP Intelligence"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["Inventory"])
app.include_router(pricing.router,   prefix="/api/pricing",   tags=["Dynamic Pricing"])
app.include_router(chatbot.router,   prefix="/api/chatbot",   tags=["AI Chatbot"])
app.include_router(reports.router,   prefix="/api/reports",   tags=["Reports"])
app.include_router(fraud.router,     prefix="/api/fraud",     tags=["Fraud Detection"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "system": "AI Sales Prediction System", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
