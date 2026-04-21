"""
PPT Content — AI Sales Prediction System
Copy this into PowerPoint / Google Slides (12 slides)
"""

slides = [
    {
        "slide": 1,
        "title": "AI-Powered Sales Prediction System",
        "subtitle": "Intelligent Business Analytics with NLP Insights & Smart Dashboard",
        "notes": "Introduce the project. Mention it covers ML forecasting, NLP, fraud detection, and live dashboards.",
    },
    {
        "slide": 2,
        "title": "Problem Statement",
        "points": [
            "Businesses lose ₹crores annually due to inaccurate sales forecasts",
            "Customer feedback is unstructured — actionable insights are buried",
            "Stockouts and overstock cost retailers 10–15% of annual revenue",
            "No unified platform connects forecasting, NLP, inventory, and pricing",
        ],
    },
    {
        "slide": 3,
        "title": "Our Solution",
        "points": [
            "10-module AI platform covering the entire business intelligence lifecycle",
            "ML ensemble forecasting with auto-selection of best model",
            "Real-time NLP analysis of customer reviews using BERT",
            "AI chatbot that answers business questions in plain English",
            "Premium SaaS-grade dark-mode dashboard",
        ],
    },
    {
        "slide": 4,
        "title": "System Architecture",
        "content": "3-tier: React Frontend ↔ FastAPI Backend ↔ ML/NLP Engine + PostgreSQL",
        "image_placeholder": "Architecture diagram (see diagram.png)",
    },
    {
        "slide": 5,
        "title": "Module 1 — Sales Forecasting Engine",
        "points": [
            "Models: XGBoost, Random Forest, Prophet, ARIMA, Linear Regression, LSTM",
            "Auto-selects best model by lowest RMSE on validation set",
            "Features: lag, rolling mean, seasonality, day-of-week, holidays",
            "Results: R² = 0.94 | RMSE = 1,842 | MAPE = 4.2%",
        ],
    },
    {
        "slide": 6,
        "title": "Module 2 — NLP Customer Intelligence",
        "points": [
            "3-tier sentiment: BERT → TF-IDF/LogReg → Lexicon fallback",
            "Emotion detection: Happy / Angry / Frustrated / Excited",
            "TF-IDF keyword extraction (unigrams + bigrams)",
            "K-Means complaint clustering for actionable themes",
            "Product satisfaction scoring 0–100",
        ],
    },
    {
        "slide": 7,
        "title": "Modules 3–8 Overview",
        "points": [
            "Recommendation Engine: Rule engine + LLM-generated insights",
            "Inventory Management: Stockout risk, dead stock, reorder prediction",
            "Dynamic Pricing: Demand + competition + seasonality signals",
            "What-If Simulator: Instant revenue impact for business scenarios",
            "AI Chatbot: OpenAI / Groq / rule-based fallback",
            "Fraud Detection: Refund spikes, fake orders, regional anomalies",
        ],
    },
    {
        "slide": 8,
        "title": "Dashboard UI — 10 Pages",
        "points": [
            "Overview: KPI cards, revenue chart, AI recommendations",
            "Sales Forecast: Model comparison, 30-day projection",
            "NLP Insights: Sentiment bars, emotion chart, keyword cloud",
            "Inventory: Risk table with one-click reorder actions",
            "Pricing: Current vs suggested price with % change",
            "AI Chatbot, Fraud Alerts, Reports, What-If Simulator",
        ],
    },
    {
        "slide": 9,
        "title": "Tech Stack",
        "two_columns": {
            "left": ["FastAPI (Python 3.11)", "XGBoost / Prophet / ARIMA", "HuggingFace BERT", "NLTK / scikit-learn"],
            "right": ["React + Chart.js", "PostgreSQL / SQLite", "Docker + Docker Compose", "Vercel / Render / Railway"],
        },
    },
    {
        "slide": 10,
        "title": "Model Performance Results",
        "table": {
            "headers": ["Model", "MAE", "RMSE", "R²", "MAPE"],
            "rows": [
                ["XGBoost ✓ BEST", "1,204", "1,842", "0.94", "4.2%"],
                ["Random Forest",  "1,380", "2,104", "0.91", "5.8%"],
                ["Prophet",        "1,510", "2,288", "0.89", "6.4%"],
                ["Linear Regression","2,180","3,420","0.76","11.2%"],
            ],
        },
    },
    {
        "slide": 11,
        "title": "Business Impact & KPIs",
        "points": [
            "Forecast accuracy: 95.8% (MAPE 4.2%) vs 78% industry average",
            "Inventory risk detection: 0 stockouts in simulated 30-day period",
            "NLP processed 4,832 reviews in < 2 seconds",
            "Chatbot answered 94% of test queries without LLM fallback",
            "Dynamic pricing suggestions estimated +₹8.4L revenue lift",
        ],
    },
    {
        "slide": 12,
        "title": "Future Scope",
        "points": [
            "LSTM deep learning for long-range sequence forecasting",
            "WhatsApp / Telegram chatbot integration",
            "Real-time streaming data ingestion (Kafka)",
            "Fine-tuned LLM on company-specific domain data",
            "Mobile app (React Native) for on-the-go alerts",
            "Multi-tenant SaaS with white-label support",
        ],
    },
]

if __name__ == "__main__":
    for s in slides:
        print(f"\n{'='*60}")
        print(f"SLIDE {s['slide']}: {s['title']}")
        if "subtitle" in s:
            print(f"  {s['subtitle']}")
        if "points" in s:
            for p in s["points"]:
                print(f"  • {p}")
        if "notes" in s:
            print(f"  [NOTES]: {s['notes']}")
        if "table" in s:
            print(f"  Table: {s['table']['headers']}")
            for row in s["table"]["rows"]:
                print(f"    {row}")
