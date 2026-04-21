# Viva Questions & Answers
# AI-Powered Sales Prediction System
# ============================================================

VIVA_QA = [

    # ── Machine Learning ──────────────────────────────────────
    {
        "category": "Machine Learning",
        "q": "Why did you use XGBoost over a simple neural network?",
        "a": (
            "XGBoost excels on tabular data with engineered features like lag values, "
            "rolling means, and calendar features. Neural networks need far more data and "
            "compute to match this. For our 365-day dataset, XGBoost achieved R²=0.94 while "
            "being 50x faster to train. We included LSTM as an advanced option for datasets "
            "with 3+ years of data where temporal patterns are deep."
        ),
    },
    {
        "category": "Machine Learning",
        "q": "How does Prophet handle seasonality differently from ARIMA?",
        "a": (
            "ARIMA models seasonality as a fixed lag pattern (AR/MA terms) and requires "
            "the series to be stationary. Prophet decomposes the series into trend + "
            "multiple seasonality components (weekly, yearly) + holiday effects using "
            "Fourier series. Prophet is more interpretable and handles missing data and "
            "trend changepoints automatically. ARIMA is more mathematically rigorous but "
            "brittle on non-stationary retail data."
        ),
    },
    {
        "category": "Machine Learning",
        "q": "What is the train/test split strategy and why?",
        "a": (
            "We use a temporal (walk-forward) split — the first 80% of chronological data "
            "for training, the last 20% for validation. Random splitting would cause data "
            "leakage because future values would appear in training. Temporal split mimics "
            "real deployment: the model always predicts future data it has never seen."
        ),
    },
    {
        "category": "Machine Learning",
        "q": "What features did you engineer for the forecasting models?",
        "a": (
            "Time-based: day of week, day of month, week number, month, quarter, year, "
            "is_weekend, is_month_end. Lag features: revenue_lag7 (same day last week), "
            "revenue_lag30 (same day last month). Rolling features: 7-day and 30-day "
            "rolling mean. These capture weekly cycles, monthly trends, and autocorrelation "
            "in the sales data."
        ),
    },
    {
        "category": "Machine Learning",
        "q": "How do you evaluate and compare multiple models automatically?",
        "a": (
            "All models are trained on the same train split, then scored by RMSE on the "
            "identical test split. The model with lowest RMSE is programmatically selected "
            "as best_model and used for future forecasting. We also report MAE, R², and MAPE "
            "so users can make informed choices based on their error tolerance preferences."
        ),
    },

    # ── NLP ───────────────────────────────────────────────────
    {
        "category": "NLP",
        "q": "What is TF-IDF and why use it for keyword extraction?",
        "a": (
            "TF-IDF (Term Frequency–Inverse Document Frequency) scores words by how often "
            "they appear in a document (TF) divided by how common they are across all "
            "documents (IDF). Words like 'the' appear everywhere so their IDF is low. Words "
            "like 'battery' appearing frequently only in negative reviews get high TF-IDF "
            "scores. This surface context-specific keywords rather than stopwords."
        ),
    },
    {
        "category": "NLP",
        "q": "Why is BERT better than a lexicon approach for sentiment analysis?",
        "a": (
            "Lexicons miss context. 'Not good' would score positive with a lexicon (sees 'good') "
            "but negative with BERT (understands negation). BERT is a transformer pretrained on "
            "billions of words — it understands sarcasm, context, and nuance. Our system tries "
            "BERT first and falls back to lexicon if BERT is unavailable, ensuring both accuracy "
            "and robustness."
        ),
    },
    {
        "category": "NLP",
        "q": "How does K-Means clustering group complaints?",
        "a": (
            "Each complaint is converted to a TF-IDF vector (100-dimensional). K-Means finds K "
            "cluster centroids that minimize within-cluster distance. Complaints close to the "
            "same centroid share similar vocabulary (e.g., 'battery', 'charging', 'drain'). "
            "The top TF-IDF terms from each centroid become the cluster theme label."
        ),
    },
    {
        "category": "NLP",
        "q": "How did you detect emotions — not just positive/negative?",
        "a": (
            "We built a domain-specific emotion lexicon mapping words to 5 emotions: "
            "happy, angry, frustrated, excited, neutral. Words like 'love', 'fantastic' → happy. "
            "'Hate', 'furious' → angry. 'Broken', 'useless' → frustrated. We count word overlaps "
            "per emotion and assign the highest-scoring emotion. For production, we'd fine-tune "
            "a BERT model on emotion-labelled data (GoEmotions dataset)."
        ),
    },

    # ── System Design ─────────────────────────────────────────
    {
        "category": "System Design",
        "q": "Why FastAPI instead of Flask or Django?",
        "a": (
            "FastAPI is async-native (via asyncio/uvicorn), which means it handles concurrent "
            "requests without blocking while ML models compute. It auto-generates OpenAPI/Swagger "
            "docs from type annotations. It's 2-3x faster than Flask in benchmarks. Pydantic "
            "integration gives us built-in request validation. Django is better for monolithic "
            "apps with templates — FastAPI is ideal for API-only microservices."
        ),
    },
    {
        "category": "System Design",
        "q": "Why SQLite for development and PostgreSQL for production?",
        "a": (
            "SQLite requires zero setup — it's a single file, perfect for local development. "
            "PostgreSQL handles concurrent writes, connection pooling, and scales to millions "
            "of records. SQLAlchemy abstracts both behind the same ORM so switching is just "
            "changing the DATABASE_URL environment variable — zero code changes."
        ),
    },
    {
        "category": "System Design",
        "q": "How does JWT authentication work in this system?",
        "a": (
            "On login, the server validates credentials, then creates a JWT token containing "
            "the user's ID and role, signed with a secret key using HS256. The client stores "
            "this token and sends it as 'Authorization: Bearer <token>' on every request. "
            "The server decodes and verifies the signature — no database lookup needed. "
            "Tokens expire after 24 hours (configurable). Role-based decorators then check "
            "if the user has sufficient permissions."
        ),
    },
    {
        "category": "System Design",
        "q": "How would you scale this to handle 100,000 users?",
        "a": (
            "Horizontally: run multiple FastAPI workers behind a load balancer (nginx/caddy). "
            "Cache ML predictions in Redis with a TTL — most users see the same forecast. "
            "Offload heavy ML training to background workers (Celery + Redis). Use PostgreSQL "
            "read replicas for dashboard queries. CDN for static assets. Containerize with "
            "Docker Swarm or Kubernetes for auto-scaling."
        ),
    },

    # ── Business / Practical ──────────────────────────────────
    {
        "category": "Business",
        "q": "How is this different from Excel-based forecasting?",
        "a": (
            "Excel forecasting is manual, static, and limited to simple trend lines. Our system "
            "automatically trains 5+ models, selects the best, incorporates complex features "
            "(lag, rolling averages, seasonality), handles thousands of products simultaneously, "
            "updates in real time, and integrates with NLP and inventory in one platform. "
            "Excel cannot detect fraud, cluster complaints, or power an AI chatbot."
        ),
    },
    {
        "category": "Business",
        "q": "What is the What-If Simulator actually computing?",
        "a": (
            "It applies empirically-derived elasticity coefficients from historical analysis. "
            "For example, historical data shows +20% ad spend correlates with +12% conversions. "
            "Price elasticity is computed from demand curves in the sales data. Festival "
            "multipliers come from comparing festival vs non-festival period averages. These "
            "multipliers combine linearly and are applied to the current forecast baseline."
        ),
    },
    {
        "category": "Business",
        "q": "How do you detect fraudulent orders?",
        "a": (
            "We use statistical anomaly detection rather than classification. "
            "For refund rate: z-score on rolling 7-day refund percentage — flag if > 3σ. "
            "For fake orders: group by device_id/IP and flag if > N orders in M minutes. "
            "For revenue crashes: compare day-of-week revenue to same day 4 weeks prior — "
            "flag if drop > 50%. No labelled fraud data needed — pure anomaly detection."
        ),
    },
    {
        "category": "Business",
        "q": "Can this system work for a real business immediately?",
        "a": (
            "Yes, with minor customization. Upload your sales CSV (date + revenue columns), "
            "upload customer reviews CSV, configure product and inventory data. The system "
            "auto-trains on your data, generates forecasts, and provides insights within "
            "minutes. For the chatbot, set OPENAI_API_KEY or GROQ_API_KEY and it answers "
            "questions about your specific business data."
        ),
    },
]


if __name__ == "__main__":
    from collections import defaultdict
    by_cat = defaultdict(list)
    for qa in VIVA_QA:
        by_cat[qa["category"]].append(qa)

    for cat, questions in by_cat.items():
        print(f"\n{'='*65}")
        print(f"  {cat.upper()}")
        print(f"{'='*65}")
        for i, qa in enumerate(questions, 1):
            print(f"\nQ{i}: {qa['q']}")
            print(f"A:  {qa['a']}")
