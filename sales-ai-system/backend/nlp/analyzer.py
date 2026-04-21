"""
NLP Customer Intelligence Pipeline
Sentiment analysis, emotion detection, keyword extraction,
complaint clustering, and product satisfaction scoring.
"""

import re
import json
from collections import Counter
from typing import List, Dict, Any

import numpy as np

# NLTK (always available)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    HAS_NLTK = True
except Exception:
    STOP_WORDS = {"the", "a", "an", "is", "it", "in", "on", "and", "or", "but"}
    lemmatizer = None
    HAS_NLTK = False

# sklearn TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Transformers (optional, heavy)
try:
    from transformers import pipeline as hf_pipeline
    _sentiment_pipe = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


# ─── Lexicons ──────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "excellent", "amazing", "great", "love", "fantastic", "perfect", "best",
    "wonderful", "awesome", "superb", "outstanding", "brilliant", "good",
    "happy", "satisfied", "pleased", "recommend", "fast", "quality",
}
NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "hate", "worst", "bad", "poor",
    "disappointing", "disappointed", "broken", "damaged", "fake", "slow",
    "refund", "return", "complaint", "useless", "waste", "fraud",
}
EMOTION_LEXICON = {
    "happy":      {"love", "excellent", "happy", "amazing", "fantastic", "great", "satisfied"},
    "angry":      {"hate", "terrible", "horrible", "furious", "angry", "worst", "unacceptable"},
    "frustrated": {"disappointed", "frustrating", "broken", "useless", "slow", "waste"},
    "excited":    {"excited", "awesome", "wow", "superb", "incredible", "mind-blowing"},
    "neutral":    set(),
}


# ─── Text Preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if HAS_NLTK:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
        return " ".join(tokens)
    return text


# ─── Sentiment ─────────────────────────────────────────────────────────────────

def classify_sentiment(text: str) -> Dict[str, Any]:
    """
    Returns sentiment using best available method:
    BERT (transformers) → TF-IDF model → lexicon fallback.
    """
    clean = preprocess(text)
    words = set(clean.split())

    if HAS_TRANSFORMERS:
        try:
            result = _sentiment_pipe(text[:512])[0]
            label = result["label"].lower()
            score = round(result["score"], 4)
            sentiment = "positive" if label == "positive" else "negative"
            return {"sentiment": sentiment, "score": score, "method": "BERT"}
        except Exception:
            pass

    # Lexicon fallback
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    if pos_count > neg_count:
        score = min(0.5 + (pos_count - neg_count) * 0.1, 0.99)
        return {"sentiment": "positive", "score": round(score, 4), "method": "lexicon"}
    elif neg_count > pos_count:
        score = min(0.5 + (neg_count - pos_count) * 0.1, 0.99)
        return {"sentiment": "negative", "score": round(score, 4), "method": "lexicon"}
    else:
        return {"sentiment": "neutral", "score": 0.5, "method": "lexicon"}


def detect_emotion(text: str) -> str:
    clean = preprocess(text)
    words = set(clean.split())
    scores = {emotion: len(words & lexicon_words) for emotion, lexicon_words in EMOTION_LEXICON.items() if lexicon_words}
    if not scores or max(scores.values()) == 0:
        return "neutral"
    return max(scores, key=scores.get)


# ─── Keyword Extraction ────────────────────────────────────────────────────────

def extract_keywords(texts: List[str], top_n: int = 20) -> List[Dict]:
    if not texts:
        return []
    if HAS_SKLEARN:
        vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            feature_names = vectorizer.get_feature_names_out()
            top_indices = scores.argsort()[::-1][:top_n]
            return [{"keyword": feature_names[i], "score": round(float(scores[i]), 4)} for i in top_indices]
        except Exception:
            pass
    # Fallback: word frequency
    all_words = []
    for text in texts:
        all_words.extend([w for w in preprocess(text).split() if len(w) > 3])
    counter = Counter(all_words)
    return [{"keyword": w, "score": round(c / len(all_words), 4)} for w, c in counter.most_common(top_n)]


# ─── Complaint Clustering ──────────────────────────────────────────────────────

def cluster_complaints(texts: List[str], n_clusters: int = 5) -> List[Dict]:
    if not texts or not HAS_SKLEARN:
        return [{"cluster": 0, "theme": "General complaints", "count": len(texts), "examples": texts[:3]}]
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    try:
        X = vectorizer.fit_transform(texts)
        k = min(n_clusters, len(texts))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        feature_names = vectorizer.get_feature_names_out()
        clusters = []
        for i in range(k):
            cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
            center = km.cluster_centers_[i]
            top_terms = [feature_names[idx] for idx in center.argsort()[::-1][:3]]
            clusters.append({
                "cluster": i,
                "theme": " / ".join(top_terms),
                "count": len(cluster_texts),
                "examples": cluster_texts[:2],
            })
        return sorted(clusters, key=lambda x: x["count"], reverse=True)
    except Exception:
        return [{"cluster": 0, "theme": "All complaints", "count": len(texts), "examples": texts[:3]}]


# ─── Main Analysis Function ────────────────────────────────────────────────────

def analyze_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Full NLP analysis pipeline for a list of review dicts.
    Each review: {"text": str, "product_id": str, "product_name": str, "rating": float}

    Returns comprehensive NLP intelligence report.
    """
    if not reviews:
        return {"error": "No reviews provided"}

    texts = [r.get("text", "") for r in reviews]
    analyzed = []
    sentiment_counts = Counter()
    emotion_counts = Counter()
    product_sentiments: Dict[str, List[float]] = {}

    for rev in reviews:
        text = rev.get("text", "")
        if not text.strip():
            continue

        sent = classify_sentiment(text)
        emotion = detect_emotion(text)
        sentiment_counts[sent["sentiment"]] += 1
        emotion_counts[emotion] += 1

        pid = rev.get("product_id", "unknown")
        score = 1.0 if sent["sentiment"] == "positive" else (-1.0 if sent["sentiment"] == "negative" else 0.0)
        product_sentiments.setdefault(pid, []).append(score)

        analyzed.append({
            "text": text[:200],
            "product_id": pid,
            "product_name": rev.get("product_name", "Unknown"),
            "rating": rev.get("rating"),
            "sentiment": sent["sentiment"],
            "sentiment_score": sent["score"],
            "emotion": emotion,
            "method": sent["method"],
        })

    total = len(analyzed)
    negative_texts = [r["text"] for r in analyzed if r["sentiment"] == "negative"]

    # Product satisfaction scores (0-100)
    product_scores = {}
    for pid, scores in product_sentiments.items():
        avg = np.mean(scores)
        product_scores[pid] = round((avg + 1) / 2 * 100, 1)

    # Trending issues from negative reviews
    complaint_clusters = cluster_complaints(negative_texts, n_clusters=4)

    # Overall NPS-style score
    pos_pct = sentiment_counts["positive"] / total * 100 if total else 0
    neg_pct = sentiment_counts["negative"] / total * 100 if total else 0
    overall_score = round(pos_pct - neg_pct, 1)

    return {
        "total_reviews": total,
        "sentiment_distribution": {
            "positive": sentiment_counts["positive"],
            "neutral":  sentiment_counts["neutral"],
            "negative": sentiment_counts["negative"],
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
        },
        "emotion_distribution": dict(emotion_counts),
        "overall_sentiment_score": overall_score,
        "keywords": extract_keywords(texts, top_n=15),
        "complaint_clusters": complaint_clusters,
        "product_satisfaction": product_scores,
        "analyzed_reviews": analyzed[:50],  # cap for payload size
        "trending_issues": [c["theme"] for c in complaint_clusters[:3]],
    }
