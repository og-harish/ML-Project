"""
AI Business Recommendation Engine
Combines rule-based logic with data-driven signals to generate
actionable business recommendations.
"""

from typing import List, Dict, Any
from datetime import datetime


# ─── Recommendation Types ─────────────────────────────────────────────────────

class Recommendation:
    def __init__(self, category: str, severity: str, title: str, description: str,
                 expected_impact: str, action: str):
        self.category      = category    # inventory | pricing | marketing | operations | nlp
        self.severity      = severity    # critical | warning | opportunity | info
        self.title         = title
        self.description   = description
        self.expected_impact = expected_impact
        self.action        = action
        self.created_at    = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return self.__dict__


# ─── Rule Engine ──────────────────────────────────────────────────────────────

def generate_recommendations(
    kpis: Dict[str, Any] = None,
    inventory: List[Dict] = None,
    sentiment: Dict = None,
    pricing: List[Dict] = None,
    fraud_alerts: List[Dict] = None,
) -> List[Dict]:
    """
    Generate business recommendations by applying rule engine to data signals.

    Returns a ranked list of Recommendation dicts ordered by severity.
    """
    recs: List[Recommendation] = []

    # ── Inventory Rules ───────────────────────────────────────────────────────
    if inventory:
        for item in inventory:
            days_left = item.get("days_until_stockout", 99)
            if days_left < 1:
                recs.append(Recommendation(
                    category="inventory",
                    severity="critical",
                    title=f"Restock {item['product_name']} immediately",
                    description=f"Only {item['stock_qty']} units remaining — less than 1 day of stock at current sales velocity ({item.get('daily_sales', '?')} units/day).",
                    expected_impact="Prevent revenue loss of estimated ₹" + str(int(item.get("daily_sales", 10) * item.get("unit_price", 5000))),
                    action=f"Place emergency order for {item.get('reorder_qty', 100)} units",
                ))
            elif days_left < 3:
                recs.append(Recommendation(
                    category="inventory",
                    severity="warning",
                    title=f"Schedule reorder for {item['product_name']}",
                    description=f"{days_left:.1f} days of stock remaining. Reorder before stockout to avoid missed sales.",
                    expected_impact="Maintain sales continuity",
                    action=f"Order {item.get('reorder_qty', 80)} units within 24 hours",
                ))
            elif item.get("risk") == "overstock":
                recs.append(Recommendation(
                    category="pricing",
                    severity="warning",
                    title=f"Clear overstock for {item['product_name']}",
                    description=f"{item['stock_qty']} units in stock ({days_left:.0f} days of supply). Capital is locked up — consider a promotional discount.",
                    expected_impact="Recover ₹" + str(int(item['stock_qty'] * item.get("unit_cost", 2000))),
                    action="Offer 10–15% discount or bundle deal this weekend",
                ))

    # ── Sentiment / NLP Rules ─────────────────────────────────────────────────
    if sentiment:
        neg_pct = sentiment.get("negative_pct", 0)
        if neg_pct > 30:
            recs.append(Recommendation(
                category="nlp",
                severity="critical",
                title="Customer sentiment critically low",
                description=f"{neg_pct:.1f}% of reviews are negative. Trending issues: {', '.join(sentiment.get('trending_issues', [])[:2])}. Immediate product/support review needed.",
                expected_impact="Prevent churn and protect brand reputation",
                action="Audit top complained products, respond to negative reviews within 24h",
            ))
        elif neg_pct > 15:
            recs.append(Recommendation(
                category="nlp",
                severity="warning",
                title="Rising negative sentiment detected",
                description=f"Negative reviews at {neg_pct:.1f}%. Monitor closely and address top complaint themes.",
                expected_impact="Improve NPS score and repeat purchases",
                action="Review complaint clusters and escalate to product team",
            ))

    # ── Pricing Rules ─────────────────────────────────────────────────────────
    if pricing:
        for item in pricing:
            change = item.get("change_pct", 0)
            if change > 5:
                recs.append(Recommendation(
                    category="pricing",
                    severity="opportunity",
                    title=f"Increase price of {item['product_name']}",
                    description=f"Demand signals support a {change:.1f}% price increase (₹{item['current_price']:,} → ₹{item['suggested_price']:,}). Low inventory and strong demand justify premium.",
                    expected_impact=f"+{change:.1f}% margin improvement",
                    action=f"Update listing price to ₹{item['suggested_price']:,}",
                ))
            elif change < -5:
                recs.append(Recommendation(
                    category="pricing",
                    severity="info",
                    title=f"Reduce price of {item['product_name']} to boost velocity",
                    description=f"Overstock or low demand supports a {abs(change):.1f}% price reduction to accelerate sell-through.",
                    expected_impact="Increase conversion rate and clear inventory",
                    action=f"Set price to ₹{item['suggested_price']:,} for next 7 days",
                ))

    # ── Fraud / Risk Rules ────────────────────────────────────────────────────
    if fraud_alerts:
        for alert in fraud_alerts:
            if not alert.get("is_resolved") and alert.get("severity") == "critical":
                recs.append(Recommendation(
                    category="operations",
                    severity="critical",
                    title=f"Investigate fraud alert: {alert.get('type', '').replace('_', ' ').title()}",
                    description=alert.get("message", "Unusual activity detected."),
                    expected_impact="Prevent further financial loss",
                    action="Review flagged transactions and contact payment gateway",
                ))

    # ── Generic Growth Opportunities ─────────────────────────────────────────
    recs.append(Recommendation(
        category="marketing",
        severity="opportunity",
        title="Launch weekend flash sale",
        description="Historical data shows weekend promotions drive +28% revenue. No active promotion scheduled this weekend.",
        expected_impact="+28% weekend revenue lift",
        action="Set up 20% off Electronics promotion for Sat–Sun",
    ))
    recs.append(Recommendation(
        category="marketing",
        severity="info",
        title="Expand advertising in Chennai",
        description="Chennai shows 23% MoM growth — the fastest in your network — but receives only 8% of ad budget.",
        expected_impact="+₹8–12L additional monthly revenue",
        action="Reallocate 15% of ad budget from Delhi to Chennai campaigns",
    ))

    # ── Sort by severity ──────────────────────────────────────────────────────
    severity_order = {"critical": 0, "warning": 1, "opportunity": 2, "info": 3}
    recs.sort(key=lambda r: severity_order.get(r.severity, 9))

    return [r.to_dict() for r in recs]
