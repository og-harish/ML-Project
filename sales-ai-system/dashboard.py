"""
Streamlit dashboard for the Sales Prediction System with NLP-based insights.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from io import StringIO
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from sales_prediction_colab import (
    DEFAULT_SALES_CSV,
    TrainingResult,
    add_time_features,
    aggregate_daily_sales,
    generate_realtime_prediction_explanation,
    generate_nlp_insights,
    load_sales_data,
    normalize_sales_columns,
    train_sales_model,
)


CITY_COORDINATES = {
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Delhi": (28.7041, 77.1025),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Kolkata": (22.5726, 88.3639),
}


st.set_page_config(
    page_title="Sales Prediction + NLP Insights",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_sales_data_from_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = load_sales_data(DEFAULT_SALES_CSV)
    return normalize_sales_columns(raw)


def parse_api_sales_response(body: str, content_type: str) -> pd.DataFrame:
    if "text/csv" in content_type or body.lstrip().startswith(("date,", "Date,")):
        return pd.read_csv(StringIO(body))

    payload = json.loads(body)
    records = payload
    if isinstance(payload, dict):
        for key in ("data", "sales", "records", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if isinstance(records, dict) and {"columns", "data"} <= set(records):
            return pd.DataFrame(records["data"], columns=records["columns"])

    if not isinstance(records, list):
        raise ValueError("Sales API response must be CSV, a JSON list, or a JSON object with data/sales/records/results/items.")
    return pd.DataFrame(records)


def fetch_realtime_sales_data(api_url: str, api_key: str) -> pd.DataFrame:
    headers = {
        "Accept": "application/json,text/csv",
        "User-Agent": "sales-prediction-dashboard/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
        headers["x-api-key"] = api_key
        headers["X-RapidAPI-Key"] = api_key

    request = Request(api_url, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except URLError as exc:
        raise ValueError(f"Unable to load real-time sales API: {exc.reason}") from exc

    raw = parse_api_sales_response(body, content_type)
    return normalize_sales_columns(raw)


@st.cache_resource(show_spinner=False)
def cached_model(serialized_csv: str, forecast_days: int) -> tuple[TrainingResult, object, pd.DataFrame]:
    df = pd.read_json(StringIO(serialized_csv))
    df["date"] = pd.to_datetime(df["date"])
    result, model = train_sales_model(df, forecast_days=forecast_days)
    return result, model, df


def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "latitude" not in out.columns:
        out["latitude"] = out["region"].map(lambda region: CITY_COORDINATES.get(str(region), (20.5937, 78.9629))[0])
    if "longitude" not in out.columns:
        out["longitude"] = out["region"].map(lambda region: CITY_COORDINATES.get(str(region), (20.5937, 78.9629))[1])
    return out


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def render_overview(df: pd.DataFrame, result: TrainingResult) -> None:
    revenue = df["revenue"].sum()
    profit = df["profit"].sum()
    units = df["units_sold"].sum()
    margin = profit / revenue * 100 if revenue else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Revenue", f"₹{revenue:,.0f}")
    with c2:
        metric_card("Total Profit", f"₹{profit:,.0f}")
    with c3:
        metric_card("Units Sold", f"{units:,.0f}")
    with c4:
        metric_card("Profit Margin", f"{margin:.1f}%")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Best Forecast Model", result.model_name)
    with c2:
        metric_card("RMSE", f"₹{result.metrics['rmse']:,.0f}")
    with c3:
        metric_card("Forecast Revenue", f"₹{result.total_predicted_revenue:,.0f}")


def render_charts(df: pd.DataFrame, result: TrainingResult) -> None:
    daily = df.groupby("date", as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
    daily["rolling_7_day"] = daily["revenue"].rolling(7, min_periods=1).mean()

    st.subheader("Sales Trend and Forecast")
    trend_chart = px.line(daily, x="date", y=["revenue", "rolling_7_day"], title="Revenue with 7-Day Rolling Average")
    st.plotly_chart(trend_chart, use_container_width=True)

    forecast_df = pd.DataFrame(result.forecast)
    forecast_chart = px.line(
        forecast_df,
        x="date",
        y=["predicted_revenue", "lower_bound", "upper_bound"],
        title="Future Revenue Forecast",
    )
    st.plotly_chart(forecast_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        category = df.groupby("product_category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(category, x="product_category", y="revenue", title="Revenue by Category"), use_container_width=True)
    with right:
        pnl = df.groupby("region", as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        st.plotly_chart(px.bar(pnl, x="region", y=["revenue", "profit"], barmode="group", title="Profit and Loss by Region"), use_container_width=True)


def render_geo_map(df: pd.DataFrame) -> None:
    st.subheader("Real-Time Region Health Map")
    mapped = add_coordinates(df)
    regional = (
        mapped.groupby(["region", "country", "latitude", "longitude"], as_index=False)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units_sold=("units_sold", "sum"))
        .sort_values("revenue", ascending=False)
    )
    fig = px.scatter_geo(
        regional,
        lat="latitude",
        lon="longitude",
        size="revenue",
        color="profit",
        hover_name="region",
        hover_data={"revenue": ":,.0f", "profit": ":,.0f", "units_sold": ":,.0f"},
        projection="natural earth",
        title="Sales Hotspots and Cold Zones",
    )
    fig.update_geos(fitbounds="locations", visible=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(regional, use_container_width=True, hide_index=True)


def render_nlp(df: pd.DataFrame, result: TrainingResult, api_key: str | None) -> None:
    st.subheader("NLP-Based Insight Extraction")
    insights = generate_nlp_insights(df, result, api_key=api_key)

    sentiment = insights.get("sentiment", {})
    if isinstance(sentiment, dict):
        sentiment_df = pd.DataFrame({"sentiment": sentiment.keys(), "score": sentiment.values()})
        st.plotly_chart(px.bar(sentiment_df, x="sentiment", y="score", title="Customer Sentiment"), use_container_width=True)

    st.markdown("### AI Narrative")
    st.info(str(insights.get("narrative", "No narrative generated.")))

    st.markdown("### Recommendations")
    for item in insights.get("recommendations", []):
        st.write(f"- {item}")

    anomalies = insights.get("anomalies", [])
    if anomalies:
        st.markdown("### Anomaly Alerts")
        for alert in anomalies:
            st.warning(str(alert))

    if insights.get("gemini_error"):
        st.caption("Gemini was unavailable for this run, so local NLP fallback insights are displayed.")

    with st.expander("Raw NLP JSON"):
        st.json(json.loads(json.dumps(insights, default=str)))


def build_realtime_features(
    df: pd.DataFrame,
    prediction_date: pd.Timestamp,
    units_sold: int,
    discount_pct: float,
    historical_revenue: float,
    profit_margin: float,
) -> pd.DataFrame:
    history = aggregate_daily_sales(df)
    probe = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "date": [prediction_date],
                    "revenue": [historical_revenue],
                    "units_sold": [units_sold],
                    "discount_pct": [discount_pct],
                    "profit": [historical_revenue * profit_margin],
                }
            ),
        ],
        ignore_index=True,
    )
    return add_time_features(probe).tail(1)


def render_prediction_form(df: pd.DataFrame, result: TrainingResult, model: object, api_key: str | None) -> None:
    st.sidebar.subheader("Live Prediction Inputs")
    region = st.sidebar.selectbox("Region", sorted(df["region"].unique()))
    category = st.sidebar.selectbox("Product category", sorted(df["product_category"].unique()))
    days = st.sidebar.slider("Forecast days", 7, 180, len(result.forecast), step=7)
    region_revenue = df.loc[df["region"] == region, "revenue"].sum()
    category_revenue = df.loc[df["product_category"] == category, "revenue"].sum()
    scaled_prediction = result.avg_daily_revenue * days * max(region_revenue / max(df["revenue"].sum(), 1), 0.01)

    st.sidebar.metric("Region forecast", f"₹{scaled_prediction:,.0f}")
    st.sidebar.caption(f"Based on {region} region share and the trained {result.model_name} model.")
    st.sidebar.metric("Category historical revenue", f"₹{category_revenue:,.0f}")

    st.sidebar.divider()
    st.sidebar.subheader("Real-Time Sale Predictor")
    prediction_date = st.sidebar.date_input("Sale date", pd.to_datetime(df["date"]).max().date() + pd.Timedelta(days=1))
    input_units = st.sidebar.number_input("Units to sell", min_value=1, value=10, step=1)
    input_discount = st.sidebar.slider("Discount %", 0, 90, int(round(float(df["discount_pct"].mean()))))
    segment = df[(df["region"] == region) & (df["product_category"] == category)]
    if segment.empty:
        segment = df[df["region"] == region]
    baseline_revenue = float(segment["revenue"].sum() / max(segment["units_sold"].sum(), 1))
    estimated_revenue = baseline_revenue * input_units * (1 - input_discount / 100)
    profit_margin = float(df["profit"].sum() / max(df["revenue"].sum(), 1))
    live_features = build_realtime_features(
        df,
        pd.Timestamp(prediction_date),
        int(input_units),
        float(input_discount),
        estimated_revenue,
        profit_margin,
    )
    predicted_sale = max(float(model.predict(live_features[result.feature_columns])[0]), 0.0)
    expected_profit = predicted_sale * profit_margin
    prediction_payload = {
        "region": region,
        "category": category,
        "sale_date": pd.Timestamp(prediction_date).date().isoformat(),
        "units_sold": int(input_units),
        "discount_pct": float(input_discount),
        "baseline_revenue_per_unit": round(baseline_revenue, 2),
        "predicted_revenue": round(predicted_sale, 2),
        "expected_profit": round(expected_profit, 2),
        "model": result.model_name,
    }

    st.session_state["live_prediction"] = prediction_payload
    st.session_state["live_prediction_explanation"] = generate_realtime_prediction_explanation(prediction_payload, api_key)
    st.sidebar.metric("Predicted sale", f"₹{predicted_sale:,.0f}")
    st.sidebar.metric("Expected profit", f"₹{expected_profit:,.0f}")


def render_realtime_prediction() -> None:
    prediction = st.session_state.get("live_prediction")
    if not prediction:
        st.info("Use the sidebar inputs to generate a real-time sale prediction.")
        return

    st.subheader("Real-Time Input Sale Prediction")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Predicted Sale", f"₹{float(prediction['predicted_revenue']):,.0f}")
    with c2:
        metric_card("Expected Profit", f"₹{float(prediction['expected_profit']):,.0f}")
    with c3:
        metric_card("Input Units", f"{int(prediction['units_sold']):,}")
    with c4:
        metric_card("Discount", f"{float(prediction['discount_pct']):.0f}%")

    prediction_df = pd.DataFrame(
        [
            {"metric": "Baseline estimate", "amount": prediction["baseline_revenue_per_unit"] * prediction["units_sold"]},
            {"metric": "Model predicted sale", "amount": prediction["predicted_revenue"]},
            {"metric": "Expected profit", "amount": prediction["expected_profit"]},
        ]
    )
    st.plotly_chart(px.bar(prediction_df, x="metric", y="amount", title="Real-Time Prediction Visualization"), use_container_width=True)
    st.info(st.session_state.get("live_prediction_explanation", "Prediction explanation unavailable."))
    st.json(prediction)


def main() -> None:
    st.title("Sales Prediction System with NLP-Based Insight Extraction")
    st.caption("Upload a CSV, preprocess sales data, train an ML forecast model, and generate real-time business insights.")

    with st.sidebar:
        st.header("Data")
        realtime_api_url = st.text_input("Real-time sales API URL (optional)", help="Returns sales rows as CSV or JSON list/object.")
        dataset_api_key = st.text_input("Dataset API key (optional)", type="password", help="Used only to fetch real-time sales data from the API URL.")
        api_key = st.text_input("Gemini API key (optional)", type="password", help="Used for AI-powered NLP and prediction explanations. Leave blank to use local fallback.")
        uploaded_file = st.file_uploader("Upload e-commerce sales CSV", type=["csv"])
        forecast_days = st.slider("Forecast window", 14, 180, 90, step=7)
        st.caption("Expected columns: date, region/city, product_category/category, units_sold/quantity, revenue.")

    if realtime_api_url.strip():
        try:
            df = fetch_realtime_sales_data(realtime_api_url.strip(), dataset_api_key)
            st.sidebar.success("Loaded real-time sales API data.")
        except ValueError as exc:
            st.sidebar.error(str(exc))
            st.stop()
    else:
        df = get_sales_data_from_file(uploaded_file)
    serialized = df.to_json(date_format="iso")
    with st.spinner("Training forecast model and generating insights..."):
        result, model, model_df = cached_model(serialized, forecast_days)

    render_prediction_form(model_df, result, model, api_key or None)

    tab_overview, tab_realtime, tab_map, tab_charts, tab_nlp, tab_data = st.tabs(
        ["Overview", "Real-Time Prediction", "World Map", "Forecast Charts", "NLP Insights", "Data"]
    )
    with tab_overview:
        render_overview(model_df, result)
    with tab_realtime:
        render_realtime_prediction()
    with tab_map:
        render_geo_map(model_df)
    with tab_charts:
        render_charts(model_df, result)
    with tab_nlp:
        render_nlp(model_df, result, api_key or None)
    with tab_data:
        st.dataframe(model_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Forecast CSV",
            pd.DataFrame(result.forecast).to_csv(index=False),
            file_name="sales_forecast.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
