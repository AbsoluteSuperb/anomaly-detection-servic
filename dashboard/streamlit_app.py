"""Streamlit dashboard for the Anomaly Detection Service.

Talks to the FastAPI backend via HTTP.
Launch: streamlit run dashboard/streamlit_app.py
Requires the API to be running at API_URL (default http://localhost:8000).
"""

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Anomaly Detection", page_icon="📊", layout="wide")

# ── Sidebar navigation ──────────────────────────────────────────────────────

page = st.sidebar.radio("Navigation", ["Overview", "Metric Explorer", "Detection", "About"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Start the server with `make run`.")
        st.stop()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None


def api_post(path: str, json: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=json, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Start the server with `make run`.")
        st.stop()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Overview")

    health = api_get("/health")
    if not health:
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", health["status"].upper())
    col2.metric("Total Days", health["total_days"])
    col3.metric("Date Range", health["date_range"])

    st.divider()

    # Metric summary cards
    metrics = api_get("/metrics")
    if metrics:
        st.subheader("Metric Summary")
        cols = st.columns(len(metrics))
        for col, m in zip(cols, metrics):
            col.metric(
                label=m["metric_name"],
                value=f"{m['last_value']:,.0f}",
                delta=f"mean {m['mean']:,.0f}",
            )

    st.divider()

    # Anomaly counts for last 7 / 30 days
    st.subheader("Recent Anomalies")

    anomalies = api_get("/anomalies")
    if anomalies is not None and len(anomalies) > 0:
        df_a = pd.DataFrame(anomalies)
        df_a["date"] = pd.to_datetime(df_a["date"])
        last_date = df_a["date"].max()

        c1, c2 = st.columns(2)
        last_7 = df_a[df_a["date"] >= last_date - pd.Timedelta(days=7)]
        last_30 = df_a[df_a["date"] >= last_date - pd.Timedelta(days=30)]
        c1.metric("Anomalies (last 7 days)", len(last_7))
        c2.metric("Anomalies (last 30 days)", len(last_30))

        # Severity breakdown
        if not df_a.empty:
            sev = df_a["severity"].value_counts()
            st.bar_chart(sev)
    else:
        st.info("No anomalies cached yet. Go to **Detection** and run a scan first.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: Metric Explorer
# ════════════════════════════════════════════════════════════════════════════

elif page == "Metric Explorer":
    st.title("Metric Explorer")

    health = api_get("/health")
    if not health:
        st.stop()

    metric = st.selectbox("Metric", health["available_metrics"])

    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=pd.Timestamp("2009-12-01"))
    end = col2.date_input("End date", value=pd.Timestamp("2011-12-09"))

    ts = api_get(f"/metrics/{metric}", {"start_date": str(start), "end_date": str(end)})
    if ts:
        df_ts = pd.DataFrame({"date": pd.to_datetime(ts["dates"]), "value": ts["values"]})

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ts["date"], y=df_ts["value"],
                                  mode="lines", name=metric,
                                  line=dict(color="steelblue", width=1.5)))

        # Overlay anomalies
        anomalies = api_get("/anomalies", {"metric": metric,
                                            "start_date": str(start),
                                            "end_date": str(end)})
        if anomalies:
            severity_styles = [("critical", "#DC3545", "x"), ("warning", "#FFC107", "diamond")]
            for sev, color, sym in severity_styles:
                pts = [a for a in anomalies if a["severity"] == sev]
                if pts:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime([a["date"] for a in pts]),
                        y=[a["value"] for a in pts],
                        mode="markers",
                        name=sev.title(),
                        marker=dict(color=color, size=10, symbol=sym,
                                     line=dict(width=1, color="black")),
                        hovertemplate="<br>".join([
                            "<b>%{x|%Y-%m-%d}</b>",
                            "Value: %{y:,.0f}",
                            "<extra></extra>",
                        ]),
                    ))

        fig.update_layout(template="plotly_white", height=450,
                          xaxis_title="Date", yaxis_title=metric,
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly table
        if anomalies:
            st.subheader(f"Anomalies ({len(anomalies)})")
            df_anom = pd.DataFrame(anomalies)
            sev_filter = st.multiselect("Severity", ["warning", "critical"],
                                         default=["warning", "critical"])
            df_anom = df_anom[df_anom["severity"].isin(sev_filter)]
            st.dataframe(df_anom, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: Detection
# ════════════════════════════════════════════════════════════════════════════

elif page == "Detection":
    st.title("Run Detection")

    health = api_get("/health")
    if not health:
        st.stop()

    col1, col2 = st.columns(2)
    detector = col1.selectbox("Detector", health["available_detectors"])
    metric_choice = col2.selectbox("Metric", ["All metrics"] + health["available_metrics"])

    col3, col4 = st.columns(2)
    start = col3.date_input("Start date", value=None, key="det_start")
    end = col4.date_input("End date", value=None, key="det_end")

    if st.button("Run Detection", type="primary"):
        payload = {"detector": detector}
        if metric_choice != "All metrics":
            payload["metric_name"] = metric_choice
        if start:
            payload["start_date"] = str(start)
        if end:
            payload["end_date"] = str(end)

        with st.spinner(f"Running {detector} detection..."):
            result = api_post("/detect", payload)

        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", result["total_anomalies"])
            c2.metric("Warnings", result["warnings"])
            c3.metric("Critical", result["criticals"])
            c4.metric("Time (ms)", f"{result['detection_time_ms']:.0f}")

            if result["anomalies"]:
                df = pd.DataFrame(result["anomalies"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.success("No anomalies detected.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: About
# ════════════════════════════════════════════════════════════════════════════

elif page == "About":
    st.title("About")

    st.markdown("""
## Detection Methods

### Z-Score (Rolling)
- **How it works:** Computes `(x - mean) / std` using a rolling window (default 30 days).
- **Pros:** Fast, interpretable, low resource usage.
- **Cons:** Assumes normal distribution; does not account for seasonality.
- **Thresholds:** |z| > 2 = warning, |z| > 3 = critical.

### IQR (Rolling)
- **How it works:** Computes Q1/Q3 on a rolling window; flags points outside `Q +/- k * IQR`.
- **Pros:** Robust to outliers in training data (unlike Z-score).
- **Cons:** No seasonality awareness.
- **Thresholds:** 1.5x IQR = warning, 3x IQR = critical.

### Prophet
- **How it works:** Facebook's time-series model with trend + weekly/yearly seasonality + holidays.
  Points outside the forecast uncertainty interval are flagged.
- **Pros:** Handles seasonality, trend, and holidays automatically.
- **Cons:** Slow to fit (~5-10s per metric); requires enough history.
- **Thresholds:** Outside 95% CI = warning, outside 99% CI = critical.

### Isolation Forest
- **How it works:** Multivariate tree-based model. Feeds revenue, orders, avg_check, and
  unique_customers simultaneously to catch joint anomalies invisible in univariate analysis.
- **Pros:** Finds multivariate anomalies (e.g., normal revenue but unusual order/check ratio).
- **Cons:** Less interpretable; requires tuning `contamination` parameter.
- **Thresholds:** Based on anomaly score distance.

### Ensemble
- **How it works:** Runs Z-Score + IQR + Isolation Forest; applies majority voting.
- **Rules:** 1 vote = ignored, 2 votes = warning, 3+ votes = critical.
- **Pros:** Reduces false positives significantly.

---

## References
- **Z-Score / IQR:** Standard statistical methods.
  [NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/)
- **Prophet:** Taylor & Letham (2018). *Forecasting at Scale*.
  [PeerJ Preprints](https://doi.org/10.7287/peerj.preprints.3190v2)
- **Isolation Forest:** Liu, Ting & Zhou (2008). *Isolation Forest*.
  [IEEE ICDM](https://doi.org/10.1109/ICDM.2008.17)

---

## Architecture
- **Backend:** FastAPI + Pydantic + pandas
- **Detectors:** scikit-learn, Prophet, custom statistical detectors
- **Dashboard:** Streamlit (this page) communicating with the API via HTTP
- **Source:** [GitHub Repository](https://github.com/)
    """)
