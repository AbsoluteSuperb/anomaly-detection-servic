from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.detection.base import Anomaly


def create_metric_plot(
    metric_name: str,
    series: pd.Series,
    anomalies: list[Anomaly],
    forecast_lower: pd.Series | None = None,
    forecast_upper: pd.Series | None = None,
    weekend_mask: pd.Series | None = None,
) -> str:
    """Build an interactive Plotly chart and return it as an HTML string.

    Parameters
    ----------
    metric_name : str
        Display name for the y-axis / title.
    series : pd.Series
        The metric time series (DatetimeIndex).
    anomalies : list[Anomaly]
        Anomaly objects to plot (supports WARNING + CRITICAL severity).
    forecast_lower / forecast_upper : pd.Series, optional
        Prophet-style confidence band to shade as the "normal corridor".
    weekend_mask : pd.Series[bool], optional
        True for weekend dates -> grey vertical shading.
    """
    fig = go.Figure()

    # --- Weekend shading ---
    if weekend_mask is not None:
        weekend_dates = weekend_mask[weekend_mask].index
        for d in weekend_dates:
            fig.add_vrect(
                x0=d,
                x1=d + pd.Timedelta(days=1),
                fillcolor="lightgray",
                opacity=0.25,
                layer="below",
                line_width=0,
            )

    # --- Normal corridor (Prophet forecast band) ---
    if forecast_lower is not None and forecast_upper is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_upper.index,
                y=forecast_upper.values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_lower.index,
                y=forecast_lower.values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(135,206,250,0.2)",
                name="Normal corridor",
                hoverinfo="skip",
            )
        )

    # --- Main metric line ---
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=metric_name,
            line=dict(color="steelblue", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        )
    )

    # --- Anomaly markers (split by severity) ---
    warnings = [a for a in anomalies if a.severity == "warning"]
    criticals = [a for a in anomalies if a.severity == "critical"]

    for group, color, symbol, label in [
        (warnings, "#FFC107", "diamond", "Warning"),
        (criticals, "#DC3545", "x", "Critical"),
    ]:
        if not group:
            continue

        dates = pd.to_datetime([a.date for a in group])
        values = [a.value for a in group]
        hovers = [
            (
                f"<b>{a.date}</b><br>"
                f"Value: {a.value:,.0f}<br>"
                f"Expected: {a.expected:,.0f}<br>"
                f"Deviation: {a.deviation_pct:+.1f}%<br>"
                f"Detector: {a.detector}<br>"
                f"{a.details}"
            )
            for a in group
        ]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="markers",
                name=label,
                marker=dict(color=color, size=10, symbol=symbol, line=dict(width=1, color="black")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hovers,
            )
        )

    # --- Layout ---
    fig.update_layout(
        title=f"{metric_name} - Anomaly Detection",
        xaxis_title="Date",
        yaxis_title=metric_name,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=40),
    )

    return fig.to_html(include_plotlyjs="cdn", full_html=True)
