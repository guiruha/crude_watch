"""The CrudeWatch black & green Plotly theme, extracted from the notebook charts.

Import requires plotly (``pip install -e '.[plots]'``).
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Palette.
GREEN = "#00E676"
BLACK = "#000000"
PANEL = "#0A0A0A"
GRID = "#1A1A1A"
TEXT = "#E0E0E0"
GLOW = "rgba(0, 230, 118, 0.08)"  # faint green fill under the line


def _rgba(hex_color: str, alpha: float) -> str:
    """Translucent ``rgba(...)`` string from a ``#RRGGBB`` hex colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def base_layout(title: str, y_title: str, x_title: str = "Date", accent: str = GREEN) -> dict:
    """Shared dark layout for a single-series time chart. ``accent`` colours the
    title and axis lines (pass a corporate green to re-theme)."""
    return dict(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color=accent, size=20)),
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=TEXT, family="Arial"),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=70, b=50),
        xaxis=dict(
            title=x_title, gridcolor=GRID,
            showline=True, linecolor=accent, linewidth=1,
            rangeslider=dict(visible=True, bgcolor=PANEL),
        ),
        yaxis=dict(
            title=y_title, gridcolor=GRID,
            showline=True, linecolor=accent, linewidth=1, zeroline=False,
        ),
    )


def line_figure(
    series,
    title: str,
    y_title: str,
    x_col: str = "date",
    y_col: str = "close",
    fill_to_zero: bool = True,
    color: str = GREEN,
) -> go.Figure:
    """A styled black/green line chart for one contract's time series.

    ``series`` is a dataframe already filtered to a single contract and sorted
    by ``x_col``. Set ``fill_to_zero=False`` for spread/fly series that cross
    zero (a baseline fill only makes sense for strictly positive outrights).
    ``color`` overrides the line/accent colour (defaults to the neon ``GREEN``).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series[x_col],
        y=series[y_col],
        mode="lines",
        name=y_title,
        line=dict(color=color, width=2),
        fill="tozeroy" if fill_to_zero else None,
        fillcolor=_rgba(color, 0.08) if fill_to_zero else None,
        hovertemplate="%{x|%b %d, %Y}<br>" + y_title + ": %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout(title, y_title, accent=color))
    return fig


def price_volume_figure(
    series,
    title: str,
    y_title: str,
    x_col: str = "date",
    y_col: str = "close",
    volume_col: str = "volume",
    fill_to_zero: bool = True,
    color: str = GREEN,
) -> go.Figure:
    """Two stacked panels sharing an x-axis: the price line on top and a volume
    bar chart below.

    Falls back to :func:`line_figure` when ``series`` has no usable ``volume_col``
    (e.g. the synthetic quarterly/semestral/yearly spreads and flies, which are
    derived from leg prices and therefore carry no traded volume).
    """
    has_volume = volume_col in getattr(series, "columns", []) and series[volume_col].notna().any()
    if not has_volume:
        return line_figure(series, title, y_title, x_col, y_col, fill_to_zero, color)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Scatter(
            x=series[x_col],
            y=series[y_col],
            mode="lines",
            name=y_title,
            line=dict(color=color, width=2),
            fill="tozeroy" if fill_to_zero else None,
            fillcolor=_rgba(color, 0.08) if fill_to_zero else None,
            hovertemplate="%{x|%b %d, %Y}<br>" + y_title + ": %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=series[x_col],
            y=series[volume_col],
            name="Volume",
            marker_color=_rgba(color, 0.45),
            marker_line_width=0,
            hovertemplate="%{x|%b %d, %Y}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(color=color, size=20)),
        template="plotly_dark",
        paper_bgcolor=BLACK,
        plot_bgcolor=BLACK,
        font=dict(color=TEXT, family="Arial"),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=70, b=50),
        showlegend=False,
        bargap=0.05,
    )
    fig.update_xaxes(gridcolor=GRID, showline=True, linecolor=color, linewidth=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(
        title_text=y_title, gridcolor=GRID, range=[series[y_col].min()-1, series[y_col].max()+1],
        showline=True, linecolor=color, linewidth=1, zeroline=False, row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Volume", gridcolor=GRID,
        showline=True, linecolor=color, linewidth=1, zeroline=False, row=2, col=1,
    )
    return fig
