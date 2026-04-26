"""
IMC Prosperity 4 – Round 3 Analysis Dashboard
==============================================
Auto-loads round 3 data on startup.
Optionally pass a backtester .log file to see your results overlaid.

Usage
-----
    python app.py
    python app.py path/to/run.log
    python app.py path/to/run.log --debug
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import Input, Output, State, dcc, html

from data_loader import (
    VEV_STRIKES,
    UNDERLYING,
    ALL_PRODUCTS,
    build_options_df,
    load_prices,
    load_trades,
    parse_log,
)
from options_math import fit_smile_parabola, smile_iv_from_coeffs

# ---------------------------------------------------------------------------
# Load all data once at startup
# ---------------------------------------------------------------------------

print("Loading prices and trades...", flush=True)
PRICES_DF = load_prices()
TRADES_DF = load_trades()
print(f"  {len(PRICES_DF):,} price rows, {len(TRADES_DF):,} trade rows", flush=True)

print("Computing options analytics (IV, smile, greeks) — ~10s...", flush=True)
OPTIONS_DF = build_options_df(PRICES_DF)
print(f"  {len(OPTIONS_DF):,} option rows across {OPTIONS_DF['product'].nunique()} strikes", flush=True)

# ---------------------------------------------------------------------------
# Optional: pre-load a log file passed as CLI argument
# ---------------------------------------------------------------------------

_LOG_ARG = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
INITIAL_LOG: dict | None = None

if _LOG_ARG:
    log_path = Path(_LOG_ARG)
    if log_path.exists():
        parsed = parse_log(log_path.read_bytes())
        if parsed:
            INITIAL_LOG = {}
            if "logs_df" in parsed:
                INITIAL_LOG["logs"] = parsed["logs_df"].to_dict("records")
            if "own_trades_df" in parsed:
                INITIAL_LOG["own_trades"] = parsed["own_trades_df"].to_dict("records")
            if "activities_df" in parsed:
                INITIAL_LOG["activities"] = parsed["activities_df"].to_dict("records")
            print(f"  Pre-loaded log: {log_path.name}  "
                  f"({len(INITIAL_LOG.get('own_trades', []))} own trades)", flush=True)
        else:
            print(f"  Warning: could not parse {log_path}", flush=True)
    else:
        print(f"  Warning: log file not found: {log_path}", flush=True)

print("Ready.\n", flush=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

STRIKE_COLORS = {
    "VEV_4000": "#2979ff",  # blue
    "VEV_4500": "#00bcd4",  # cyan
    "VEV_5000": "#00e676",  # green
    "VEV_5100": "#ffee58",  # yellow
    "VEV_5200": "#ffa726",  # orange
    "VEV_5300": "#ef5350",  # red
    "VEV_5400": "#ab47bc",  # purple
    "VEV_5500": "#ec407a",  # pink
    "VEV_6000": "#26c6da",  # teal
    "VEV_6500": "#9ccc65",  # lime
}

# TradingView / GitHub dark palette
BG       = "#131722"   # main background
CARD_BG  = "#1e222d"   # panel background
BORDER   = "#2a2e39"   # borders & dividers
GRID     = "#2a2e39"   # chart gridlines
TEXT     = "#d1d4dc"   # primary text
TEXT_DIM = "#787b86"   # secondary / label text
ACCENT   = "#2962ff"   # blue accent

BID_COLOR        = "#2196f3"   # bid (blue)
ASK_COLOR        = "#ef5350"   # ask (red)
OWN_TRADE_COLOR  = "#ff9800"   # own trades (amber)
ANON_TRADE_COLOR = "#90a4ae"   # anonymous trades

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Prosperity 4 · R3", suppress_callback_exceptions=True)

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _label(txt):
    return html.Span(txt, style={
        "fontSize": "10px", "fontWeight": "600", "letterSpacing": "0.5px",
        "textTransform": "uppercase", "color": TEXT_DIM, "marginBottom": "5px",
        "display": "block",
    })

def _card(*children, style=None):
    s = {
        "background": CARD_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "6px",
        "padding": "10px 12px",
    }
    if style:
        s.update(style)
    return html.Div(list(children), style=s)

def _section_title(txt, subtitle=""):
    return html.Div([
        html.Span(txt, style={"color": TEXT, "fontWeight": "600", "fontSize": "12px"}),
        html.Span(f"  {subtitle}", style={"color": TEXT_DIM, "fontSize": "11px"}) if subtitle else None,
    ], style={"marginBottom": "8px"})

DAY_OPTIONS = [
    {"label": "All days", "value": "all"},
    {"label": "Day 0",    "value": "0"},
    {"label": "Day 1",    "value": "1"},
    {"label": "Day 2",    "value": "2"},
]

NORM_OPTIONS = [
    {"label": "None",            "value": "none"},
    {"label": "Normalise by mid","value": "mid"},
    {"label": "Normalise by theoretical (BS)", "value": "theoretical"},
]

PRODUCT_OPTIONS = [{"label": p, "value": p} for p in ALL_PRODUCTS]
STRIKE_OPTIONS  = [{"label": f"{s}  K={k}", "value": s} for s, k in VEV_STRIKES.items()]

_cfg = {"displayModeBar": True, "scrollZoom": True, "displaylogo": False}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_ROW = {"display": "flex", "gap": "8px", "padding": "8px 12px", "background": BG,
        "alignItems": "stretch"}

app.layout = html.Div([
    # ── Header ────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("◈ ", style={"color": ACCENT, "fontSize": "16px"}),
            html.Span("IMC Prosperity 4", style={
                "color": "#ffffff", "fontWeight": "700", "fontSize": "14px",
                "letterSpacing": "0.3px",
            }),
            html.Span(" · Round 3  /  Order Book & Options",
                      style={"color": TEXT_DIM, "fontSize": "12px", "marginLeft": "6px"}),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div([
            dcc.Upload(
                id="log-upload",
                children=html.Div([
                    html.Span("⬆ ", style={"fontSize": "11px"}),
                    html.Span("Load .log file", style={"fontSize": "12px"}),
                ], style={"color": TEXT_DIM}),
                style={"padding": "5px 12px", "cursor": "pointer",
                       "borderRadius": "5px", "border": f"1px dashed {BORDER}"},
            ),
            html.Span(id="log-status", children="no log loaded",
                      style={"fontSize": "11px", "color": TEXT_DIM, "fontStyle": "italic"}),
        ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "10px 18px",
        "background": CARD_BG,
        "borderBottom": f"1px solid {BORDER}",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    dcc.Store(id="log-store", data=INITIAL_LOG),

    # ── Tabs ───────────────────────────────────────────────────────────────
    dcc.Tabs(value="tab-ob", children=[

        # ── Tab 1: Order Book ──────────────────────────────────────────────
        dcc.Tab(label="📈  Order Book", value="tab-ob",
                style={"color": TEXT_DIM, "background": BG, "border": "none",
                       "padding": "8px 18px", "fontSize": "12px", "fontWeight": "500"},
                selected_style={"color": "#ffffff", "background": CARD_BG,
                                "borderTop": f"2px solid {ACCENT}", "border": "none",
                                "padding": "8px 18px", "fontSize": "12px", "fontWeight": "600"},
                children=[

            # Controls bar
            html.Div([
                _card(
                    _label("Day"),
                    dcc.RadioItems(id="ob-day", options=DAY_OPTIONS, value="all",
                                  inline=True,
                                  inputStyle={"marginRight": "4px"},
                                  labelStyle={"marginRight": "12px", "color": TEXT, "fontSize": "12px"}),
                    style={"flex": "0 0 auto"},
                ),
                _card(
                    _label("Product"),
                    dcc.Dropdown(id="ob-product", options=PRODUCT_OPTIONS,
                                 value=UNDERLYING, clearable=False,
                                 style={"minWidth": "220px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _label("Normalise"),
                    dcc.Dropdown(id="ob-norm", options=NORM_OPTIONS, value="none",
                                 clearable=False,
                                 style={"minWidth": "220px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _label("Downsample  (1 = full resolution)"),
                    dcc.Slider(id="ob-ds", min=1, max=50, step=1, value=5,
                               marks={1: {"label": "1", "style": {"color": TEXT_DIM}},
                                      10: {"label": "10", "style": {"color": TEXT_DIM}},
                                      50: {"label": "50", "style": {"color": TEXT_DIM}}},
                               tooltip={"placement": "bottom", "always_visible": False}),
                    style={"flex": "1"},
                ),
                _card(
                    _label("Layers"),
                    dcc.Checklist(id="ob-show",
                                  options=[{"label": " Book", "value": "book"},
                                           {"label": " Trades", "value": "trades"},
                                           {"label": " Own", "value": "own"}],
                                  value=["book", "trades"],
                                  inline=True,
                                  inputStyle={"marginRight": "4px"},
                                  labelStyle={"marginRight": "10px", "color": TEXT, "fontSize": "12px"}),
                    style={"flex": "0 0 auto"},
                ),
            ], style={**_ROW, "padding": "10px 12px"}),

            # Main order book chart
            html.Div(
                _card(dcc.Graph(id="ob-chart", config=_cfg, style={"height": "490px"})),
                style={"padding": "0 12px 8px", "background": BG},
            ),

            # PnL + Position row
            html.Div([
                _card(
                    _section_title("PnL"),
                    dcc.Graph(id="pnl-chart", config=_cfg, style={"height": "190px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _section_title("Position"),
                    dcc.Graph(id="pos-chart", config=_cfg, style={"height": "190px"}),
                    style={"flex": "1"},
                ),
            ], style={**_ROW, "padding": "0 12px 8px"}),

            # Log viewer
            html.Div(
                _card(
                    _section_title("Log Viewer", "hover chart → sync timestamp"),
                    html.Pre(id="log-viewer",
                             children="Upload a .log file above to see algorithm output here.",
                             style={"maxHeight": "160px", "overflow": "auto",
                                    "margin": "0", "color": "#8b949e",
                                    "fontSize": "11px", "lineHeight": "1.6",
                                    "whiteSpace": "pre-wrap"}),
                ),
                style={"padding": "0 12px 14px", "background": BG},
            ),
        ]),

        # ── Tab 2: Options ─────────────────────────────────────────────────
        dcc.Tab(label="⚡  Options Analytics", value="tab-opts",
                style={"color": TEXT_DIM, "background": BG, "border": "none",
                       "padding": "8px 18px", "fontSize": "12px", "fontWeight": "500"},
                selected_style={"color": "#ffffff", "background": CARD_BG,
                                "borderTop": "2px solid #7c3aed", "border": "none",
                                "padding": "8px 18px", "fontSize": "12px", "fontWeight": "600"},
                children=[

            # Controls bar
            html.Div([
                _card(
                    _label("Day"),
                    dcc.RadioItems(id="opt-day", options=DAY_OPTIONS, value="all",
                                  inline=True,
                                  inputStyle={"marginRight": "4px"},
                                  labelStyle={"marginRight": "12px", "color": TEXT, "fontSize": "12px"}),
                    style={"flex": "0 0 auto"},
                ),
                _card(
                    _label("Strikes"),
                    dcc.Checklist(id="opt-strikes", options=STRIKE_OPTIONS,
                                  value=list(VEV_STRIKES.keys()),
                                  inline=True,
                                  inputStyle={"marginRight": "4px"},
                                  labelStyle={"marginRight": "12px", "color": TEXT, "fontSize": "12px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _label("Smile snapshot  (blank = all time)"),
                    dcc.Input(id="smile-ts", type="number", debounce=True,
                              placeholder="global_ts e.g. 500000",
                              style={"width": "190px"}),
                    style={"flex": "0 0 auto"},
                ),
            ], style={**_ROW, "padding": "10px 12px"}),

            # Row 1: IV Smile + IV Deviations
            html.Div([
                _card(
                    _section_title("IV Smile", "moneyness vs IV · fitted parabola"),
                    dcc.Graph(id="iv-smile", config=_cfg, style={"height": "350px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _section_title("IV Deviations over Time", "v_t − v̂_t per strike"),
                    dcc.Graph(id="iv-dev", config=_cfg, style={"height": "370px"}),
                    style={"flex": "1"},
                ),
            ], style={**_ROW, "padding": "0 12px 8px"}),

            # Row 2: Market vs Theoretical + Price Deviations
            html.Div([
                _card(
                    _section_title("Market vs BS Theoretical", "solid = market · dashed = theoretical"),
                    dcc.Graph(id="price-vs-theo", config=_cfg, style={"height": "350px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _section_title("Price Deviations", "market − theoretical  (scalping signal)"),
                    dcc.Graph(id="price-dev", config=_cfg, style={"height": "350px"}),
                    style={"flex": "1"},
                ),
            ], style={**_ROW, "padding": "0 12px 8px"}),

            # Row 3: Greeks
            html.Div([
                _card(_section_title("Δ  Delta"),
                      dcc.Graph(id="delta-chart", config=_cfg, style={"height": "270px"}),
                      style={"flex": "1"}),
                _card(_section_title("Γ  Gamma"),
                      dcc.Graph(id="gamma-chart", config=_cfg, style={"height": "270px"}),
                      style={"flex": "1"}),
                _card(_section_title("ν  Vega"),
                      dcc.Graph(id="vega-chart",  config=_cfg, style={"height": "270px"}),
                      style={"flex": "1"}),
            ], style={**_ROW, "padding": "0 12px 14px"}),
        ]),

        # ── Tab 3: Results ─────────────────────────────────────────────────
        dcc.Tab(label="🏆  Results", value="tab-results",
                style={"color": TEXT_DIM, "background": BG, "border": "none",
                       "padding": "8px 18px", "fontSize": "12px", "fontWeight": "500"},
                selected_style={"color": "#ffffff", "background": CARD_BG,
                                "borderTop": "2px solid #00e676", "border": "none",
                                "padding": "8px 18px", "fontSize": "12px", "fontWeight": "600"},
                children=[

            # Controls
            html.Div([
                _card(
                    _label("Day"),
                    dcc.RadioItems(id="res-day", options=DAY_OPTIONS, value="all",
                                  inline=True,
                                  inputStyle={"marginRight": "4px"},
                                  labelStyle={"marginRight": "12px", "color": TEXT, "fontSize": "12px"}),
                    style={"flex": "0 0 auto"},
                ),
                _card(
                    _label("Product  (for trade overlay)"),
                    dcc.Dropdown(id="res-product", options=PRODUCT_OPTIONS,
                                 value=UNDERLYING, clearable=False,
                                 style={"minWidth": "220px"}),
                    style={"flex": "1"},
                ),
                html.Div(
                    id="res-log-hint",
                    children=html.Span(
                        f"{'✓  Log pre-loaded: ' + _LOG_ARG if _LOG_ARG else '← Upload a .log file in the header, or pass it as: python app.py run.log'}",
                        style={"fontSize": "11px",
                               "color": "#00e676" if _LOG_ARG else TEXT_DIM,
                               "fontStyle": "italic"},
                    ),
                    style={"display": "flex", "alignItems": "center", "padding": "0 4px"},
                ),
            ], style={**_ROW, "padding": "10px 12px"}),

            # Cumulative PnL across all products
            html.Div(
                _card(
                    _section_title("Cumulative PnL — All Products", "from activitiesLog"),
                    dcc.Graph(id="res-total-pnl", config=_cfg, style={"height": "300px"}),
                ),
                style={"padding": "0 12px 8px", "background": BG},
            ),

            # Per-product PnL grid + trade overlay
            html.Div([
                _card(
                    _section_title("Per-Product PnL", "each line = one symbol"),
                    dcc.Graph(id="res-perprod-pnl", config=_cfg, style={"height": "320px"}),
                    style={"flex": "1"},
                ),
                _card(
                    _section_title("Price + Trade Overlay",
                                   "▲ buy  ▼ sell  · green = profitable  red = loss"),
                    dcc.Graph(id="res-trade-overlay", config=_cfg, style={"height": "340px"}),
                    style={"flex": "1"},
                ),
            ], style={**_ROW, "padding": "0 12px 8px"}),

            # Trade log table
            html.Div(
                _card(
                    _section_title("Own Trade Log"),
                    html.Div(id="res-trade-table",
                             style={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"}),
                ),
                style={"padding": "0 12px 14px", "background": BG},
            ),
        ]),
    ]),
], style={"background": BG, "minHeight": "100vh", "color": TEXT,
          "fontFamily": "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days(day_str):
    return [0, 1, 2] if day_str == "all" else [int(day_str)]


def _empty(msg="No data"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=40, r=20, t=30, b=30),
        annotations=[dict(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(color=TEXT_DIM, size=13))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _base_layout(title="", xlabel="Timestamp", ylabel=""):
    return dict(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="'Inter', 'Segoe UI', sans-serif", size=11),
        margin=dict(l=56, r=16, t=10, b=36),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(size=10, color=TEXT_DIM)),
            gridcolor=BORDER, gridwidth=1,
            tickfont=dict(color=TEXT_DIM, size=10),
            linecolor=BORDER, zerolinecolor=BORDER,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(size=10, color=TEXT_DIM)),
            gridcolor=BORDER, gridwidth=1,
            tickfont=dict(color=TEXT_DIM, size=10),
            linecolor=BORDER, zerolinecolor=BORDER,
        ),
        legend=dict(
            font=dict(size=10, color=TEXT),
            bgcolor="rgba(30,34,45,0.85)",
            bordercolor=BORDER, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER,
                        font=dict(color=TEXT, size=11)),
    )


# ---------------------------------------------------------------------------
# Callback: log upload
# ---------------------------------------------------------------------------

@app.callback(
    Output("log-store", "data"),
    Output("log-status", "children"),
    Input("log-upload", "contents"),
    State("log-upload", "filename"),
    prevent_initial_call=True,
)
def upload_log(contents, filename):
    if contents is None:
        return None, "No log loaded"
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    parsed = parse_log(decoded)
    if parsed is None:
        return None, f"Failed to parse {filename}"
    result = {}
    if "logs_df" in parsed:
        result["logs"] = parsed["logs_df"].to_dict("records")
    if "own_trades_df" in parsed:
        result["own_trades"] = parsed["own_trades_df"].to_dict("records")
    n_own = len(result.get("own_trades", []))
    return result, f"{filename}  ({n_own} own trades)"


# ---------------------------------------------------------------------------
# Callback: Order Book chart + PnL + Position
# ---------------------------------------------------------------------------

@app.callback(
    Output("ob-chart",  "figure"),
    Output("pnl-chart", "figure"),
    Output("pos-chart", "figure"),
    Input("ob-product", "value"),
    Input("ob-day",     "value"),
    Input("ob-norm",    "value"),
    Input("ob-ds",      "value"),
    Input("ob-show",    "value"),
    Input("log-store",  "data"),
)
def update_ob(product, day_str, norm_mode, ds, show_flags, log_data):
    days = _days(day_str)
    pdata = PRICES_DF[
        (PRICES_DF["product"] == product) & PRICES_DF["day"].isin(days)
    ].copy()

    if pdata.empty:
        e = _empty(f"No data for {product}")
        return e, e, e

    step = max(1, int(ds or 5))
    pdata = pdata.iloc[::step].reset_index(drop=True)
    xts = pdata["global_ts"]

    # Normalisation reference
    ref = np.zeros(len(pdata))
    if norm_mode == "mid":
        ref = pdata["mid_price"].fillna(0).values
    elif norm_mode == "theoretical" and product in VEV_STRIKES:
        opt_sub = OPTIONS_DF[
            (OPTIONS_DF["product"] == product) & OPTIONS_DF["day"].isin(days)
        ].iloc[::step]
        if not opt_sub.empty:
            merged = pdata.merge(
                opt_sub[["global_ts", "theoretical_price"]], on="global_ts", how="left"
            )
            ref = merged["theoretical_price"].fillna(pdata["mid_price"]).values

    # ── Order book figure ─────────────────────────────────────────────────
    fig = go.Figure()

    if "book" in show_flags:
        for lvl in range(1, 4):
            bp = pdata[f"bid_price_{lvl}"] - ref
            bv = pdata[f"bid_volume_{lvl}"]
            ap = pdata[f"ask_price_{lvl}"] - ref
            av = pdata[f"ask_volume_{lvl}"]
            opacity = max(0.35, 1.0 - 0.22 * (lvl - 1))

            valid_b = bv.notna() & bp.notna()
            if valid_b.any():
                sz = (bv[valid_b].clip(1, 80) * 0.35 + 3).clip(3, 12)
                fig.add_trace(go.Scatter(
                    x=xts[valid_b], y=bp[valid_b], mode="markers",
                    name="Bid" if lvl == 1 else f"Bid L{lvl}",
                    legendgroup="bid", showlegend=(lvl == 1),
                    marker=dict(color=BID_COLOR, size=sz, opacity=opacity, symbol="circle"),
                    hovertemplate=f"<b>Bid L{lvl}</b> ts=%{{x}}<br>price=%{{y:.1f}}  vol=%{{text}}<extra></extra>",
                    text=bv[valid_b].astype(int).astype(str),
                ))

            valid_a = av.notna() & ap.notna()
            if valid_a.any():
                sz = (av[valid_a].clip(1, 80) * 0.35 + 3).clip(3, 12)
                fig.add_trace(go.Scatter(
                    x=xts[valid_a], y=ap[valid_a], mode="markers",
                    name="Ask" if lvl == 1 else f"Ask L{lvl}",
                    legendgroup="ask", showlegend=(lvl == 1),
                    marker=dict(color=ASK_COLOR, size=sz, opacity=opacity, symbol="circle"),
                    hovertemplate=f"<b>Ask L{lvl}</b> ts=%{{x}}<br>price=%{{y:.1f}}  vol=%{{text}}<extra></extra>",
                    text=av[valid_a].astype(int).astype(str),
                ))

    # Mid line
    fig.add_trace(go.Scatter(
        x=xts, y=pdata["mid_price"] - ref, mode="lines", name="Mid",
        line=dict(color=TEXT_DIM, width=1, dash="dot"), opacity=0.5,
    ))

    # Historical trades
    if "trades" in show_flags and not TRADES_DF.empty:
        tdata = TRADES_DF[
            (TRADES_DF["symbol"] == product) & TRADES_DF["day"].isin(days)
        ]
        if not tdata.empty:
            # Interpolate ref to trade timestamps
            ref_interp = np.interp(tdata["global_ts"].values, xts.values,
                                   ref if norm_mode != "none" else np.zeros(len(pdata)))
            fig.add_trace(go.Scatter(
                x=tdata["global_ts"], y=tdata["price"] - ref_interp,
                mode="markers", name="Trades",
                marker=dict(color=ANON_TRADE_COLOR, size=7, symbol="diamond",
                            line=dict(width=0.5, color="#fff")),
                hovertemplate="<b>Trade</b> ts=%{x}<br>price=%{y:.1f}  qty=%{text}<extra></extra>",
                text=tdata["quantity"].astype(str),
            ))

    # Own trades from log
    if "own" in show_flags and log_data and "own_trades" in log_data:
        own_df = pd.DataFrame(log_data["own_trades"])
        if not own_df.empty and "symbol" in own_df.columns:
            own = own_df[own_df["symbol"] == product]
            if not own.empty:
                side = own.get("side", pd.Series("UNKNOWN", index=own.index))
                is_buy = side == "BUY"
                fig.add_trace(go.Scatter(
                    x=own["timestamp"], y=own["price"],
                    mode="markers", name="Own",
                    marker=dict(color=OWN_TRADE_COLOR, size=10,
                                symbol=["cross" if b else "x" for b in is_buy],
                                line=dict(width=2, color=OWN_TRADE_COLOR)),
                    hovertemplate="<b>Own trade</b> ts=%{x}<br>price=%{y:.1f}<extra></extra>",
                ))

    ylabel = f"Price − {norm_mode}" if norm_mode != "none" else "Price"
    fig.update_layout(**_base_layout(f"{product}  ·  Order Book", ylabel=ylabel))

    # ── PnL ─────────────────────────────────────────────────────────────
    pnl_fig = go.Figure(go.Scatter(
        x=xts, y=pdata["profit_and_loss"], mode="lines", name="PnL",
        line=dict(color="#22d3ee", width=1.5),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.07)",
    ))
    pnl_fig.update_layout(**_base_layout("PnL", ylabel="SeaShells"))

    # ── Position ────────────────────────────────────────────────────────
    pos_fig = go.Figure()
    if log_data and "own_trades" in log_data:
        own_df = pd.DataFrame(log_data["own_trades"])
        if not own_df.empty and "symbol" in own_df.columns:
            own = own_df[own_df["symbol"] == product].sort_values("timestamp")
            if not own.empty:
                side = own.get("side", pd.Series("UNKNOWN", index=own.index))
                is_buy = (side == "BUY").astype(int)
                position = (own["quantity"] * (is_buy * 2 - 1)).cumsum()
                pos_fig.add_trace(go.Scatter(
                    x=own["timestamp"], y=position, mode="lines+markers", name="Position",
                    line=dict(color="#f59e0b", width=1.5), marker=dict(size=4),
                ))
    else:
        pos_fig.add_annotation(text="Upload .log for position", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(color=TEXT_DIM, size=11))
    pos_fig.update_layout(**_base_layout("Position", ylabel="Units"))

    return fig, pnl_fig, pos_fig


# ---------------------------------------------------------------------------
# Callback: log viewer (synced to hover)
# ---------------------------------------------------------------------------

@app.callback(
    Output("log-viewer", "children"),
    Input("ob-chart", "hoverData"),
    State("log-store", "data"),
    prevent_initial_call=True,
)
def update_log_viewer(hover_data, log_data):
    if log_data is None or "logs" not in log_data:
        return "Upload a .log file above."
    if hover_data is None:
        return "Hover over chart to sync."

    hover_ts = hover_data["points"][0].get("x", None)
    if hover_ts is None:
        return ""
    real_ts = int(hover_ts) % 1_000_000

    matching = [e for e in log_data["logs"] if abs(e.get("timestamp", -1) - real_ts) <= 100]
    if not matching:
        return f"No log at ts={real_ts}"

    parts = []
    for e in matching[:5]:
        if e.get("lambdaLog"):
            parts.append(f"[ts={e['timestamp']}]\n{e['lambdaLog']}")
        if e.get("sandboxLog"):
            parts.append(f"[sandbox] {e['sandboxLog']}")
    return "\n\n".join(parts) if parts else f"Empty log at ts={real_ts}"


# ---------------------------------------------------------------------------
# Callbacks: Options charts
# ---------------------------------------------------------------------------

def _opt_filter(day_str, strikes):
    days = _days(day_str)
    return OPTIONS_DF[OPTIONS_DF["day"].isin(days) & OPTIONS_DF["product"].isin(strikes)]


@app.callback(
    Output("iv-smile", "figure"),
    Input("opt-day",    "value"),
    Input("opt-strikes","value"),
    Input("smile-ts",   "value"),
)
def update_iv_smile(day_str, strikes, smile_ts):
    if not strikes:
        return _empty("Select strikes")
    sub = _opt_filter(day_str, strikes)

    # If a specific timestamp is given, show the smile at that snapshot
    if smile_ts is not None:
        sub = sub[np.abs(sub["global_ts"] - int(smile_ts)) <= 100_000]

    fig = go.Figure()

    # Plot ask IV (red) and bid IV (blue) per strike
    for prod in strikes:
        pdata = sub[sub["product"] == prod]
        c = STRIKE_COLORS.get(prod, "#fff")

        ask_pts = pdata.dropna(subset=["moneyness", "ask_iv"])
        if not ask_pts.empty:
            fig.add_trace(go.Scatter(
                x=ask_pts["moneyness"], y=ask_pts["ask_iv"], mode="markers",
                name=f"{prod} ask",
                legendgroup=prod, legendgrouptitle_text=prod if prod == strikes[0] else None,
                marker=dict(color=ASK_COLOR, size=4, opacity=0.45, symbol="circle"),
                hovertemplate=f"<b>{prod} ask</b>  m=%{{x:.4f}}  IV=%{{y:.5f}}<extra></extra>",
            ))

        bid_pts = pdata.dropna(subset=["moneyness", "bid_iv"])
        if not bid_pts.empty:
            fig.add_trace(go.Scatter(
                x=bid_pts["moneyness"], y=bid_pts["bid_iv"], mode="markers",
                name=f"{prod} bid",
                legendgroup=prod,
                marker=dict(color=BID_COLOR, size=4, opacity=0.45, symbol="circle"),
                hovertemplate=f"<b>{prod} bid</b>  m=%{{x:.4f}}  IV=%{{y:.5f}}<extra></extra>",
            ))

    # Fit and draw ask smile + bid smile
    all_m_min, all_m_max = [], []
    for iv_col, color, label in [
        ("ask_iv", ASK_COLOR, "Ask smile"),
        ("bid_iv", BID_COLOR, "Bid smile"),
    ]:
        valid = sub.dropna(subset=["moneyness", iv_col])
        if len(valid) < 3:
            continue
        coeffs = fit_smile_parabola(valid["moneyness"].values, valid[iv_col].values)
        if coeffs is None:
            continue
        m_lo, m_hi = valid["moneyness"].min(), valid["moneyness"].max()
        all_m_min.append(m_lo)
        all_m_max.append(m_hi)
        m_rng = np.linspace(m_lo, m_hi, 400)
        iv_fit = smile_iv_from_coeffs(m_rng, coeffs)
        fig.add_trace(go.Scatter(
            x=m_rng, y=iv_fit, mode="lines", name=label,
            line=dict(color=color, width=2.5, dash="dash"),
            hovertemplate=f"<b>{label}</b>  m=%{{x:.4f}}  IV=%{{y:.5f}}<extra></extra>",
        ))
        a, b, c = coeffs
        ypos = 0.97 if iv_col == "ask_iv" else 0.90
        fig.add_annotation(
            text=f"{label}:  a={a:.5f}  b={b:.5f}  c={c:.5f}",
            xref="paper", yref="paper", x=0.02, y=ypos,
            showarrow=False, font=dict(size=10, color=color),
            bgcolor="rgba(30,34,45,0.75)", bordercolor=BORDER,
        )

    # ATM vertical reference line at moneyness = 0
    fig.add_vline(
        x=0,
        line=dict(color=TEXT_DIM, width=1.5, dash="dot"),
        annotation_text="ATM",
        annotation_position="top",
        annotation_font=dict(color=TEXT_DIM, size=10),
    )

    fig.update_layout(**_base_layout("IV Smile", xlabel="Log-moneyness  log(S/K)/√T  (positive = ITM)",
                                     ylabel="IV (per √Solvenarian day)"))
    return fig


@app.callback(
    Output("iv-dev", "figure"),
    Input("opt-day",    "value"),
    Input("opt-strikes","value"),
)
def update_iv_dev(day_str, strikes):
    if not strikes:
        return _empty("Select strikes")
    sub = _opt_filter(day_str, strikes)
    fig = go.Figure()
    for prod in strikes:
        pdata = sub[sub["product"] == prod].dropna(subset=["iv_dev"])
        if pdata.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pdata["global_ts"], y=pdata["iv_dev"], mode="lines", name=prod,
            line=dict(color=STRIKE_COLORS.get(prod, "#fff"), width=1), opacity=0.85,
            hovertemplate=f"<b>{prod}</b>  ts=%{{x}}<br>IV_dev=%{{y:.5f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))
    fig.update_layout(**_base_layout("IV Deviations over Time",
                                     ylabel="v_t − v̂_t"))
    return fig


@app.callback(
    Output("price-vs-theo", "figure"),
    Input("opt-day",    "value"),
    Input("opt-strikes","value"),
)
def update_price_theo(day_str, strikes):
    if not strikes:
        return _empty("Select strikes")
    sub = _opt_filter(day_str, strikes)
    fig = go.Figure()
    for prod in strikes:
        pdata = sub[sub["product"] == prod]
        if pdata.empty:
            continue
        c = STRIKE_COLORS.get(prod, "#fff")
        fig.add_trace(go.Scatter(
            x=pdata["global_ts"], y=pdata["mid_price"], mode="lines", name=f"{prod}",
            line=dict(color=c, width=1), legendgroup=prod,
            hovertemplate=f"<b>{prod} mkt</b>  ts=%{{x}}<br>%{{y:.2f}}<extra></extra>",
        ))
        theo = pdata["theoretical_price"].dropna()
        if not theo.empty:
            fig.add_trace(go.Scatter(
                x=pdata.loc[theo.index, "global_ts"], y=theo, mode="lines",
                name=f"{prod} theo", line=dict(color=c, width=1.5, dash="dash"),
                legendgroup=prod, showlegend=False,
                hovertemplate=f"<b>{prod} theo</b>  ts=%{{x}}<br>%{{y:.2f}}<extra></extra>",
            ))
    fig.update_layout(**_base_layout("Market vs Theoretical  (dashed=BS)",
                                     ylabel="Price"))
    return fig


@app.callback(
    Output("price-dev",  "figure"),
    Input("opt-day",    "value"),
    Input("opt-strikes","value"),
)
def update_price_dev(day_str, strikes):
    if not strikes:
        return _empty("Select strikes")
    sub = _opt_filter(day_str, strikes)
    fig = go.Figure()
    for prod in strikes:
        pdata = sub[sub["product"] == prod].dropna(subset=["price_dev"])
        if pdata.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pdata["global_ts"], y=pdata["price_dev"], mode="lines", name=prod,
            line=dict(color=STRIKE_COLORS.get(prod, "#fff"), width=1), opacity=0.85,
            hovertemplate=f"<b>{prod}</b>  ts=%{{x}}<br>dev=%{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=BORDER, width=1, dash="dot"))
    fig.update_layout(**_base_layout("Price Deviations  (market − theoretical)",
                                     ylabel="Price dev"))
    return fig


@app.callback(
    Output("delta-chart", "figure"),
    Output("gamma-chart", "figure"),
    Output("vega-chart",  "figure"),
    Input("opt-day",    "value"),
    Input("opt-strikes","value"),
)
def update_greeks(day_str, strikes):
    if not strikes:
        e = _empty("Select strikes")
        return e, e, e
    sub = _opt_filter(day_str, strikes)

    def _greek_fig(col, label):
        fig = go.Figure()
        for prod in strikes:
            pdata = sub[sub["product"] == prod].dropna(subset=[col])
            if pdata.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pdata["global_ts"], y=pdata[col], mode="lines", name=prod,
                line=dict(color=STRIKE_COLORS.get(prod, "#fff"), width=1), opacity=0.85,
                hovertemplate=f"<b>{prod}</b>  ts=%{{x}}<br>{col}=%{{y:.5f}}<extra></extra>",
            ))
        fig.update_layout(**_base_layout(label, ylabel=label))
        return fig

    return _greek_fig("delta", "Δ Delta"), _greek_fig("gamma", "Γ Gamma"), _greek_fig("vega", "ν Vega")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Callbacks: Tab 3 – Results
# ---------------------------------------------------------------------------

# Colour pool for per-product PnL lines
_PROD_COLORS = [
    "#2979ff", "#00e676", "#ffee58", "#ef5350", "#ab47bc",
    "#00bcd4", "#ffa726", "#ec407a", "#9ccc65", "#26c6da",
    "#80cbc4", "#ff7043",
]


def _build_activities(log_data) -> pd.DataFrame | None:
    """Extract activitiesLog DataFrame from log_store dict."""
    if not log_data or "activities" not in log_data:
        return None
    df = pd.DataFrame(log_data["activities"])
    if df.empty:
        return None
    df["profit_and_loss"] = pd.to_numeric(df.get("profit_and_loss", 0), errors="coerce").fillna(0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
    return df


def _build_own_trades(log_data) -> pd.DataFrame | None:
    if not log_data or "own_trades" not in log_data:
        return None
    df = pd.DataFrame(log_data["own_trades"])
    if df.empty:
        return None
    df["price"]    = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


@app.callback(
    Output("res-total-pnl",     "figure"),
    Output("res-perprod-pnl",   "figure"),
    Output("res-trade-overlay", "figure"),
    Output("res-trade-table",   "children"),
    Input("res-day",     "value"),
    Input("res-product", "value"),
    Input("log-store",   "data"),
)
def update_results(day_str, product, log_data):
    days = _days(day_str)
    acts = _build_activities(log_data)
    own  = _build_own_trades(log_data)

    no_log = _empty("No log loaded — upload a .log file or pass it via:  python app.py run.log")

    # ── 1. Cumulative total PnL ──────────────────────────────────────────
    if acts is None:
        total_fig = no_log
    else:
        acts_f = acts[acts["day"].isin(days)]
        # Sum PnL across products at each timestamp
        total_ts = (
            acts_f.groupby("global_ts")["profit_and_loss"]
            .sum()
            .reset_index()
            .sort_values("global_ts")
        )
        total_fig = go.Figure(go.Scatter(
            x=total_ts["global_ts"],
            y=total_ts["profit_and_loss"],
            mode="lines",
            name="Total PnL",
            line=dict(color="#00e676", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.07)",
            hovertemplate="ts=%{x}<br>PnL=%{y:,.0f}<extra></extra>",
        ))
        # Shade positive/negative regions differently
        layout = _base_layout(ylabel="Cumulative PnL (SeaShells)")
        layout["yaxis"].update(zeroline=True, zerolinecolor="#444c56", zerolinewidth=1)
        total_fig.update_layout(**layout)

    # ── 2. Per-product PnL ───────────────────────────────────────────────
    if acts is None:
        perprod_fig = no_log
    else:
        acts_f = acts[acts["day"].isin(days)]
        products_in_log = sorted(acts_f["product"].unique()) if "product" in acts_f.columns else []
        perprod_fig = go.Figure()
        for i, prod in enumerate(products_in_log):
            sub = acts_f[acts_f["product"] == prod].sort_values("global_ts")
            color = _PROD_COLORS[i % len(_PROD_COLORS)]
            perprod_fig.add_trace(go.Scatter(
                x=sub["global_ts"],
                y=sub["profit_and_loss"],
                mode="lines",
                name=prod,
                line=dict(color=color, width=1.2),
                opacity=0.9,
                hovertemplate=f"<b>{prod}</b>  ts=%{{x}}<br>PnL=%{{y:,.1f}}<extra></extra>",
            ))
        perprod_fig.update_layout(**_base_layout(ylabel="PnL"))

    # ── 3. Price + trade overlay for selected product ────────────────────
    price_sub = PRICES_DF[
        (PRICES_DF["product"] == product) & PRICES_DF["day"].isin(days)
    ].iloc[::5]  # downsample for speed

    trade_fig = go.Figure()

    # Mid price line
    trade_fig.add_trace(go.Scatter(
        x=price_sub["global_ts"],
        y=price_sub["mid_price"],
        mode="lines",
        name="Mid price",
        line=dict(color="#546e7a", width=1),
        hovertemplate="ts=%{x}<br>mid=%{y:.2f}<extra></extra>",
    ))

    # Bid/ask band (L1 only)
    valid_ba = price_sub["bid_price_1"].notna() & price_sub["ask_price_1"].notna()
    if valid_ba.any():
        trade_fig.add_trace(go.Scatter(
            x=pd.concat([price_sub.loc[valid_ba, "global_ts"],
                          price_sub.loc[valid_ba, "global_ts"].iloc[::-1]]),
            y=pd.concat([price_sub.loc[valid_ba, "ask_price_1"],
                          price_sub.loc[valid_ba, "bid_price_1"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(84,110,122,0.12)",
            line=dict(width=0),
            name="Bid-Ask band",
            hoverinfo="skip",
            showlegend=True,
        ))

    # Own trades with P&L-delta colour
    if own is not None:
        own_prod = own[own["symbol"] == product].copy().sort_values("timestamp").reset_index(drop=True)
        if not own_prod.empty:
            # Use pre-computed side column (set in parse_log)
            side_col = own_prod.get("side", pd.Series("UNKNOWN", index=own_prod.index))
            buys     = own_prod[side_col == "BUY"]
            sells    = own_prod[side_col == "SELL"]
            unknowns = own_prod[side_col == "UNKNOWN"]

            for df_side, color, sym, name in [
                (buys,     "#00e676", "triangle-up",   "Buy"),
                (sells,    "#ef5350", "triangle-down",  "Sell"),
                (unknowns, "#90a4ae", "circle",          "Trade"),
            ]:
                if df_side.empty:
                    continue
                trade_fig.add_trace(go.Scatter(
                    x=df_side["timestamp"],
                    y=df_side["price"],
                    mode="markers",
                    name=name,
                    marker=dict(
                        color=color,
                        size=(df_side["quantity"].clip(1, 30) * 0.6 + 8).clip(8, 18),
                        symbol=sym,
                        line=dict(width=1.5, color=color),
                        opacity=0.9,
                    ),
                    hovertemplate=(
                        f"<b>{name}</b>  ts=%{{x}}<br>"
                        "price=%{y:.2f}  qty=%{text}<extra></extra>"
                    ),
                    text=df_side["quantity"].astype(int).astype(str),
                ))

    trade_fig.update_layout(
        **_base_layout(f"{product}  ·  Price + Trade Overlay", ylabel="Price"),
    )

    # ── 4. Trade table ────────────────────────────────────────────────────
    if own is None:
        table = html.Span("No own trades in log.", style={"color": TEXT_DIM, "fontSize": "12px"})
    else:
        own_f = own[own["symbol"].isin(
            [p for p in ALL_PRODUCTS if p in own["symbol"].unique()]
        )].sort_values("timestamp")

        # Summary per product
        summary = []
        for prod, grp in own_f.groupby("symbol"):
            side = grp.get("side", pd.Series("UNKNOWN", index=grp.index))
            is_b   = side == "BUY"
            n_buy  = int(is_b.sum())
            n_sell = int((side == "SELL").sum())
            vol    = int(grp["quantity"].sum())
            avg_buy  = grp.loc[is_b,  "price"].mean() if n_buy  else float("nan")
            avg_sell = grp.loc[side == "SELL", "price"].mean() if n_sell else float("nan")
            summary.append({"Symbol": prod, "Buys": n_buy, "Sells": n_sell,
                             "Total Vol": vol,
                             "Avg Buy": f"{avg_buy:.2f}" if n_buy else "—",
                             "Avg Sell": f"{avg_sell:.2f}" if n_sell else "—"})

        if not summary:
            table = html.Span("No trades.", style={"color": TEXT_DIM})
        else:
            cols = list(summary[0].keys())
            header_cells = [
                html.Th(c, style={
                    "padding": "6px 12px", "textAlign": "left",
                    "color": TEXT_DIM, "fontSize": "11px",
                    "fontWeight": "600", "letterSpacing": "0.5px",
                    "textTransform": "uppercase",
                    "borderBottom": f"1px solid {BORDER}",
                })
                for c in cols
            ]
            rows = []
            for row in summary:
                cells = [
                    html.Td(str(row[c]), style={
                        "padding": "5px 12px", "color": TEXT,
                        "fontSize": "12px", "borderBottom": f"1px solid {BORDER}",
                        "fontFamily": "'JetBrains Mono', monospace",
                    })
                    for c in cols
                ]
                rows.append(html.Tr(cells))

            table = html.Table(
                [html.Thead(html.Tr(header_cells)), html.Tbody(rows)],
                style={"width": "100%", "borderCollapse": "collapse",
                       "background": CARD_BG},
            )

    return total_fig, perprod_fig, trade_fig, table


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    port = 8050
    print(f"Dashboard → http://localhost:{port}")
    app.run(debug=debug, port=port)
