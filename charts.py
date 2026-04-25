"""
charts.py
=========
Plotly Visualization Module — Silk Tensile Testing Simulator
-------------------------------------------------------------
Generates all interactive charts used in the Streamlit dashboard.

Each chart function follows a dark-themed professional aesthetic consistent
with a clinical / laboratory instrumentation interface.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared Design Tokens ────────────────────────────────────────────────────
DARK_BG           = "#0d1117"
PANEL_BG          = "#161b22"
GRID_COLOR        = "#21262d"
TEXT_COLOR        = "#e6edf3"
ACCENT_TEAL       = "#00d4aa"
ACCENT_GOLD       = "#f0b429"
ACCENT_RED        = "#f85149"
ACCENT_BLUE       = "#58a6ff"
ACCENT_PURPLE     = "#a371f7"
FONT_FAMILY       = "Inter, Helvetica Neue, Arial, sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor = DARK_BG,
    plot_bgcolor  = PANEL_BG,
    font          = dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
    margin        = dict(l=60, r=20, t=60, b=60),
    xaxis         = dict(
        gridcolor     = GRID_COLOR,
        linecolor     = GRID_COLOR,
        tickcolor     = GRID_COLOR,
        showline      = True,
        mirror        = True,
    ),
    yaxis         = dict(
        gridcolor     = GRID_COLOR,
        linecolor     = GRID_COLOR,
        tickcolor     = GRID_COLOR,
        showline      = True,
        mirror        = True,
    ),
    legend        = dict(
        bgcolor       = PANEL_BG,
        bordercolor   = GRID_COLOR,
        borderwidth   = 1,
    ),
)


# ── 1. Full Stress-Strain Curve ─────────────────────────────────────────────
def plot_stress_strain_curve(
    strains:        np.ndarray,
    stresses:       np.ndarray,
    applied_strain: float,
    applied_stress: float,
    processing_method: str,
    uts_mpa:        float,
    yield_strain:   float,
) -> go.Figure:
    """
    Renders the complete stress-strain curve with:
      • The theoretical material curve (solid gradient line)
      • A live operating point marker (user's applied load)
      • Vertical reference lines for yield point and UTS
      • Shaded elastic region for visual guidance
    """
    fig = go.Figure()

    yield_idx = int(yield_strain / strains[-1] * len(strains)) if strains[-1] > 0 else 0

    # ── Shaded linear-elastic region ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = np.concatenate([strains[:yield_idx], strains[:yield_idx][::-1]]),
        y    = np.concatenate([stresses[:yield_idx], np.zeros(yield_idx)]),
        fill = "toself",
        fillcolor = "rgba(0, 212, 170, 0.07)",
        line = dict(color="rgba(0,0,0,0)"),
        name = "Linear Elastic Region",
        hoverinfo = "skip",
    ))

    # ── Main stress-strain line ─────────────────────────────────────────────
    color = ACCENT_TEAL if processing_method == "Degummed" else ACCENT_PURPLE
    fig.add_trace(go.Scatter(
        x    = strains * 100,   # Convert to %
        y    = stresses,
        mode = "lines",
        name = f"Silk Fibroin ({processing_method})",
        line = dict(color=color, width=2.5),
        hovertemplate = "<b>Strain:</b> %{x:.3f}%<br><b>Stress:</b> %{y:.2f} MPa<extra></extra>",
    ))

    # ── Yield point vertical reference ──────────────────────────────────────
    yield_stress = stresses[yield_idx] if yield_idx < len(stresses) else 0
    fig.add_vline(
        x           = yield_strain * 100,
        line_dash   = "dash",
        line_color  = ACCENT_GOLD,
        line_width  = 1.5,
        annotation  = dict(
            text      = f"Yield<br>σ={yield_stress:.0f} MPa",
            font      = dict(color=ACCENT_GOLD, size=11),
            xanchor   = "left",
            yanchor   = "top",
        ),
    )

    # ── UTS horizontal reference ─────────────────────────────────────────────
    fig.add_hline(
        y           = uts_mpa,
        line_dash   = "dot",
        line_color  = ACCENT_RED,
        line_width  = 1.5,
        annotation  = dict(
            text      = f"UTS = {uts_mpa:.0f} MPa",
            font      = dict(color=ACCENT_RED, size=11),
            xanchor   = "right",
            yanchor   = "bottom",
        ),
    )

    # ── Operating point (user's applied load) ───────────────────────────────
    is_over = applied_stress > uts_mpa
    marker_color = ACCENT_RED if is_over else ACCENT_GOLD
    marker_symbol = "x" if is_over else "circle"
    label = "⚠ FRACTURE ZONE" if is_over else "Operating Point"

    fig.add_trace(go.Scatter(
        x    = [applied_strain * 100],
        y    = [min(applied_stress, uts_mpa * 1.05)],
        mode = "markers+text",
        name = label,
        marker = dict(
            symbol = marker_symbol,
            size   = 14,
            color  = marker_color,
            line   = dict(color="white", width=1.5),
        ),
        text      = [f"  {label}"],
        textfont  = dict(color=marker_color, size=11),
        textposition = "middle right",
        hovertemplate = f"<b>{label}</b><br>Strain: {applied_strain*100:.4f}%<br>Stress: {applied_stress:.2f} MPa<extra></extra>",
    ))

    # Build layout explicitly — avoid spreading LAYOUT_BASE dict which contains
    # xaxis/yaxis keys; merging those manually prevents "multiple values" TypeError.
    layout = dict(
        paper_bgcolor = DARK_BG,
        plot_bgcolor  = PANEL_BG,
        font          = dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
        margin        = dict(l=60, r=20, t=60, b=60),
        title = dict(
            text    = "📈  Stress–Strain Curve — Bombyx mori Silk Fibroin",
            font    = dict(size=16, color=TEXT_COLOR),
            x       = 0.5,
            xanchor = "center",
        ),
        xaxis = dict(
            gridcolor = GRID_COLOR,
            linecolor = GRID_COLOR,
            tickcolor = GRID_COLOR,
            showline  = True,
            mirror    = True,
            title     = dict(text="Strain  ε  (%)", font=dict(size=13)),
        ),
        yaxis = dict(
            gridcolor = GRID_COLOR,
            linecolor = GRID_COLOR,
            tickcolor = GRID_COLOR,
            showline  = True,
            mirror    = True,
            title     = dict(text="Stress  σ  (MPa)", font=dict(size=13)),
        ),
        height     = 460,
        showlegend = True,
        legend     = dict(
            bgcolor     = PANEL_BG,
            bordercolor = GRID_COLOR,
            borderwidth = 1,
            x           = 0.02,
            y           = 0.98,
            xanchor     = "left",
            yanchor     = "top",
        ),
    )
    fig.update_layout(**layout)
    return fig


# ── 2. Comparative Radar Chart ──────────────────────────────────────────────
def plot_comparison_radar() -> go.Figure:
    """
    Spider/radar chart comparing Degummed vs Non-degummed silk fibroin
    across five mechanical/biological properties.
    Scores normalized 0–10 relative to each property's typical range.
    """
    categories = [
        "Elastic Modulus", "Tensile Strength", "Elongation",
        "Biocompatibility", "Processability",
    ]
    # Normalized scores (0–10) for each material variant
    degummed_scores     = [9.5, 9.0, 5.5, 9.0, 8.0]
    non_degummed_scores = [6.5, 5.5, 9.0, 7.5, 6.0]

    cats = categories + [categories[0]]  # Close the polygon
    deg  = degummed_scores + [degummed_scores[0]]
    non  = non_degummed_scores + [non_degummed_scores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r     = deg, theta = cats,
        fill  = "toself", name = "Degummed",
        line  = dict(color=ACCENT_TEAL, width=2),
        fillcolor = "rgba(0, 212, 170, 0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r     = non, theta = cats,
        fill  = "toself", name = "Non-degummed",
        line  = dict(color=ACCENT_PURPLE, width=2),
        fillcolor = "rgba(163, 113, 247, 0.15)",
    ))
    fig.update_layout(
        polar = dict(
            bgcolor  = PANEL_BG,
            radialaxis = dict(
                visible    = True,
                range      = [0, 10],
                gridcolor  = GRID_COLOR,
                linecolor  = GRID_COLOR,
                tickfont   = dict(size=9, color=TEXT_COLOR),
            ),
            angularaxis = dict(
                gridcolor = GRID_COLOR,
                linecolor = GRID_COLOR,
                tickfont  = dict(size=11, color=TEXT_COLOR),
            ),
        ),
        paper_bgcolor = DARK_BG,
        font    = dict(family=FONT_FAMILY, color=TEXT_COLOR),
        title   = dict(
            text    = "🕸  Material Property Comparison",
            font    = dict(size=14, color=TEXT_COLOR),
            x=0.5, xanchor="center",
        ),
        legend  = dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR, borderwidth=1),
        margin  = dict(l=60, r=60, t=60, b=60),
        height  = 380,
    )
    return fig


# ── 3. Force vs. Stress Bar Gauge ──────────────────────────────────────────
def plot_stress_gauge(applied_stress: float, uts_mpa: float) -> go.Figure:
    """
    Bullet/gauge chart showing how far the applied stress is from the UTS limit.
    Colour transitions green → amber → red as the specimen approaches failure.
    """
    ratio = min(applied_stress / uts_mpa, 1.05) if uts_mpa > 0 else 0.0
    pct   = ratio * 100

    if pct < 50:
        bar_color = "#00d4aa"     # Safe — teal
    elif pct < 80:
        bar_color = "#f0b429"     # Caution — amber
    else:
        bar_color = "#f85149"     # Danger — red

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = applied_stress,
        delta  = dict(reference=uts_mpa, valueformat=".1f", suffix=" MPa"),
        number = dict(suffix=" MPa", font=dict(size=28, color=TEXT_COLOR)),
        title  = dict(
            text = "Applied Stress vs. UTS Limit",
            font = dict(size=13, color=TEXT_COLOR),
        ),
        gauge  = dict(
            axis  = dict(
                range      = [0, uts_mpa * 1.1],
                tickcolor  = TEXT_COLOR,
                tickfont   = dict(size=10),
                ticksuffix = " MPa",
            ),
            bar   = dict(color=bar_color, thickness=0.25),
            bgcolor     = PANEL_BG,
            borderwidth = 1,
            bordercolor = GRID_COLOR,
            steps = [
                dict(range=[0,             uts_mpa * 0.5],  color="rgba(0, 212, 170, 0.12)"),
                dict(range=[uts_mpa * 0.5, uts_mpa * 0.8],  color="rgba(240, 180, 41, 0.12)"),
                dict(range=[uts_mpa * 0.8, uts_mpa * 1.1],  color="rgba(248, 81, 73, 0.12)"),
            ],
            threshold = dict(
                line  = dict(color=ACCENT_RED, width=3),
                thickness = 0.75,
                value     = uts_mpa,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor = DARK_BG,
        font          = dict(family=FONT_FAMILY, color=TEXT_COLOR),
        height        = 280,
        margin        = dict(l=30, r=30, t=60, b=20),
    )
    return fig
