"""
app.py
======
Silk Biomaterial Tensile Testing Simulator
==========================================
Subject: Biomaterials and Artificial Organs
Material: Silk Fibroin (Bombyx mori)

Entry point for the Streamlit dashboard. Integrates:
  - physics_engine.py  → mechanical calculations
  - charts.py          → Plotly visualizations
  - report_generator.py → CSV export

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np

# ── Local modules ────────────────────────────────────────────────────────────
from physics_engine import SilkSpecimen, generate_stress_strain_curve, calculate_all_metrics
from charts import plot_stress_strain_curve, plot_comparison_radar, plot_stress_gauge
from report_generator import build_csv_report

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Silk Tensile Testing Simulator",
    page_icon   = "🧵",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Dark professional theme
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-main:     #0d1117;
    --bg-panel:    #161b22;
    --bg-card:     #1c2128;
    --border:      #30363d;
    --teal:        #00d4aa;
    --gold:        #f0b429;
    --red:         #f85149;
    --blue:        #58a6ff;
    --purple:      #a371f7;
    --text-main:   #e6edf3;
    --text-muted:  #8b949e;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: border-color .25s;
}
div[data-testid="metric-container"]:hover { border-color: var(--teal) !important; }
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: .05em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
}

/* ── Inputs & sliders ── */
input, textarea, select,
div[data-baseweb="input"] input,
div[data-baseweb="select"] * {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
}
.stSlider [data-testid="stThumbValue"] { color: var(--teal) !important; }
.stSlider .st-bk { background: var(--teal) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Alert / info boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #00d4aa22, #00d4aa44) !important;
    border: 1px solid var(--teal) !important;
    color: var(--teal) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: .04em !important;
    transition: all .3s !important;
    padding: 0.6rem 1.4rem !important;
}
.stDownloadButton > button:hover {
    background: var(--teal) !important;
    color: #0d1117 !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00a896) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.5rem !important;
    letter-spacing: .05em !important;
    transition: opacity .25s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Expander ── */
details { 
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px 12px !important;
}
summary { color: var(--teal) !important; font-weight: 600 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: 10px !important;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--teal) !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } 
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Section heading helper ── */
.section-heading {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
    margin-top: 16px;
}

/* ── Bio card ── */
.bio-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--teal);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
    line-height: 1.7;
}
.bio-card h4 { color: var(--teal); margin: 0 0 8px; font-size: 1rem; }
.bio-card p  { color: var(--text-muted); margin: 0; font-size: 0.87rem; }

/* ── Warning banner ── */
.fracture-banner {
    background: rgba(248,81,73,0.12);
    border: 1px solid var(--red);
    border-radius: 10px;
    padding: 14px 20px;
    color: var(--red);
    font-weight: 600;
    letter-spacing: .03em;
    font-size: 0.95rem;
    text-align: center;
}

/* ── Safe banner ── */
.safe-banner {
    background: rgba(0,212,170,0.08);
    border: 1px solid var(--teal);
    border-radius: 10px;
    padding: 14px 20px;
    color: var(--teal);
    font-weight: 600;
    font-size: 0.9rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Input Parameters
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧵 Silk Fibroin Simulator")
    st.caption("Bombyx mori · Tensile Testing Engine")
    st.divider()

    # ── Material (fixed — this simulator is silk-specific) ──────────────────
    st.markdown('<p class="section-heading">Material</p>', unsafe_allow_html=True)
    material = st.selectbox(
        "Biomaterial",
        ["Silk Fibroin (Bombyx mori)"],
        help="This simulator is calibrated specifically for B. mori silk fibroin.",
        key="material_select",
    )

    # ── Processing Method ────────────────────────────────────────────────────
    st.markdown('<p class="section-heading">Processing Method</p>', unsafe_allow_html=True)
    processing_method = st.radio(
        "Preparation",
        ["Degummed", "Non-degummed"],
        help=(
            "**Degummed:** Sericin coating removed via alkaline boiling. "
            "Increases E and UTS by exposing pure fibroin core.\n\n"
            "**Non-degummed:** Sericin intact. Lower stiffness but higher elongation."
        ),
        index=0,
        key="proc_method",
    )

    st.divider()

    # ── Specimen Geometry ────────────────────────────────────────────────────
    st.markdown('<p class="section-heading">Specimen Geometry</p>', unsafe_allow_html=True)

    diameter_mm = st.number_input(
        "Specimen Diameter (mm)",
        min_value=0.01,
        max_value=50.0,
        value=1.0,
        step=0.01,
        format="%.3f",
        help="Circular cross-section diameter of the silk specimen or fibre bundle.",
        key="diameter",
    )

    initial_length_mm = st.number_input(
        "Initial Gauge Length  L₀  (mm)",
        min_value=1.0,
        max_value=500.0,
        value=50.0,
        step=0.5,
        format="%.1f",
        help="Free length between grips before loading. ASTM D2101 recommends ≥50 mm.",
        key="gauge_length",
    )

    st.divider()

    # ── Loading ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-heading">Applied Load</p>', unsafe_allow_html=True)

    applied_force_N = st.number_input(
        "Applied Force  F  (N)",
        min_value=0.0,
        max_value=50000.0,
        value=10.0,
        step=0.1,
        format="%.2f",
        help="Axial tensile force applied to the specimen.",
        key="force",
    )

    st.divider()

    # ── Elastic Modulus override ─────────────────────────────────────────────
    st.markdown('<p class="section-heading">Advanced</p>', unsafe_allow_html=True)
    with st.expander("Override Elastic Modulus", expanded=False):
        default_E = 15.0 if processing_method == "Degummed" else 10.5
        custom_E = st.slider(
            "Elastic Modulus  E  (GPa)",
            min_value=5.0,
            max_value=25.0,
            value=default_E,
            step=0.5,
            help="Default: 15 GPa (Degummed), 10.5 GPa (Non-degummed). Adjust to match your experimental data.",
            key="custom_E",
        )
    st.divider()
    st.caption("© 2026 · Biomaterials & Artificial Organs · Silk Testing Module")


# ════════════════════════════════════════════════════════════════════════════
#  PHYSICS ENGINE — Build specimen & compute
# ════════════════════════════════════════════════════════════════════════════

# Create specimen object (modular data class from physics_engine.py)
specimen = SilkSpecimen(
    diameter_mm       = diameter_mm,
    initial_length_mm = initial_length_mm,
    applied_force_N   = applied_force_N,
    processing_method = processing_method,
)

# Temporarily override E if user changed the slider
import physics_engine as _pe
if processing_method == "Degummed":
    _saved = _pe.ELASTIC_MODULUS_DEGUMMED_GPA
    _pe.ELASTIC_MODULUS_DEGUMMED_GPA = custom_E
else:
    _saved = _pe.ELASTIC_MODULUS_NON_DEGUMMED_GPA
    _pe.ELASTIC_MODULUS_NON_DEGUMMED_GPA = custom_E

# Recompute after potential E override
specimen = SilkSpecimen(
    diameter_mm       = diameter_mm,
    initial_length_mm = initial_length_mm,
    applied_force_N   = applied_force_N,
    processing_method = processing_method,
)

metrics = calculate_all_metrics(specimen)
strains, stresses = generate_stress_strain_curve(specimen)

# Restore module-level constant
if processing_method == "Degummed":
    _pe.ELASTIC_MODULUS_DEGUMMED_GPA = _saved
else:
    _pe.ELASTIC_MODULUS_NON_DEGUMMED_GPA = _saved


# ════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("# 🧵 Silk Biomaterial Tensile Testing Simulator")
    st.markdown(
        f"**Material:** Silk Fibroin *(Bombyx mori)* &nbsp;|&nbsp; "
        f"**Method:** {processing_method} &nbsp;|&nbsp; "
        f"**E =** {custom_E} GPa",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = build_csv_report(specimen, strains, stresses)
    st.download_button(
        label     = "⬇ Download Report",
        data      = csv_bytes,
        file_name = f"silk_tensile_report_{processing_method.lower()}.csv",
        mime      = "text/csv",
        key       = "dl_report",
    )

st.divider()

# ── Fracture / Safe status banner ───────────────────────────────────────────
if metrics["is_above_uts"]:
    st.markdown(
        f'<div class="fracture-banner">⚠️  FRACTURE ALERT — Applied stress '
        f'({metrics["tensile_stress_mpa"]:.2f} MPa) exceeds the Ultimate Tensile Strength '
        f'({metrics["uts_mpa"]:.0f} MPa). Specimen would rupture under this load.</div>',
        unsafe_allow_html=True,
    )
else:
    load_pct = metrics["tensile_stress_mpa"] / metrics["uts_mpa"] * 100
    st.markdown(
        f'<div class="safe-banner">✅  Specimen is within safe operating range — '
        f'{load_pct:.1f}% of UTS utilised.</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — KEY METRIC CARDS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-heading">🔬 Computed Mechanical Metrics</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Tensile Stress  σ",
    f"{metrics['tensile_stress_mpa']:.3f} MPa",
    delta="σ = F / A₀",
    help="Engineering stress: force per unit original cross-sectional area.",
)
c2.metric(
    "Tensile Strain  ε",
    f"{metrics['tensile_strain']:.5f}",
    delta="ε = σ / E",
    help="Dimensionless engineering strain derived from Hooke's Law.",
)
c3.metric(
    "Elastic Modulus  E",
    f"{metrics['elastic_modulus_gpa']:.2f} GPa",
    delta=f"{processing_method}",
    help="Stiffness — slope of the linear elastic region of the stress-strain curve.",
)
c4.metric(
    "% Elongation",
    f"{metrics['percent_elongation']:.4f}%",
    delta=f"ΔL = {metrics['delta_length_mm']:.4f} mm",
    help="Percentage extension of the gauge length under applied load.",
)
c5.metric(
    "UTS Limit",
    f"{metrics['uts_mpa']:.0f} MPa",
    delta="Fracture threshold",
    help="Ultimate Tensile Strength — maximum stress before fracture.",
)
c6.metric(
    "Cross-Section  A₀",
    f"{metrics['cross_section_area_mm2']:.4f} mm²",
    delta=f"d = {diameter_mm} mm",
    help="A₀ = π(d/2)² — original circular cross-sectional area.",
)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — TABS (Stress-Strain | Comparison | Gauge | Theory)
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Stress–Strain Curve",
    "🕸  Material Comparison",
    "⚡  Stress Gauge",
    "📚  Physics Reference",
])

# ── Tab 1: Stress-Strain Curve ───────────────────────────────────────────────
with tab1:
    fig_ss = plot_stress_strain_curve(
        strains            = strains,
        stresses           = stresses,
        applied_strain     = metrics["tensile_strain"],
        applied_stress     = metrics["tensile_stress_mpa"],
        processing_method  = processing_method,
        uts_mpa            = metrics["uts_mpa"],
        yield_strain       = specimen.yield_strain,
    )
    st.plotly_chart(fig_ss, use_container_width=True)

    with st.expander("📋 Curve Data Table (first 30 points)"):
        import pandas as pd
        df_curve = pd.DataFrame({
            "Strain (dimensionless)": strains[::max(1, len(strains)//30)][:30],
            "Strain (%)":            (strains * 100)[::max(1, len(strains)//30)][:30],
            "Stress (MPa)":          stresses[::max(1, len(stresses)//30)][:30],
        })
        st.dataframe(df_curve.style.format("{:.5f}"), use_container_width=True)

# ── Tab 2: Radar Comparison ──────────────────────────────────────────────────
with tab2:
    col_radar, col_compare = st.columns([3, 2])
    with col_radar:
        st.plotly_chart(plot_comparison_radar(), use_container_width=True)
    with col_compare:
        st.markdown("### Degummed vs. Non-degummed")
        import pandas as pd
        df_comp = pd.DataFrame({
            "Property":                  ["Elastic Modulus", "UTS", "Elongation at Break", "Yield Strain"],
            "Degummed":                  ["15.0 GPa",        "650 MPa", "15%",              "2.5%"],
            "Non-degummed":              ["10.5 GPa",        "380 MPa", "28%",              "3.5%"],
            "Reason":                    [
                "Sericin removal ↑ crystallinity",
                "Denser beta-sheet network",
                "Sericin adds chain mobility",
                "Sericin delays yield",
            ],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.info(
            "**Why degum?** Sericin triggers immune responses in vivo. "
            "Degummed silk fibroin is the gold-standard form for implantable "
            "scaffolds and vascular grafts (ISO 10993 biocompatibility requirements)."
        )

# ── Tab 3: Stress Gauge ──────────────────────────────────────────────────────
with tab3:
    col_g1, col_g2 = st.columns([2, 3])
    with col_g1:
        st.plotly_chart(
            plot_stress_gauge(metrics["tensile_stress_mpa"], metrics["uts_mpa"]),
            use_container_width=True,
        )
    with col_g2:
        st.markdown("### Load Analysis")
        load_ratio = metrics["tensile_stress_mpa"] / metrics["uts_mpa"]

        progress_color = (
            "🟥" if load_ratio > 0.8 else
            "🟨" if load_ratio > 0.5 else
            "🟩"
        )
        st.markdown(f"**Stress Utilisation:** {load_ratio*100:.2f}%  {progress_color}")
        st.progress(min(load_ratio, 1.0))
        st.markdown("<br>", unsafe_allow_html=True)

        col_l1, col_l2 = st.columns(2)
        col_l1.metric("Applied Stress",  f"{metrics['tensile_stress_mpa']:.4f} MPa")
        col_l1.metric("Remaining Margin", f"{max(metrics['uts_mpa'] - metrics['tensile_stress_mpa'], 0):.2f} MPa")
        col_l2.metric("UTS (limit)",      f"{metrics['uts_mpa']:.0f} MPa")
        col_l2.metric("Safety Factor",
                      f"{metrics['uts_mpa']/metrics['tensile_stress_mpa']:.2f}×"
                      if metrics['tensile_stress_mpa'] > 0 else "∞")

# ── Tab 4: Physics Reference ─────────────────────────────────────────────────
with tab4:
    st.markdown("### Governing Equations")
    col_eq1, col_eq2 = st.columns(2)

    with col_eq1:
        st.markdown(r"""
**Engineering Tensile Stress**
$$\sigma = \frac{F}{A_0}$$
- $\sigma$ = Tensile stress (Pa or MPa)
- $F$ = Applied axial force (N)
- $A_0$ = Original cross-sectional area (m² or mm²)

---

**Engineering Tensile Strain**
$$\varepsilon = \frac{\Delta L}{L_0} = \frac{\sigma}{E}$$
- $\varepsilon$ = Dimensionless strain
- $\Delta L$ = Extension (mm)
- $L_0$ = Original gauge length (mm)
- $E$ = Elastic Modulus (GPa)
""")

    with col_eq2:
        st.markdown(r"""
**Hooke's Law (Linear Elastic)**
$$\sigma = E \cdot \varepsilon$$
- Valid in the elastic region (before yield point)
- Slope of the linear region of the stress-strain curve equals $E$

---

**Cross-Sectional Area (Cylinder)**
$$A_0 = \pi \left(\frac{d}{2}\right)^2$$
- $d$ = Specimen diameter (mm)

---

**% Elongation**
$$\% \text{ Elongation} = \varepsilon \times 100$$
""")

    st.divider()

    # Simulation parameters table
    st.markdown("### Current Simulation Parameters")
    import pandas as pd
    df_params = pd.DataFrame({
        "Parameter":   [
            "Material", "Processing Method", "Specimen Diameter",
            "Gauge Length (L₀)", "Applied Force", "Cross-Section Area (A₀)",
            "Elastic Modulus (E)", "Tensile Stress (σ)", "Tensile Strain (ε)",
            "ΔL (extension)", "% Elongation", "UTS Limit",
        ],
        "Value": [
            "Silk Fibroin (Bombyx mori)", processing_method,
            f"{diameter_mm} mm", f"{initial_length_mm} mm",
            f"{applied_force_N} N", f"{metrics['cross_section_area_mm2']:.6f} mm²",
            f"{metrics['elastic_modulus_gpa']} GPa",
            f"{metrics['tensile_stress_mpa']:.4f} MPa",
            f"{metrics['tensile_strain']:.6f}",
            f"{metrics['delta_length_mm']:.4f} mm",
            f"{metrics['percent_elongation']:.4f}%",
            f"{metrics['uts_mpa']:.0f} MPa",
        ],
    })
    st.dataframe(df_params, use_container_width=True, hide_index=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — BIOCOMPATIBILITY SUMMARY
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-heading">🏥 Biocompatibility Summary — Silk Fibroin in Artificial Organs</p>',
            unsafe_allow_html=True)

bios = [
    (
        "🫀 Vascular Grafts",
        "Small-diameter (<6 mm) vascular grafts fabricated from electrospun silk fibroin "
        "exhibit compliance matching to native blood vessels, reducing intimal hyperplasia. "
        "The β-sheet structure provides suture retention strength, while the amorphous regions "
        "allow compliance under pulsatile blood pressure (80–120 mmHg). Degummed fibres "
        "minimise thrombogenicity by removing sericin-associated complement activation.",
    ),
    (
        "🩹 Skin Scaffolds & Wound Healing",
        "Silk fibroin 3D scaffolds support keratinocyte and fibroblast adhesion via RGD-like "
        "binding motifs. The controllable degradation rate (weeks to months via protease XIV) "
        "matches the wound re-epithelialisation timeline. Semi-permeability allows oxygen and "
        "nutrient diffusion while maintaining a moist wound microenvironment.",
    ),
    (
        "🦴 Bone & Cartilage Tissue Engineering",
        "Silk-hydroxyapatite composite scaffolds mimic the organic-inorganic hierarchy of "
        "native cortical bone. Compressive moduli of 0.5–10 MPa (sponge form) or up to "
        "1.5 GPa (dense fibre) are achievable through processing control. The slow "
        "in vivo degradation (>12 months) provides long-term structural support during osteogenesis.",
    ),
    (
        "👁️ Corneal & Ocular Implants",
        "Silk fibroin films of 1–30 μm thickness are optically transparent (>90% transmittance "
        "at 550 nm) and mechanically tunable for anterior segment prosthetics. Surface "
        "modification with fibronectin or RGD peptides promotes corneal epithelial cell "
        "attachment without immunogenic response — critical for avascular tissue applications.",
    ),
    (
        "🧠 Neural Scaffolds",
        "Porous silk conduits (~10 mm length, 1.5 mm inner diameter) bridge peripheral nerve "
        "gaps by providing physical guidance for Schwann cell migration. Electrical conductivity "
        "can be enhanced via graphene oxide coating while maintaining the biocompatibility profile "
        "approved under ISO 10993 biological evaluation standards.",
    ),
]

cols = st.columns(2)
for i, (title, body) in enumerate(bios):
    with cols[i % 2]:
        st.markdown(
            f'<div class="bio-card"><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="text-align:center; color:#8b949e; font-size:0.78rem;">'
    '🧵 &nbsp; Silk Biomaterial Tensile Testing Simulator &nbsp;·&nbsp; '
    'Biomaterials &amp; Artificial Organs &nbsp;·&nbsp; '
    'Physics: σ = E·ε &nbsp;·&nbsp; Material: Bombyx mori Silk Fibroin'
    '</p>',
    unsafe_allow_html=True,
)
