"""
report_generator.py
===================
CSV Report Export Module — Silk Tensile Testing Simulator
----------------------------------------------------------
Generates downloadable CSV reports containing:
  1. Specimen metadata and input parameters
  2. Computed mechanical metrics
  3. Full stress-strain curve data points

The CSV is returned as an in-memory bytes object compatible with
Streamlit's st.download_button().
"""

import io
import csv
import datetime
import numpy as np
from physics_engine import SilkSpecimen, calculate_all_metrics


def build_csv_report(
    specimen: SilkSpecimen,
    strains: np.ndarray,
    stresses: np.ndarray,
) -> bytes:
    """
    Builds a structured CSV report.

    Parameters
    ----------
    specimen  : SilkSpecimen — the test specimen with all inputs
    strains   : np.ndarray  — strain values from stress-strain curve (dimensionless)
    stresses  : np.ndarray  — stress values from stress-strain curve (MPa)

    Returns
    -------
    bytes     : UTF-8 encoded CSV bytes ready for Streamlit download
    """
    metrics = calculate_all_metrics(specimen)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output = io.StringIO()
    writer = csv.writer(output)

    # ── Header Block ────────────────────────────────────────────────────────
    writer.writerow(["SILK FIBROIN TENSILE TESTING SIMULATION REPORT"])
    writer.writerow(["Generated", timestamp])
    writer.writerow(["Material", "Silk Fibroin (Bombyx mori)"])
    writer.writerow(["Processing Method", specimen.processing_method])
    writer.writerow([])

    # ── Input Parameters ────────────────────────────────────────────────────
    writer.writerow(["─── INPUT PARAMETERS ───"])
    writer.writerow(["Parameter", "Value", "Unit"])
    writer.writerow(["Specimen Diameter",            specimen.diameter_mm,        "mm"])
    writer.writerow(["Initial Gauge Length (L₀)",   specimen.initial_length_mm,  "mm"])
    writer.writerow(["Applied Force",                specimen.applied_force_N,    "N"])
    writer.writerow(["Cross-Sectional Area",         metrics["cross_section_area_mm2"], "mm²"])
    writer.writerow([])

    # ── Computed Mechanical Metrics ──────────────────────────────────────────
    writer.writerow(["─── COMPUTED MECHANICAL METRICS ───"])
    writer.writerow(["Metric", "Value", "Unit"])
    writer.writerow(["Tensile Stress (σ)",           metrics["tensile_stress_mpa"],      "MPa"])
    writer.writerow(["Tensile Strain (ε)",           metrics["tensile_strain"],           "dimensionless"])
    writer.writerow(["ΔL (Extension)",               metrics["delta_length_mm"],          "mm"])
    writer.writerow(["% Elongation",                 metrics["percent_elongation"],       "%"])
    writer.writerow(["Elastic Modulus (E)",          metrics["elastic_modulus_gpa"],      "GPa"])
    writer.writerow(["Ultimate Tensile Strength",    metrics["uts_mpa"],                  "MPa"])
    writer.writerow(["Yield Strain",                 metrics["yield_strain_pct"],         "%"])
    writer.writerow(["Elongation at Break",          metrics["elongation_at_break_pct"],  "%"])
    writer.writerow(["Fracture Risk",                "YES" if metrics["is_above_uts"] else "NO", ""])
    writer.writerow([])

    # ── Governing Equations ─────────────────────────────────────────────────
    writer.writerow(["─── GOVERNING EQUATIONS ───"])
    writer.writerow(["Equation", "Formula", "Reference"])
    writer.writerow(["Tensile Stress",    "σ = F / A₀",          "Engineering mechanics"])
    writer.writerow(["Tensile Strain",    "ε = σ / E  (elastic)", "Hooke's Law"])
    writer.writerow(["Extension",         "ΔL = ε × L₀",         "Continuum mechanics"])
    writer.writerow(["Elastic Modulus E", "E = σ / ε",           "Linear elastic model"])
    writer.writerow([])

    # ── Full Stress-Strain Curve Data ────────────────────────────────────────
    writer.writerow(["─── STRESS-STRAIN CURVE DATA ───"])
    writer.writerow(["Strain (dimensionless)", "Strain (%)", "Stress (MPa)"])
    for eps, sig in zip(strains, stresses):
        writer.writerow([f"{eps:.6f}", f"{eps*100:.4f}", f"{sig:.4f}"])

    return output.getvalue().encode("utf-8")
