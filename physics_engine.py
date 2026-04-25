"""
physics_engine.py
=================
Silk Fibroin Tensile Testing Physics Engine
--------------------------------------------
This module encapsulates all material science calculations for the
tensile testing simulation of Silk Fibroin (Bombyx mori).

Material Science Background:
-----------------------------
Silk Fibroin is a semi-crystalline biopolymer composed of beta-sheets
(crystalline domains) connected by amorphous regions. Its mechanical
behavior follows a two-phase stress-strain response:
  1. Linear elastic region  → governed by Hooke's Law: σ = E·ε
  2. Post-yield / fracture  → plastic deformation and final rupture

Key literature values for Bombyx mori Silk Fibroin:
  - Elastic Modulus (E):     10–17 GPa  (default: 15 GPa)
  - Ultimate Tensile Strength: 300–700 MPa
  - Elongation at Break:     10–35%
  - Degumming increases E and UTS by removing sericin coating.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ── Material Constants ──────────────────────────────────────────────────────
ELASTIC_MODULUS_DEGUMMED_GPA      = 15.0   # GPa  — sericin removed
ELASTIC_MODULUS_NON_DEGUMMED_GPA  = 10.5   # GPa  — sericin present (lower E)
UTS_DEGUMMED_MPA                  = 650.0  # MPa  — ultimate tensile strength
UTS_NON_DEGUMMED_MPA              = 380.0  # MPa
YIELD_STRAIN_DEGUMMED             = 0.025  # 2.5% strain at yield point
YIELD_STRAIN_NON_DEGUMMED         = 0.035
ELONGATION_AT_BREAK_DEGUMMED      = 0.15   # 15%
ELONGATION_AT_BREAK_NON_DEGUMMED  = 0.28   # 28%


@dataclass
class SilkSpecimen:
    """
    Represents a single silk fibroin test specimen with its geometry
    and loading conditions.
    """
    diameter_mm: float        # Cross-section diameter of the fibre/specimen
    initial_length_mm: float  # Gauge length (L₀) before loading
    applied_force_N: float    # Applied axial tensile force
    processing_method: str    # 'Degummed' or 'Non-degummed'

    # ── Derived geometry ────────────────────────────────────────────────────
    @property
    def radius_m(self) -> float:
        """Convert diameter (mm) → radius (m) for SI unit compliance."""
        return (self.diameter_mm / 2.0) / 1000.0

    @property
    def cross_sectional_area_m2(self) -> float:
        """
        Cross-sectional area assuming circular geometry:
            A = π · r²
        Unit: m²
        """
        return math.pi * (self.radius_m ** 2)

    @property
    def cross_sectional_area_mm2(self) -> float:
        """Area in mm² for display convenience."""
        return math.pi * ((self.diameter_mm / 2.0) ** 2)

    # ── Elastic Modulus selection ────────────────────────────────────────────
    @property
    def elastic_modulus_gpa(self) -> float:
        """
        Return the Elastic Modulus (E) based on processing method.
        Degumming removes sericin protein, increasing crystallinity and
        thus stiffness of the remaining fibroin core.
        """
        if self.processing_method == "Degummed":
            return ELASTIC_MODULUS_DEGUMMED_GPA
        return ELASTIC_MODULUS_NON_DEGUMMED_GPA

    @property
    def elastic_modulus_mpa(self) -> float:
        """E converted to MPa (1 GPa = 1000 MPa)."""
        return self.elastic_modulus_gpa * 1000.0

    # ── Primary Mechanical Calculations ─────────────────────────────────────
    @property
    def tensile_stress_mpa(self) -> float:
        """
        Engineering Tensile Stress (σ):
            σ = F / A₀
        where F = applied force (N), A₀ = original cross-sectional area (m²)
        Result converted from Pa → MPa (÷ 1,000,000)
        """
        area_m2 = self.cross_sectional_area_m2
        if area_m2 == 0:
            return 0.0
        stress_pa = self.applied_force_N / area_m2
        return stress_pa / 1e6  # Pa → MPa

    @property
    def tensile_strain(self) -> float:
        """
        Engineering Tensile Strain (ε) derived from linear elastic model:
            ε = σ / E
        This gives the strain corresponding to the applied stress
        under the assumption of linear elasticity (valid below yield point).
        Dimensionless.
        """
        if self.elastic_modulus_mpa == 0:
            return 0.0
        return self.tensile_stress_mpa / self.elastic_modulus_mpa

    @property
    def delta_length_mm(self) -> float:
        """
        Change in gauge length (ΔL):
            ΔL = ε × L₀
        Unit: mm
        """
        return self.tensile_strain * self.initial_length_mm

    @property
    def percent_elongation(self) -> float:
        """
        % Elongation at current load:
            % Elongation = ε × 100
        """
        return self.tensile_strain * 100.0

    @property
    def uts_mpa(self) -> float:
        """Ultimate Tensile Strength limit for this specimen type (MPa)."""
        if self.processing_method == "Degummed":
            return UTS_DEGUMMED_MPA
        return UTS_NON_DEGUMMED_MPA

    @property
    def yield_strain(self) -> float:
        """Yield strain for this specimen type (dimensionless)."""
        if self.processing_method == "Degummed":
            return YIELD_STRAIN_DEGUMMED
        return YIELD_STRAIN_NON_DEGUMMED

    @property
    def elongation_at_break(self) -> float:
        """Total strain at fracture (dimensionless)."""
        if self.processing_method == "Degummed":
            return ELONGATION_AT_BREAK_DEGUMMED
        return ELONGATION_AT_BREAK_NON_DEGUMMED

    @property
    def is_above_uts(self) -> bool:
        """Check if applied stress exceeds the UTS (specimen would fracture)."""
        return self.tensile_stress_mpa > self.uts_mpa


# ── Full Stress-Strain Curve Generator ─────────────────────────────────────
def generate_stress_strain_curve(
    specimen: SilkSpecimen,
    num_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a realistic multi-region stress-strain curve for silk fibroin.

    The curve models three physiological regions:
    ─────────────────────────────────────────────────────────────────────────
    Region 1 │ Linear Elastic (0 → yield strain)
              │   σ = E · ε   (Hooke's Law)
    ─────────────────────────────────────────────────────────────────────────
    Region 2 │ Strain Hardening (yield → 0.8 × break strain)
              │   Non-linear, models beta-sheet alignment / crystallisation
              │   Uses square-root approximation of work-hardening
    ─────────────────────────────────────────────────────────────────────────
    Region 3 │ Necking & Fracture (≥ 80% break strain → break)
              │   Rapid stress drop simulating catastrophic fibre failure
    ─────────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    specimen   : SilkSpecimen instance
    num_points : resolution of the curve

    Returns
    -------
    strains    : np.ndarray (dimensionless)
    stresses   : np.ndarray (MPa)
    """
    E_mpa        = specimen.elastic_modulus_mpa
    uts          = specimen.uts_mpa
    eps_yield    = specimen.yield_strain
    eps_break    = specimen.elongation_at_break
    sigma_yield  = E_mpa * eps_yield       # Stress at yield point (MPa)

    strains  = np.linspace(0, eps_break, num_points)
    stresses = np.zeros(num_points)

    eps_harden_end = eps_break * 0.80      # Hardening → 80% of break strain
    eps_neck_start = eps_harden_end
    eps_neck_end   = eps_break

    for i, eps in enumerate(strains):
        if eps <= eps_yield:
            # ── Region 1: Linear elastic (σ = E·ε) ──
            stresses[i] = E_mpa * eps

        elif eps <= eps_harden_end:
            # ── Region 2: Strain hardening ──
            # Maps yield-stress → UTS via a sqrt work-hardening approximation
            t = (eps - eps_yield) / (eps_harden_end - eps_yield)
            stresses[i] = sigma_yield + (uts - sigma_yield) * np.sqrt(t)

        else:
            # ── Region 3: Necking / fracture softening ──
            # Linear stress drop from UTS → 0 as specimen ruptures
            t = (eps - eps_neck_start) / (eps_neck_end - eps_neck_start)
            stresses[i] = uts * (1.0 - t)

    return strains, stresses


def calculate_all_metrics(specimen: SilkSpecimen) -> dict:
    """
    Aggregate all computed metrics for a given specimen into a dictionary.
    Used by the dashboard to populate metric cards and tables.
    """
    return {
        "tensile_stress_mpa":       round(specimen.tensile_stress_mpa,   4),
        "tensile_strain":           round(specimen.tensile_strain,         6),
        "delta_length_mm":          round(specimen.delta_length_mm,        4),
        "percent_elongation":       round(specimen.percent_elongation,     4),
        "elastic_modulus_gpa":      round(specimen.elastic_modulus_gpa,    2),
        "elastic_modulus_mpa":      round(specimen.elastic_modulus_mpa,    1),
        "cross_section_area_mm2":   round(specimen.cross_sectional_area_mm2, 6),
        "uts_mpa":                  round(specimen.uts_mpa,                1),
        "yield_strain_pct":         round(specimen.yield_strain * 100,     2),
        "elongation_at_break_pct":  round(specimen.elongation_at_break * 100, 1),
        "is_above_uts":             specimen.is_above_uts,
        "processing_method":        specimen.processing_method,
    }
