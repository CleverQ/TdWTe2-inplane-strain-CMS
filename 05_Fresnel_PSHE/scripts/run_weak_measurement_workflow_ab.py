from __future__ import annotations

import argparse
import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WORKFLOW_ROOT.parents[1]
TABLES = WORKFLOW_ROOT / "tables"
FIGURES = WORKFLOW_ROOT / "figures"
REPORTS = WORKFLOW_ROOT / "reports"
MANUSCRIPT = WORKFLOW_ROOT / "manuscript_text"
LOGS = WORKFLOW_ROOT / "logs"
os.environ.setdefault("MPLCONFIGDIR", str(LOGS / "matplotlib_cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import PchipInterpolator


SUPPORTED_MODES = ("sanity_0pct", "layered_0pct", "full_ab_workflow")
LAMBDA_NM = 632.8
THICKNESS_M = 10e-9
W0_M = 14.5e-6
Z_M = 0.1391
N_INCIDENT = 1.0
N_SUBSTRATE = 1.515
DX_DEG = 0.001
GAMMA_I_PROVISIONAL_DEG = 0.0
POST_SELECTION_PROVISIONAL_DEG = 89.0
PARAMETER_STATUS = "PROVISIONAL_NEED_USER_CONFIRM"
SENSITIVITY_STATUS = "NEED_USER_CONFIRM"
DETECTOR_GAMMA_I_DEG = 45.0
GAMMA_I_SENSITIVITY_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 89.0)
POST_SELECTION_SENSITIVITY_DEG = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
SENSITIVITY_Y_SCALE = "log"
POST_SELECTION_X_SCALE = "log"
FRESNEL_TOL = 1e-12
FIXED_THETA_DEG = np.round(np.arange(30.0, 89.9 + 1e-12, 0.02), 10)
SCREEN_THETA_DEG = np.round(np.arange(55.0, 85.0 + 1e-12, 0.10), 10)
BASE_LAMBDA_SCREEN_NM = np.arange(300.0, 2000.0 + 1e-12, 2.0)

OPTIC_GRIDS_MAT = Path(
    os.environ.get(
        "PSHE_OPTIC_GRIDS_MAT",
        str(Path(__file__).resolve().parents[2] / "04_Optical_conductivity" / "data" / "processed" / "grids.mat"),
    )
)
ARCHIVE = PROJECT_ROOT / "archive_current_Chen_PSHE_consistency_0pct_632p8nm"
V9_FRESNEL_REFERENCE = ARCHIVE / "tables" / "current_vs_Chen_fresnel_from_sigma_0pct_632p8nm_v9.csv"
REFERENCE_FILES = [
    V9_FRESNEL_REFERENCE,
    ARCHIVE / "scripts" / "generate_current_vs_chen_from_sigma_v9.py",
    ARCHIVE / "scripts" / "compare_current_vs_Chen_pshe_models_ab_v6_0pct.py",
    ARCHIVE / "reports" / "input_convention_verification.md",
    REPORTS / "analysis_sigma2PSHE_weak_input_audit.md",
    REPORTS / "recovered_full_workflow_logic.md",
]

LAYERED_OUTPUT_COLUMNS = (
    "theta_deg", "lambda_nm", "branch",
    "delta_y_spatial_m", "delta_y_spatial_nm",
    "Theta_y_detector_m", "Theta_y_detector_nm", "Theta_y_detector_um",
    "detector_y_m", "detector_y_nm", "detector_y_um",
    "y_weak_m", "y_weak_nm", "y_weak_um", "y_weak_mm",
    "enhancement_detector_over_spatial", "enhancement_weak_over_spatial",
    "detector_gamma_i_deg", "weak_gamma_i_deg", "post_selection_angle_deg",
    "sigma_i_branch", "parameter_status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_environment() -> None:
    for path in [OPTIC_GRIDS_MAT, *REFERENCE_FILES]:
        if not path.exists():
            raise FileNotFoundError(f"Required read-only source is missing: {path}")
    for path in (TABLES, FIGURES, REPORTS, MANUSCRIPT, LOGS):
        path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=2)
def load_grids(path_text: str) -> dict[str, Any]:
    raw = sio.loadmat(Path(path_text), squeeze_me=True, struct_as_record=False)
    required = {
        "omega_eV", "lambda_nm", "eps_percent", "direction", "case_labels", "source_dir",
        "Re_xx", "Im_xx", "Re_yy", "Im_yy", "Re_xy", "Im_xy", "Re_yx", "Im_yx",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise KeyError(f"Missing grids.mat fields: {missing}")
    return raw


def axis_indices(raw: dict[str, Any], axis: str) -> np.ndarray:
    if axis not in {"a", "b"}:
        raise ValueError("axis must be 'a' or 'b'")
    eps = np.asarray(raw["eps_percent"], dtype=float).ravel()
    direction = np.asarray(raw["direction"], dtype=object).ravel().astype(str)
    mask = (direction == axis) | ((direction == "ref") & np.isclose(eps, 0.0, rtol=0.0, atol=1e-12))
    indices = np.where(mask)[0]
    indices = indices[np.argsort(eps[indices])]
    if len(indices) != 6 or not np.allclose(eps[indices], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        raise RuntimeError(f"Unexpected {axis}-axis strain cases: {eps[indices]}")
    return indices


def load_axis_sigma_s_cm(path: Path, axis: str, lambda_nm: float) -> dict[str, Any]:
    raw = load_grids(str(path))
    indices = axis_indices(raw, axis)
    eps = np.asarray(raw["eps_percent"], dtype=float).ravel()[indices]
    omega = np.asarray(raw["omega_eV"], dtype=float).ravel()
    order = np.argsort(omega)
    energy_eV = 1239.841984 / float(lambda_nm)
    if energy_eV < omega[order][0] or energy_eV > omega[order][-1]:
        raise ValueError(f"lambda={lambda_nm} nm lies outside optical-conductivity coverage")
    sigma: dict[str, np.ndarray] = {}
    for component in ("xx", "yy", "xy", "yx"):
        real_grid = np.asarray(raw[f"Re_{component}"], dtype=float)[indices, :]
        imag_grid = np.asarray(raw[f"Im_{component}"], dtype=float)[indices, :]
        values = []
        for row in range(len(indices)):
            re = float(PchipInterpolator(omega[order], real_grid[row, order], extrapolate=False)(energy_eV))
            im = float(PchipInterpolator(omega[order], imag_grid[row, order], extrapolate=False)(energy_eV))
            values.append(re + 1j * im)
        sigma[component] = np.asarray(values, dtype=complex)
    labels = np.asarray(raw["case_labels"], dtype=object).ravel().astype(str)
    sources = np.asarray(raw["source_dir"], dtype=object).ravel().astype(str)
    return {
        "axis": axis,
        "lambda_nm": float(lambda_nm),
        "energy_eV": energy_eV,
        "strain_percent": eps,
        "case_label": labels[indices],
        "source_dir": sources[indices],
        "sigma_bulk_s_cm": sigma,
    }


def bulk_s_cm_to_sheet_s(sigma_bulk_s_cm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    bulk_s_m = {key: np.asarray(value, dtype=complex) * 100.0 for key, value in sigma_bulk_s_cm.items()}
    sheet = {f"s_{key}": value * THICKNESS_M for key, value in bulk_s_m.items()}
    return {
        **{f"bulk_{key}_S_m": value for key, value in bulk_s_m.items()},
        **sheet,
        "sigma_pp": sheet["s_xx"],
        "sigma_ss": sheet["s_yy"],
        "sigma_ps": (sheet["s_xy"] + sheet["s_yx"]) / 2.0,
        "sigma_sym": (sheet["s_xy"] - sheet["s_yx"]) / 2.0,
    }


def chen_fresnel_matrix(theta_deg: np.ndarray, lambda_nm: float, sheet: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))[:, None]
    eps0 = 8.854187817e-12
    mu0 = 4.0 * math.pi * 1e-7
    c0 = 1.0 / math.sqrt(mu0 * eps0)
    k0 = 2.0 * math.pi / (lambda_nm * 1e-9)
    omega = k0 * c0
    n_rel = N_SUBSTRATE / N_INCIDENT
    epsilon = n_rel**2 * eps0
    z0 = math.sqrt(mu0 / eps0)
    theta_t = np.arcsin(np.sin(theta) * N_INCIDENT / N_SUBSTRATE)
    k_iz = k0 * np.cos(theta)
    k_tz = n_rel * k0 * np.cos(theta_t)
    sigma_pp = sheet["sigma_pp"][None, :]
    sigma_ss = sheet["sigma_ss"][None, :]
    sigma_ps = sheet["sigma_ps"][None, :]
    sigma_sym = sheet["sigma_sym"][None, :]
    alpha_p_pos = (k_iz * epsilon + k_tz * eps0 + k_iz * k_tz * sigma_pp / omega) / eps0
    alpha_p_neg = (k_iz * epsilon - k_tz * eps0 + k_iz * k_tz * sigma_pp / omega) / eps0
    alpha_s_pos = k_iz + k_tz + mu0 * omega * sigma_ss
    alpha_s_neg = -k_iz + k_tz + mu0 * omega * sigma_ss
    beta = z0**2 * k_iz * k_tz * (sigma_ps**2 - sigma_sym**2)
    gamma_pos = 2.0 * z0 * k_iz * k_tz * (sigma_ps + sigma_sym)
    gamma_neg = 2.0 * z0 * k_iz * k_tz * (sigma_ps - sigma_sym)
    denominator = alpha_s_pos * alpha_p_pos + beta
    if np.any(np.abs(denominator) < 1e-300):
        raise ZeroDivisionError("Chen Fresnel denominator reached zero")
    return {
        "r_pp": (alpha_s_pos * alpha_p_neg + beta) / denominator,
        "r_ss": -(alpha_s_neg * alpha_p_pos + beta) / denominator,
        "r_ps": -gamma_neg / denominator,
        "r_sp": -gamma_pos / denominator,
    }


def fresnel_with_derivatives(
    theta_deg: np.ndarray, lambda_nm: float, sheet: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fresnel = chen_fresnel_matrix(theta_deg, lambda_nm, sheet)
    forward = chen_fresnel_matrix(np.asarray(theta_deg, dtype=float) + DX_DEG, lambda_nm, sheet)
    dx_rad = math.radians(DX_DEG)
    derivatives = {f"d{key}": (forward[key] - fresnel[key]) / dx_rad for key in ("r_pp", "r_ss", "r_ps", "r_sp")}
    return fresnel, derivatives


def beam_constants(lambda_nm: float) -> tuple[float, float]:
    k0 = 2.0 * math.pi / (lambda_nm * 1e-9)
    return k0, k0 * W0_M**2 / 2.0


def _complex_cross_y_components(
    theta_deg: np.ndarray,
    lambda_nm: float,
    fresnel: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
    gamma_i_deg: float,
) -> dict[str, np.ndarray]:
    """Strict y-layer extraction of PSHE_complex_cross.m for an explicit gamma_i."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))[:, None]
    cot = 1.0 / np.tan(theta)
    k0, z_rayleigh = beam_constants(lambda_nm)
    p, s, x, y = (fresnel[key] for key in ("r_pp", "r_ss", "r_ps", "r_sp"))
    dp, ds, dx, dy = (derivatives[key] for key in ("dr_pp", "dr_ss", "dr_ps", "dr_sp"))
    gamma_i = math.radians(gamma_i_deg)
    cg, sg = math.cos(gamma_i), math.sin(gamma_i)
    chi = (
        cg**2 * (2.0 * k0 * z_rayleigh * (np.abs(p) ** 2 + np.abs(y) ** 2) + np.abs(dp) ** 2 + np.abs(dy) ** 2)
        + sg**2 * (2.0 * k0 * z_rayleigh * (np.abs(x) ** 2 + np.abs(s) ** 2) + np.abs(dx) ** 2 + np.abs(ds) ** 2)
        + 2.0 * sg * cg * (2.0 * k0 * z_rayleigh * np.real(p * np.conj(x)) + np.real(dp * np.conj(dx)))
        + 2.0 * sg * cg * (2.0 * k0 * z_rayleigh * np.real(y * np.conj(s)) + np.real(dy * np.conj(ds)))
        + cot**2 * (
            np.abs(p) ** 2 + np.abs(s) ** 2 + np.abs(x) ** 2 + np.abs(y) ** 2
            + 2.0 * np.real(p * np.conj(s)) - 2.0 * np.real(x * np.conj(y))
        )
    )
    chi_sigma = (
        cg**2 * (4.0 * k0 * z_rayleigh * np.imag(p * np.conj(y)) + 2.0 * np.imag(dp * np.conj(dy)))
        + sg**2 * (4.0 * k0 * z_rayleigh * np.imag(x * np.conj(s)) + 2.0 * np.imag(dx * np.conj(ds)))
        + 2.0 * sg * cg * (2.0 * k0 * z_rayleigh * np.imag(p * np.conj(s)) + np.imag(dp * np.conj(ds)))
        + 2.0 * sg * cg * (2.0 * k0 * z_rayleigh * np.imag(x * np.conj(y)) + np.imag(dx * np.conj(dy)))
        + 2.0 * cot**2 * (
            np.imag(p * np.conj(y)) - np.imag(p * np.conj(x))
            + np.imag(s * np.conj(y)) - np.imag(s * np.conj(x))
        )
    )
    beta_delta = 2.0 * cot * (
        np.imag(x * np.conj(p)) + np.imag(s * np.conj(y))
        + 2.0 * cg**2 * np.imag(p * np.conj(y))
        + 2.0 * sg**2 * np.imag(x * np.conj(s))
        + 2.0 * cg * sg * (np.imag(x * np.conj(y)) + np.imag(p * np.conj(s)))
    )
    beta_sigma_delta = 2.0 * cot * (
        np.real(y * np.conj(x)) - np.real(p * np.conj(s))
        - cg**2 * (np.abs(y) ** 2 + np.abs(p) ** 2)
        - sg**2 * (np.abs(s) ** 2 + np.abs(x) ** 2)
        - 2.0 * cg * sg * (np.real(p * np.conj(x)) + np.real(y * np.conj(s)))
    )
    beta_theta = 2.0 * cot * (
        (sg**2 - cg**2) * np.real(p * np.conj(x))
        + (sg**2 - cg**2) * np.real(y * np.conj(s))
        + cg * sg * (np.abs(p) ** 2 - np.abs(s) ** 2 - np.abs(x) ** 2 + np.abs(y) ** 2)
    )
    beta_sigma_theta = 2.0 * cot * (
        (cg**2 - sg**2) * np.imag(p * np.conj(s))
        + (cg**2 - sg**2) * np.imag(x * np.conj(y))
        - 2.0 * cg * sg * (np.imag(p * np.conj(y)) - np.imag(x * np.conj(s)))
    )
    output: dict[str, np.ndarray] = {}
    for branch, helicity in (("plus", 1.0), ("minus", -1.0)):
        denominator = chi - helicity * chi_sigma
        if np.any(np.isclose(denominator, 0.0, rtol=0.0, atol=1e-300)):
            raise ZeroDivisionError(f"PSHE_complex_cross denominator reached zero for {branch}")
        delta = (beta_delta + helicity * beta_sigma_delta) / denominator * z_rayleigh
        theta_detector = (beta_theta + helicity * beta_sigma_theta) / denominator * Z_M
        output[f"delta_y_spatial_{branch}_m"] = delta
        output[f"Theta_y_detector_{branch}_m"] = theta_detector
        output[f"detector_y_{branch}_m"] = delta + theta_detector
    return output


def intrinsic_spatial_pshe(
    theta_deg: np.ndarray,
    lambda_nm: float,
    fresnel: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
    gamma_i_deg: float = DETECTOR_GAMMA_I_DEG,
) -> dict[str, np.ndarray]:
    components = _complex_cross_y_components(theta_deg, lambda_nm, fresnel, derivatives, gamma_i_deg)
    return {key: value for key, value in components.items() if key.startswith("delta_y_spatial_")}


def detector_plane_propagated_displacement(
    theta_deg: np.ndarray,
    lambda_nm: float,
    fresnel: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
    gamma_i_deg: float = DETECTOR_GAMMA_I_DEG,
) -> dict[str, np.ndarray]:
    components = _complex_cross_y_components(theta_deg, lambda_nm, fresnel, derivatives, gamma_i_deg)
    return {
        key: value
        for key, value in components.items()
        if key.startswith("Theta_y_detector_") or key.startswith("detector_y_")
    }


def circular_abc_total_plus(
    theta_deg: np.ndarray,
    lambda_nm: float,
    fresnel: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
) -> np.ndarray:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))[:, None]
    k0, z_rayleigh = beam_constants(lambda_nm)
    p, s, x, y = (fresnel[key] for key in ("r_pp", "r_ss", "r_ps", "r_sp"))
    dp, ds, dx, dy = (derivatives[key] for key in ("dr_pp", "dr_ss", "dr_ps", "dr_sp"))
    propagation = Z_M - 1j * z_rayleigh
    a = (-dp + 1j * dy) / propagation
    b = (-(x - y) + 1j * (p + s)) / np.tan(theta) / propagation
    c = p - 1j * y
    denominator = 2.0 * k0 * z_rayleigh / (Z_M**2 + z_rayleigh**2) * np.abs(c) ** 2 + np.abs(a) ** 2 + np.abs(b) ** 2
    return 2.0 * np.real(b * np.conj(c)) / denominator


def weak_readout(
    theta_deg: np.ndarray,
    lambda_nm: float,
    fresnel: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
    gamma_i_deg: float = GAMMA_I_PROVISIONAL_DEG,
    gamma_2_deg: float = POST_SELECTION_PROVISIONAL_DEG,
) -> dict[str, np.ndarray]:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))[:, None]
    gamma_i = math.radians(gamma_i_deg)
    gamma_2 = math.radians(gamma_2_deg)
    k0, z_rayleigh = beam_constants(lambda_nm)
    propagation = Z_M - 1j * z_rayleigh
    p, s, x, y = (fresnel[key] for key in ("r_pp", "r_ss", "r_ps", "r_sp"))
    dp, ds, dx, dy = (derivatives[key] for key in ("dr_pp", "dr_ss", "dr_ps", "dr_sp"))
    cos_i, sin_i = math.cos(gamma_i), math.sin(gamma_i)
    cos_2, sin_2 = math.cos(gamma_2), math.sin(gamma_2)
    a = -(cos_2 * (cos_i * dp + sin_i * dx) + sin_2 * (cos_i * dy + sin_i * ds)) / propagation
    b = -(
        math.cos(gamma_2 - gamma_i) * (x - y)
        + math.sin(gamma_2 - gamma_i) * (p + s)
    ) / np.tan(theta) / propagation
    c = cos_2 * (cos_i * p + sin_i * x) + sin_2 * (cos_i * y + sin_i * s)
    denominator = 2.0 * k0 * z_rayleigh / (Z_M**2 + z_rayleigh**2) * np.abs(c) ** 2 + np.abs(a) ** 2 + np.abs(b) ** 2
    return {
        "x_weak_m": 2.0 * np.real(a * np.conj(c)) / denominator,
        "y_weak_m": 2.0 * np.real(b * np.conj(c)) / denominator,
        "denominator_m_inv2": denominator,
    }


def compute_axis_state(axis: str, lambda_nm: float, theta_deg: np.ndarray) -> dict[str, Any]:
    axis_data = load_axis_sigma_s_cm(OPTIC_GRIDS_MAT, axis, lambda_nm)
    sheet = bulk_s_cm_to_sheet_s(axis_data["sigma_bulk_s_cm"])
    fresnel, derivatives = fresnel_with_derivatives(theta_deg, lambda_nm, sheet)
    intrinsic = intrinsic_spatial_pshe(theta_deg, lambda_nm, fresnel, derivatives)
    detector = detector_plane_propagated_displacement(theta_deg, lambda_nm, fresnel, derivatives)
    weak = weak_readout(theta_deg, lambda_nm, fresnel, derivatives)
    return {
        "axis_data": axis_data,
        "sheet": sheet,
        "fresnel": fresnel,
        "derivatives": derivatives,
        "intrinsic_spatial": intrinsic,
        "detector_plane": detector,
        "weak": weak,
    }


def run_v9_fresnel_gate() -> tuple[pd.DataFrame, float]:
    reference = pd.read_csv(V9_FRESNEL_REFERENCE)
    theta = reference["theta_i_deg"].to_numpy(float)
    state = compute_axis_state("a", LAMBDA_NM, theta)
    rows = []
    for element in ("r_pp", "r_ss", "r_ps", "r_sp"):
        calculated = state["fresnel"][element][:, 0]
        archived = reference[f"{element}_Chen_raw"].map(complex).to_numpy(complex)
        for angle, calc, ref in zip(theta, calculated, archived):
            rows.append(
                {
                    "theta_i_deg": angle,
                    "element": element,
                    "calculated_real": calc.real,
                    "calculated_imag": calc.imag,
                    "verified_v9_real": ref.real,
                    "verified_v9_imag": ref.imag,
                    "abs_difference": abs(calc - ref),
                }
            )
    check = pd.DataFrame(rows)
    maximum = float(check["abs_difference"].max())
    if maximum >= FRESNEL_TOL:
        raise RuntimeError(f"Fresnel v9 gate failed: {maximum:.12e} >= {FRESNEL_TOL:.1e}")
    return check, maximum


def classify_monotonic(values: np.ndarray, tolerance: float) -> tuple[str, str, bool]:
    differences = np.diff(np.asarray(values, dtype=float))
    signs = np.where(differences > tolerance, "+", np.where(differences < -tolerance, "-", "0"))
    nonzero = signs[signs != "0"]
    if len(nonzero) == 0:
        label = "flat"
    elif np.all(nonzero == "+"):
        label = "monotonic increase"
    elif np.all(nonzero == "-"):
        label = "monotonic decrease"
    else:
        label = "nonmonotonic"
    stable = bool(np.all(differences >= -tolerance) or np.all(differences <= tolerance))
    return label, "".join(signs.tolist()), stable


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    coefficient = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficient, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    return float(coefficient[0]), float(coefficient[1]), r2


def build_candidate_metrics(
    axis: str,
    lambda_nm: float,
    theta_deg: np.ndarray,
    strain_percent: np.ndarray,
    response: np.ndarray,
    response_quantity: str,
) -> pd.DataFrame:
    rows = []
    for index, angle in enumerate(theta_deg):
        values = np.asarray(response[index, :], dtype=float)
        tolerance = max(1e-15, 1e-6 * float(np.max(np.abs(values))))
        monotonicity, sign_sequence, stable = classify_monotonic(values, tolerance)
        local = np.diff(values) / np.diff(strain_percent)
        slope, intercept, r2 = linear_fit(strain_percent, values)
        endpoint = float(values[-1] - values[0])
        peak_to_peak = float(np.max(values) - np.min(values))
        endpoint_dominates = abs(endpoint) >= 0.85 * peak_to_peak if peak_to_peak > 0 else False
        practical_theta = 55.0 <= float(angle) <= 85.0
        practical_lambda = 400.0 <= float(lambda_nm) <= 1600.0
        recommended = bool(stable and monotonicity != "nonmonotonic" and endpoint_dominates and practical_theta and practical_lambda)
        sensitivity = endpoint / (strain_percent[-1] - strain_percent[0])
        # 继承旧 full_pshe_workflow：未通过数值推荐门控的候选仍保留有限分数，
        # 但施加 0.1 惩罚。recommended 标志与最终结论权限保持独立。
        gate_weight = 1.0 if recommended else 0.1
        score = (
            abs(sensitivity)
            * gate_weight
            * max(r2, 0.0)
            / (1.0 + abs(lambda_nm - LAMBDA_NM) / 1000.0)
        )
        rows.append(
            {
                "axis": axis,
                "lambda_nm": float(lambda_nm),
                "theta_i_deg": float(angle),
                "response_quantity": response_quantity,
                "sensitivity_per_percent": sensitivity,
                "endpoint_change": endpoint,
                "peak_to_peak_amplitude": peak_to_peak,
                "monotonicity": monotonicity,
                "finite_difference_sign_sequence": sign_sequence,
                "finite_difference_min": float(np.min(local)),
                "finite_difference_max": float(np.max(local)),
                "linear_slope": slope,
                "linear_intercept": intercept,
                "linear_r2": r2,
                "recommended_by_numeric_gate": recommended,
                "score": score,
                "gamma_i_deg": GAMMA_I_PROVISIONAL_DEG,
                "post_selection_gamma_2_deg": POST_SELECTION_PROVISIONAL_DEG,
                "parameter_status": PARAMETER_STATUS,
                "final_conclusion_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def long_fresnel_table(axis: str, state: dict[str, Any], theta_deg: np.ndarray) -> pd.DataFrame:
    strain = state["axis_data"]["strain_percent"]
    rows = []
    for ti, angle in enumerate(theta_deg):
        for si, epsilon in enumerate(strain):
            row: dict[str, Any] = {"axis": axis, "strain_percent": epsilon, "theta_i_deg": angle, "lambda_nm": LAMBDA_NM}
            for element in ("r_pp", "r_ss", "r_ps", "r_sp"):
                value = state["fresnel"][element][ti, si]
                row[f"{element}_real"] = value.real
                row[f"{element}_imag"] = value.imag
                row[f"abs_{element}"] = abs(value)
            rows.append(row)
    return pd.DataFrame(rows)


def long_pshe_table(axis: str, state: dict[str, Any], theta_deg: np.ndarray) -> pd.DataFrame:
    strain = state["axis_data"]["strain_percent"]
    rows = []
    for ti, angle in enumerate(theta_deg):
        for si, epsilon in enumerate(strain):
            spatial = float(state["intrinsic_spatial"]["delta_y_spatial_plus_m"][ti, si])
            angular = float(state["detector_plane"]["Theta_y_detector_plus_m"][ti, si])
            detector = float(state["detector_plane"]["detector_y_plus_m"][ti, si])
            rows.append(
                {
                    "axis": axis, "strain_percent": epsilon, "theta_i_deg": angle, "lambda_nm": LAMBDA_NM,
                    "branch": "plus", "main_pshe_quantity": "intrinsic_spatial_PSHE_displacement",
                    "delta_y_spatial_m": spatial, "delta_y_spatial_nm": spatial * 1e9,
                    "Theta_y_detector_m": angular, "Theta_y_detector_nm": angular * 1e9,
                    "detector_y_m": detector, "detector_y_nm": detector * 1e9, "detector_y_um": detector * 1e6,
                    "parameter_status": PARAMETER_STATUS,
                }
            )
    return pd.DataFrame(rows)


def long_readout_table(axis: str, state: dict[str, Any], theta_deg: np.ndarray) -> pd.DataFrame:
    strain = state["axis_data"]["strain_percent"]
    rows = []
    for ti, angle in enumerate(theta_deg):
        for si, epsilon in enumerate(strain):
            value = float(state["weak"]["y_weak_m"][ti, si])
            rows.append(
                {
                    "axis": axis, "strain_percent": epsilon, "theta_i_deg": angle, "lambda_nm": LAMBDA_NM,
                    "branch": "plus", "gamma_i_deg": GAMMA_I_PROVISIONAL_DEG,
                    "post_selection_gamma_2_deg": POST_SELECTION_PROVISIONAL_DEG,
                    "weak_readout_m": value, "weak_readout_nm": value * 1e9,
                    "weak_readout_um": value * 1e6, "weak_readout_mm": value * 1e3,
                    "upstream_layer": "Fresnel matrix and angular derivatives; no scalar PSHE displacement input",
                    "readout_definition": "post-selection readout; not intrinsic PSHE",
                    "parameter_status": PARAMETER_STATUS,
                }
            )
    return pd.DataFrame(rows)


def _zero_reference_index(state: dict[str, Any]) -> int:
    strain = np.asarray(state["axis_data"]["strain_percent"], dtype=float)
    matches = np.where(np.isclose(strain, 0.0, rtol=0.0, atol=1e-12))[0]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one 0% reference column, found {len(matches)}")
    return int(matches[0])


def _safe_abs_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        np.abs(np.asarray(numerator, dtype=float)),
        np.abs(np.asarray(denominator, dtype=float)),
        out=np.full_like(np.asarray(numerator, dtype=float), np.nan),
        where=np.abs(denominator) > np.finfo(float).tiny,
    )


def build_layered_0pct_table(state: dict[str, Any], theta_deg: np.ndarray) -> pd.DataFrame:
    index = _zero_reference_index(state)
    delta = np.asarray(state["intrinsic_spatial"]["delta_y_spatial_plus_m"][:, index], dtype=float)
    theta_detector = np.asarray(state["detector_plane"]["Theta_y_detector_plus_m"][:, index], dtype=float)
    detector = np.asarray(state["detector_plane"]["detector_y_plus_m"][:, index], dtype=float)
    y_weak = np.asarray(state["weak"]["y_weak_m"][:, index], dtype=float)
    frame = pd.DataFrame(
        {
            "theta_deg": np.asarray(theta_deg, dtype=float),
            "lambda_nm": LAMBDA_NM,
            "branch": "plus",
            "delta_y_spatial_m": delta,
            "delta_y_spatial_nm": delta * 1e9,
            "Theta_y_detector_m": theta_detector,
            "Theta_y_detector_nm": theta_detector * 1e9,
            "Theta_y_detector_um": theta_detector * 1e6,
            "detector_y_m": detector,
            "detector_y_nm": detector * 1e9,
            "detector_y_um": detector * 1e6,
            "y_weak_m": y_weak,
            "y_weak_nm": y_weak * 1e9,
            "y_weak_um": y_weak * 1e6,
            "y_weak_mm": y_weak * 1e3,
            "enhancement_detector_over_spatial": _safe_abs_ratio(detector, delta),
            "enhancement_weak_over_spatial": _safe_abs_ratio(y_weak, delta),
            "detector_gamma_i_deg": DETECTOR_GAMMA_I_DEG,
            "weak_gamma_i_deg": GAMMA_I_PROVISIONAL_DEG,
            "post_selection_angle_deg": POST_SELECTION_PROVISIONAL_DEG,
            "sigma_i_branch": "NOT_USED_IN_ACTIVE_WEAK_FORMULA",
            "parameter_status": SENSITIVITY_STATUS,
        }
    )
    return frame.loc[:, LAYERED_OUTPUT_COLUMNS]


def build_weak_parameter_sensitivity(
    state: dict[str, Any], theta_deg: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = _zero_reference_index(state)
    delta = np.asarray(state["intrinsic_spatial"]["delta_y_spatial_plus_m"][:, index], dtype=float)
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for gamma_i_deg in GAMMA_I_SENSITIVITY_DEG:
        for post_selection_deg in POST_SELECTION_SENSITIVITY_DEG:
            weak = weak_readout(
                theta_deg,
                LAMBDA_NM,
                state["fresnel"],
                state["derivatives"],
                gamma_i_deg=gamma_i_deg,
                gamma_2_deg=post_selection_deg,
            )["y_weak_m"][:, index]
            enhancement = _safe_abs_ratio(weak, delta)
            block = pd.DataFrame(
                {
                    "theta_deg": theta_deg,
                    "lambda_nm": LAMBDA_NM,
                    "gamma_i_deg": gamma_i_deg,
                    "post_selection_angle_deg": post_selection_deg,
                    "post_selection_interpretation": "provisional absolute gamma_2; NEED_USER_CONFIRM",
                    "sigma_i_branch": "NOT_USED_IN_ACTIVE_WEAK_FORMULA",
                    "delta_y_spatial_nm": delta * 1e9,
                    "y_weak_m": weak,
                    "y_weak_nm": weak * 1e9,
                    "y_weak_um": weak * 1e6,
                    "y_weak_mm": weak * 1e3,
                    "enhancement_weak_over_spatial": enhancement,
                    "parameter_status": SENSITIVITY_STATUS,
                }
            )
            rows.append(block)
            max_index = int(np.nanargmax(np.abs(weak)))
            summaries.append(
                {
                    "gamma_i_deg": gamma_i_deg,
                    "post_selection_angle_deg": post_selection_deg,
                    "sigma_i_branch": "NOT_USED_IN_ACTIVE_WEAK_FORMULA",
                    "theta_at_max_abs_readout_deg": float(theta_deg[max_index]),
                    "max_abs_y_weak_m": float(np.nanmax(np.abs(weak))),
                    "max_abs_y_weak_nm": float(np.nanmax(np.abs(weak)) * 1e9),
                    "max_abs_y_weak_um": float(np.nanmax(np.abs(weak)) * 1e6),
                    "max_abs_y_weak_mm": float(np.nanmax(np.abs(weak)) * 1e3),
                    "median_abs_y_weak_um": float(np.nanmedian(np.abs(weak)) * 1e6),
                    "max_enhancement_weak_over_spatial": float(np.nanmax(enhancement)),
                    "median_enhancement_weak_over_spatial": float(np.nanmedian(enhancement)),
                    "parameter_status": SENSITIVITY_STATUS,
                    "conclusion_allowed": False,
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def layered_parameter_audit_table(max_fresnel_residual: float) -> pd.DataFrame:
    _, z_rayleigh = beam_constants(LAMBDA_NM)
    rows = [
        ("optical_conductivity_unit_chain", "S/cm -> S/m -> sheet S", "CONFIRMED", "d applied exactly once"),
        ("film_thickness_m", THICKNESS_M, "CONFIRMED", "10 nm"),
        ("sigma_ps", "(s_xy+s_yx)/2", "CONFIRMED", "Chen script convention"),
        ("sigma_sym", "(s_xy-s_yx)/2", "CONFIRMED", "Chen script convention"),
        ("fresnel_v9_max_abs_residual", max_fresnel_residual, "PASS", "must be <1e-12"),
        ("main_PSHE_quantity", "intrinsic spatial PSHE displacement", "CONFIRMED", "delta_y spatial, formal main layer"),
        ("detector_quantity", "PSHE_complex_cross detector-plane propagated displacement", "DIAGNOSTIC", "delta_y+Theta_y; not intrinsic total"),
        ("weak_quantity", "post-selection centroid/readout", "PROVISIONAL", "computed directly from Fresnel derivatives and selection"),
        ("w0_m", W0_M, "NEED_USER_CONFIRM", "14.5 um default"),
        ("z_R_m", z_rayleigh, "DERIVED_FROM_UNCONFIRMED_W0", "k0*w0^2/2"),
        ("z_r_m", Z_M, "NEED_USER_CONFIRM", "0.1391 m default"),
        ("detector_gamma_i_deg", DETECTOR_GAMMA_I_DEG, "NEED_USER_CONFIRM", "strict PSHE_complex_cross value"),
        ("weak_gamma_i_deg", GAMMA_I_PROVISIONAL_DEG, "NEED_USER_CONFIRM", "default readout curve only"),
        ("post_selection_angle_deg", POST_SELECTION_PROVISIONAL_DEG, "NEED_USER_CONFIRM", "default readout curve only"),
        ("post_selection_scan_interpretation", "absolute gamma_2", "NEED_USER_CONFIRM", "may instead be offset from orthogonality"),
        ("sigma_i_branch", "NOT_USED_IN_ACTIVE_WEAK_FORMULA", "NEED_USER_CONFIRM", "no plus/minus sigma_i branch in active weak code"),
        ("old_25000nm_status", "excluded from formal intrinsic PSHE", "CONFIRMED", "gamma_i=0 simplified detector-plane term"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "status", "source_or_note"])


def parameter_audit_table() -> pd.DataFrame:
    _, z_rayleigh = beam_constants(LAMBDA_NM)
    rows = [
        ("optical_conductivity_unit_chain", "S/cm -> S/m -> sheet S", "CONFIRMED", "d applied exactly once"),
        ("film_thickness_m", THICKNESS_M, "CONFIRMED", "10 nm"),
        ("w0_m", W0_M, "DEFAULT_AUDIT_VALUE", "14.5 um from weak script"),
        ("z_R_m_at_632p8nm", z_rayleigh, "DERIVED", "k0*w0^2/2"),
        ("z_m", Z_M, "DEFAULT_AUDIT_VALUE", "0.1391 m from weak script"),
        ("gamma_i_deg", GAMMA_I_PROVISIONAL_DEG, "NEED_USER_CONFIRM", "provisional workflow value"),
        ("post_selection_gamma_2_deg", POST_SELECTION_PROVISIONAL_DEG, "NEED_USER_CONFIRM", "provisional near-orthogonal diagnostic value"),
        ("sigma_i_branch", "NOT_USED_IN_ACTIVE_WEAK_FORMULA", "NEED_USER_CONFIRM", "separate circular branch only"),
        ("main_PSHE_quantity", "intrinsic_spatial_PSHE_displacement", "CONFIRMED", "formal main PSHE layer"),
        ("weak_output_raw_unit", "m", "CONFIRMED", "also exported as nm um mm"),
        ("workflow_conclusion_status", PARAMETER_STATUS, "PROVISIONAL", "not a final screening conclusion"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "status", "source_or_note"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman", "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold", "axes.linewidth": 1.2,
            "axes.unicode_minus": False, "xtick.direction": "in", "ytick.direction": "in",
        }
    )


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"Incident angle $\theta_i$ (deg)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def plot_fresnel_four_elements(state_0pct: dict[str, Any], theta_deg: np.ndarray) -> None:
    setup_style()
    colors = {"r_pp": "#0072B2", "r_ss": "#D55E00", "r_ps": "#009E73", "r_sp": "#CC79A7"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for ax, element in zip(axes.ravel(), colors):
        ax.plot(theta_deg, np.abs(state_0pct["fresnel"][element][:, 0]), color=colors[element], lw=1.8, marker="o", ms=2.8, markevery=150)
        ax.set_title(fr"$|r_{{{element[-2:]}}}|$", fontsize=11)
        style_axis(ax, "Magnitude")
    fig.suptitle("0% reference Fresnel coefficients; full v9 theta range", fontsize=12)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.91, wspace=0.28, hspace=0.30)
    fig.savefig(FIGURES / "weak_fresnel_four_elements_632p8nm_ab.png", dpi=600, facecolor="white")
    plt.close(fig)


def plot_fresnel_overlay(check: pd.DataFrame) -> None:
    setup_style()
    colors = {"r_pp": "#0072B2", "r_ss": "#D55E00", "r_ps": "#009E73", "r_sp": "#CC79A7"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for ax, element in zip(axes.ravel(), colors):
        data = check.loc[check["element"] == element]
        calculated = np.hypot(data["calculated_real"], data["calculated_imag"])
        verified = np.hypot(data["verified_v9_real"], data["verified_v9_imag"])
        ax.plot(data["theta_i_deg"], calculated, color=colors[element], lw=1.8, ls="-", marker="o", ms=2.8, markevery=150, label="workflow")
        ax.plot(data["theta_i_deg"], verified, color="black", lw=1.2, ls="--", marker="^", ms=2.8, markevery=(75, 150), label="verified v9")
        ax.set_title(fr"$|r_{{{element[-2:]}}}|$", fontsize=11)
        style_axis(ax, "Magnitude")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fresnel verification overlay: identical theta and object", fontsize=12)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.91, wspace=0.28, hspace=0.30)
    fig.savefig(FIGURES / "weak_fresnel_verified_overlay_632p8nm.png", dpi=600, facecolor="white")
    plt.close(fig)


def plot_axis_strain_curves(table: pd.DataFrame, value_column: str, ylabel: str, title: str, output: Path) -> None:
    setup_style()
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=False)
    for ax, axis in zip(axes, ("a", "b")):
        axis_data = table.loc[table["axis"] == axis]
        for color, marker, epsilon in zip(colors, markers, sorted(axis_data["strain_percent"].unique())):
            data = axis_data.loc[np.isclose(axis_data["strain_percent"], epsilon)]
            ax.plot(data["theta_i_deg"], data[value_column], color=color, lw=1.5, marker=marker, ms=2.8, markevery=180, label=f"{epsilon:g}%")
        ax.axhline(0.0, color="0.65", lw=0.7)
        ax.set_title(f"{axis}-axis", fontsize=11)
        style_axis(ax, ylabel)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.88, wspace=0.27)
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)


def plot_best_theta_fit(metrics: pd.DataFrame, readout: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, axis, color in zip(axes, ("a", "b"), ("#0072B2", "#D55E00")):
        candidates = metrics.loc[(metrics["axis"] == axis) & np.isfinite(metrics["score"])].sort_values("score", ascending=False)
        best = candidates.iloc[0]
        data = readout.loc[(readout["axis"] == axis) & np.isclose(readout["theta_i_deg"], best["theta_i_deg"])]
        x = data["strain_percent"].to_numpy(float)
        y = data["weak_readout_um"].to_numpy(float)
        coefficient = np.polyfit(x, y, 1)
        ax.plot(x, y, color=color, marker="o", lw=1.6, label="provisional readout")
        ax.plot(x, np.polyval(coefficient, x), color="black", ls="--", lw=1.1, label=f"linear fit, R²={best['linear_r2']:.4f}")
        ax.set_title(f"{axis}-axis, theta={best['theta_i_deg']:.2f}°", fontsize=10)
        ax.set_xlabel("Strain (%)", fontsize=11)
        ax.set_ylabel("Weak readout (um)", fontsize=11)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed 632.8 nm provisional best-theta fit", fontsize=12)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.86, wspace=0.28)
    fig.savefig(FIGURES / "weak_fixed_632p8nm_best_theta_fit_ab.png", dpi=600, facecolor="white")
    plt.close(fig)


def plot_sensitivity_map(metrics: pd.DataFrame, axis: str) -> None:
    setup_style()
    data = metrics.loc[metrics["axis"] == axis]
    pivot = data.pivot(index="theta_i_deg", columns="lambda_nm", values="sensitivity_per_percent")
    values = pivot.to_numpy(float) * 1e6
    vmax = np.nanpercentile(np.abs(values), 99.0)
    fig, ax = plt.subplots(figsize=(6.69, 4.72))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(LAMBDA_NM, color="black", ls="--", lw=0.9)
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel(r"Incident angle $\theta_i$ (deg)", fontsize=11)
    ax.set_title(f"{axis}-axis provisional weak-readout sensitivity", fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Sensitivity (um/%)", fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.14, top=0.90)
    fig.savefig(FIGURES / f"weak_theta_lambda_sensitivity_map_{axis}_axis.png", dpi=600, facecolor="white")
    plt.close(fig)


def plot_best_comparison(best: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.72))
    label_names = {
        "fixed_632p8nm": "fixed 632.8 nm",
        "theta_lambda_global": "theta-lambda",
    }
    labels = [f"{row.axis}-axis\n{label_names[row.candidate_type]}" for row in best.itertuples()]
    values = np.abs(best["sensitivity_per_percent"].to_numpy(float)) * 1e6
    colors = ["#0072B2" if row.axis == "a" else "#D55E00" for row in best.itertuples()]
    markers = ["o" if row.candidate_type == "fixed_632p8nm" else "^" for row in best.itertuples()]
    for index, (value, color, marker) in enumerate(zip(values, colors, markers)):
        ax.scatter(index, value, s=55, color=color, marker=marker, zorder=3)
        ax.vlines(index, 0, value, color=color, lw=1.3, alpha=0.7)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Absolute weak-readout sensitivity (um/%)", fontsize=11)
    ax.set_title("Provisional candidates; parameters require confirmation", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.22, top=0.90)
    fig.savefig(FIGURES / "weak_best_candidates_comparison_ab.png", dpi=600, facecolor="white")
    plt.close(fig)


def _save_single_curve(
    x: np.ndarray,
    y: np.ndarray,
    output: Path,
    ylabel: str,
    title: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.3))
    ax.plot(x, y, color=color, lw=1.8, marker=marker, ms=3.3, markevery=150, label=label)
    ax.axhline(0.0, color="0.65", lw=0.7)
    style_axis(ax, ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.90)
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)


def _weak_display_column(table: pd.DataFrame) -> tuple[str, str, float]:
    maximum = float(np.nanmax(np.abs(table["y_weak_m"])))
    if maximum >= 1e-3:
        return "y_weak_mm", "Weak-measurement readout (mm)", maximum * 1e3
    if maximum >= 1e-6:
        return "y_weak_um", "Weak-measurement readout (um)", maximum * 1e6
    return "y_weak_nm", "Weak-measurement readout (nm)", maximum * 1e9


def plot_layered_0pct_outputs(layered: pd.DataFrame) -> None:
    theta = layered["theta_deg"].to_numpy(float)
    _save_single_curve(
        theta, layered["delta_y_spatial_nm"].to_numpy(float),
        FIGURES / "layered_0pct_intrinsic_spatial_pshe_632p8nm.png",
        "Intrinsic spatial PSHE displacement (nm)",
        "0% reference intrinsic spatial PSHE, plus branch",
        r"$\Delta_y^+$ (formal PSHE layer)", "#0072B2", "o",
    )

    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.3))
    ax.plot(theta, layered["detector_y_um"], color="#D55E00", lw=1.8, marker="o", ms=3.3, markevery=150, label=r"Detector $y^+=\Delta_y^++\Theta_y^+$")
    ax.plot(theta, layered["Theta_y_detector_um"], color="#009E73", lw=1.4, ls="--", marker="s", ms=3.0, markevery=(75, 150), label=r"Detector angular contribution $\Theta_y^+$")
    ax.axhline(0.0, color="0.65", lw=0.7)
    style_axis(ax, "Detector-plane displacement (um)")
    ax.set_title(r"0% detector-plane propagation diagnostic, $\gamma_i=45^\circ$", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.90)
    fig.savefig(FIGURES / "layered_0pct_detector_plane_displacement_632p8nm.png", dpi=600, facecolor="white")
    plt.close(fig)

    weak_column, weak_ylabel, _ = _weak_display_column(layered)
    _save_single_curve(
        theta, layered[weak_column].to_numpy(float),
        FIGURES / "layered_0pct_weak_readout_632p8nm.png",
        weak_ylabel,
        "0% provisional post-selection centroid/readout",
        rf"$\gamma_i={GAMMA_I_PROVISIONAL_DEG:g}^\circ$, $\gamma_2={POST_SELECTION_PROVISIONAL_DEG:g}^\circ$; NEED USER CONFIRM",
        "#CC79A7", "D",
    )

    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.3))
    detector_enhancement = layered["enhancement_detector_over_spatial"].replace([np.inf, -np.inf], np.nan)
    weak_enhancement = layered["enhancement_weak_over_spatial"].replace([np.inf, -np.inf], np.nan)
    ax.plot(theta, detector_enhancement, color="#D55E00", lw=1.7, marker="o", ms=3.1, markevery=150, label=r"$|y_{detector}|/|\Delta_y|$")
    ax.plot(theta, weak_enhancement, color="#CC79A7", lw=1.5, ls="--", marker="D", ms=3.0, markevery=(75, 150), label=r"$|y_{weak}|/|\Delta_y|$")
    ax.set_yscale("log")
    style_axis(ax, "Absolute enhancement factor")
    ax.set_title("0% layer-to-spatial enhancement factors", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.90)
    fig.savefig(FIGURES / "layered_0pct_enhancement_factor_632p8nm.png", dpi=600, facecolor="white")
    plt.close(fig)


def plot_parameter_sensitivity(summary: pd.DataFrame) -> None:
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.5))
    for color, marker, post in zip(colors, markers, POST_SELECTION_SENSITIVITY_DEG):
        data = summary.loc[np.isclose(summary["post_selection_angle_deg"], post)]
        ax.plot(data["gamma_i_deg"], data["max_abs_y_weak_mm"], color=color, marker=marker, lw=1.4, label=f"post={post:g} deg")
    ax.set_xlabel(r"Pre-selection $\gamma_i$ (deg)", fontsize=11)
    ax.set_ylabel("Maximum absolute weak readout (mm)", fontsize=11)
    ax.set_yscale(SENSITIVITY_Y_SCALE)
    ax.set_title("0% weak-readout sensitivity to pre-selection", fontsize=12)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.90)
    fig.savefig(FIGURES / "weak_readout_vs_gamma_0pct_632p8nm.png", dpi=600, facecolor="white")
    plt.close(fig)

    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.5))
    for color, marker, gamma in zip(colors, markers, GAMMA_I_SENSITIVITY_DEG):
        data = summary.loc[np.isclose(summary["gamma_i_deg"], gamma)]
        ax.plot(data["post_selection_angle_deg"], data["max_abs_y_weak_mm"], color=color, marker=marker, lw=1.4, label=f"gamma={gamma:g} deg")
    ax.set_xlabel("Post-selection angle (deg; provisional absolute gamma_2)", fontsize=10)
    ax.set_ylabel("Maximum absolute weak readout (mm)", fontsize=11)
    ax.set_xscale(POST_SELECTION_X_SCALE)
    ax.set_yscale(SENSITIVITY_Y_SCALE)
    ax.set_title("0% weak-readout sensitivity to post-selection", fontsize=12)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.90)
    fig.savefig(FIGURES / "weak_readout_vs_postselection_0pct_632p8nm.png", dpi=600, facecolor="white")
    plt.close(fig)

    pivot = summary.pivot(index="gamma_i_deg", columns="post_selection_angle_deg", values="max_enhancement_weak_over_spatial")
    log_values = np.log10(np.maximum(pivot.to_numpy(float), np.finfo(float).tiny))
    setup_style()
    fig, ax = plt.subplots(figsize=(6.69, 4.6))
    image = ax.imshow(log_values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    ax.set_xlabel("Post-selection angle (deg; provisional absolute gamma_2)", fontsize=10)
    ax.set_ylabel(r"Pre-selection $\gamma_i$ (deg)", fontsize=11)
    ax.set_title("0% maximum weak/spatial enhancement; parameter audit", fontsize=12)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"$\log_{10}\max(|y_{weak}|/|\Delta_y|)$", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.90, bottom=0.16, top=0.90)
    fig.savefig(FIGURES / "weak_enhancement_parameter_map_0pct_632p8nm.png", dpi=600, facecolor="white")
    plt.close(fig)


def write_layered_reports(
    max_residual: float,
    layered: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> None:
    max_spatial_nm = float(np.nanmax(np.abs(layered["delta_y_spatial_nm"])))
    max_detector_um = float(np.nanmax(np.abs(layered["detector_y_um"])))
    weak_column, weak_ylabel, max_weak_display = _weak_display_column(layered)
    max_sensitivity_mm = float(np.nanmax(sensitivity_summary["max_abs_y_weak_mm"]))
    min_sensitivity_mm = float(np.nanmin(sensitivity_summary["max_abs_y_weak_mm"]))
    main_report = f"""# 0% reference 分层 PSHE 与 weak readout 审计

## 结论边界

正式 PSHE 主层采用 intrinsic spatial PSHE displacement。`PSHE_complex_cross.m` 的 `delta+Theta` 统一命名为 detector-plane propagated displacement；weak 输出是独立 post-selection centroid/readout。旧 `gamma_i=0` 简化项产生的约 25,000 nm 不再称为 complete/total intrinsic PSHE。严格 `gamma_i=45°` 的 detector angular contribution 专项审计最大约 7839 nm。

## 输入与 Fresnel

输入链为 S/cm → S/m → sheet S，d=10 nm 且只乘一次。Chen 定义为 `sigma_ps=(s_xy+s_yx)/2`、`sigma_sym=(s_xy-s_yx)/2`。0% reference、632.8 nm、相同 theta 网格下，Fresnel v9 最大 residual 为 `{max_residual:.12e}`，门槛 `<1e-12`，状态 PASS。

## 三层级数量级

- intrinsic spatial PSHE 最大绝对值：`{max_spatial_nm:.12e} nm`；
- detector-plane propagated displacement 最大绝对值：`{max_detector_um:.12e} um`；
- 默认 provisional weak readout 最大绝对值：`{max_weak_display:.12e}`，图轴为 `{weak_ylabel}`，对应列 `{weak_column}`。

三者未放入同一线性 y 轴。detector 层显式依赖 `z_r`、`z_R`、`w0` 和 `gamma_i=45°`；weak 层直接使用 A/B/C post-selection 公式，不读取 detector 位移标量。

## 参数敏感性

扫描 `gamma_i={list(GAMMA_I_SENSITIVITY_DEG)}` deg 与 post-selection `{list(POST_SELECTION_SENSITIVITY_DEG)}` deg。42 个组合的 theta-resolved readout 全部标记 `NEED_USER_CONFIRM`。各组合最大绝对 readout 范围为 `{min_sensitivity_mm:.12e}`–`{max_sensitivity_mm:.12e} mm`。该范围只说明参数依赖显著，不构成最优参数或实验推荐。

## 图件用途

主文候选：intrinsic spatial PSHE 图；在参数确认后可加入 weak readout 图。detector-plane 图、enhancement 图与参数敏感性图适合补充材料或方法审计。进入 a/b strain 前必须确认 `w0`、`z_r`、实际 `gamma_i`、post-selection 的绝对角/正交偏置解释，以及是否另建 circular `sigma_i` 分支。
"""
    (REPORTS / "analysis_layered_pshe_weak_readout_0pct.md").write_text(main_report, encoding="utf-8")

    pending = """# 进入 a/b workflow 前需要确认的参数

| 参数 | 当前处理 | 必须确认的问题 |
|---|---|---|
| w0 | 14.5 um 默认值 | 是否为实验束腰 |
| z_r | 0.1391 m 默认值 | 是否为实际探测面距离 |
| detector gamma_i | 45 deg | 是否采用 PSHE_complex_cross 原脚本前选角 |
| weak gamma_i | 0–89 deg 敏感性扫描 | 实际前选角 |
| post-selection angle | 0.1–5 deg 作为绝对 gamma_2 扫描 | 是否应解释为相对正交偏置 |
| sigma_i | active weak 公式中不存在 | 是否另行建立 circular plus/minus 分支 |

所有相关表格状态均为 `NEED_USER_CONFIRM`；当前不得输出 best candidate 或进入 full a/b screening。
"""
    (REPORTS / "parameter_confirmation_needed.md").write_text(pending, encoding="utf-8")

    manuscript = f"""# 0% reference 分层 PSHE 与弱测量表述草案

在 632.8 nm 下，光电导率经 S/cm 到 S/m 转换后乘以 10 nm 厚度得到面电导率，并采用已验证的 Chen convention 构造 Fresnel matrix。该结果与 v9 参考的最大差异为 `{max_residual:.3e}`。界面处 intrinsic spatial PSHE displacement 被作为正式 PSHE 层，其最大绝对值为 `{max_spatial_nm:.3f} nm`。包含传播角贡献的 `delta+Theta` 被单独定义为 detector-plane propagated displacement，而不解释为材料本征 total PSHE。weak-measurement response 则由 Fresnel 系数、角导数、光束传播以及前/后选择参数直接计算，并作为独立 post-selection centroid/readout 报告。

当前 gamma_i、post-selection 策略、w0、z_r 及 circular branch 尚待实验参数确认。因此，0% 参数扫描仅用于展示模型敏感性和建立论文图框架，不用于声明最佳工作点或应变传感结论。
"""
    (MANUSCRIPT / "manuscript_text_layered_pshe_weak_readout_0pct.md").write_text(manuscript, encoding="utf-8")


def write_documentation(max_residual: float, best: pd.DataFrame, lambda_count: int) -> None:
    pending = "gamma_i、post-selection angle 策略、sigma_i 分支"
    best_lines = "\n".join(
        f"- {row.axis}-axis {row.candidate_type}: lambda={row.lambda_nm:.1f} nm, theta={row.theta_i_deg:.2f} deg, "
        f"sensitivity={row.sensitivity_per_percent * 1e6:.6g} um/%（provisional）"
        for row in best.itertuples()
    )
    report = f"""# a/b 轴 weak measurement workflow 分析报告

## 1. 计算链和输入单位

计算链固定为 optical conductivity → S/cm to S/m → sheet S → Chen-convention Fresnel matrix，并从该层分别计算 intrinsic spatial PSHE、detector-plane propagated displacement 和 post-selection readout。d=10 nm 且厚度只乘一次。

## 2. Fresnel v9 gate

0% reference、lambda=632.8 nm 和 v9 完全相同 theta 范围下，最大 Fresnel difference 为 `{max_residual:.12e}`，通过 `<1e-12` 门槛。Fresnel 图与旧图外观若不同，原因是 theta 范围或绘图对象不同，而不是 Fresnel 数值改变；overlay 图使用相同 theta 和相同 Chen raw 元素。

## 3. 主 PSHE 层

主 PSHE 层使用 intrinsic spatial PSHE displacement。`delta+Theta` 仅命名为 detector-plane propagated displacement，并显式依赖传播距离和 `gamma_i`。主图仅绘制 plus 分支。

## 4. weak readout

weak readout 是从 Fresnel 与角度导数层直接分支的独立 post-selection readout。它不读取 intrinsic spatial 或 detector-plane 位移标量。raw y_weak 单位为 m，并同时输出 nm、um、mm。

## 5. 参数状态

w0=14.5 um 和 z=0.1391 m 作为当前默认审计值。仍需人工确认：{pending}。本轮 provisional screening 使用 gamma_i=0 deg、gamma_2=89 deg；该选择不能写成最终实验方案。

## 6. fixed 与 theta-lambda provisional 结果

theta-lambda 候选波长数：{lambda_count}。所有 candidate 表均设置 `parameter_status={PARAMETER_STATUS}` 和 `final_conclusion_allowed=False`。

{best_lines}

## 7. 图表用途

- 可作为主文候选：intrinsic spatial PSHE plus-branch 图；
- 仅适合补充材料或内部诊断：detector-plane propagation、weak parameter sensitivity 和 theta-lambda maps；
- 暂不可作为论文主文结论图：weak best-theta、best-candidate comparison 和所有 provisional weak screening 图，直至参数确认并复算。

## 8. 结论边界

weak readout 统一称为 post-selection readout / weak-measurement response，不写成 intrinsic PSHE 被改变。当前输出是完整流程验证，不是最终实验参数推荐。
"""
    (REPORTS / "analysis_weak_full_workflow_ab.md").write_text(report, encoding="utf-8")

    convention = f"""# Weak workflow 公式与输入 convention 汇总

- 光电导率：bulk S/cm → bulk S/m → sheet S；d=10 nm，仅乘一次。
- Chen 输入：`sigma_ps=(s_xy+s_yx)/2`；`sigma_sym=(s_xy-s_yx)/2`。
- Fresnel 数值参考：归档 v9；本轮最大 residual `{max_residual:.12e}`。
- 主 PSHE：intrinsic spatial PSHE displacement；detector-plane propagation 单独诊断。
- weak output：raw m，同时输出 nm、um、mm。
- weak readout 是 post-selection response，不替代 intrinsic PSHE。
- NEED_USER_CONFIRM：{pending}。
"""
    (REPORTS / "weak_formula_and_input_convention_summary.md").write_text(convention, encoding="utf-8")

    manuscript = """# Weak measurement workflow manuscript text

## 中文表述

本工作流首先将单位为 S/cm 的体光电导率转换为 S/m，并使用 10 nm 厚度一次性得到面电导率。Chen convention Fresnel matrix 与归档 v9 参考对齐。主 PSHE 层采用 intrinsic spatial PSHE displacement；`delta+Theta` 单独定义为 detector-plane propagated displacement。weak-measurement response 由独立 post-selection 公式直接计算，原始输出单位为 m。

当前 gamma_i、post-selection angle 策略和 sigma_i 分支尚未完成实验参数确认，因此 weak screening 结果只能作为 provisional workflow validation，不能作为最终灵敏度或最佳工作点结论。

## English wording

The workflow converts bulk optical conductivity from S/cm to S/m and applies the 10 nm thickness exactly once. The principal PSHE layer is the intrinsic spatial PSHE displacement. The propagated `delta+Theta` quantity is reported separately as a detector-plane diagnostic, while the weak-measurement response is evaluated directly from the post-selection centroid formula with raw output in metres. Candidate screening remains provisional until all selection and beam parameters are confirmed.
"""
    (MANUSCRIPT / "manuscript_text_weak_full_workflow_ab.md").write_text(manuscript, encoding="utf-8")


def write_readme_and_description() -> None:
    readme = f"""# Weak measurement workflow

本目录实现分层 weak measurement workflow：optical conductivity → sheet conductivity → Chen Fresnel → {{intrinsic spatial PSHE；detector-plane propagation；post-selection readout}}。

运行方式：

```powershell
python scripts\\run_weak_measurement_workflow_ab.py --mode sanity_0pct
python scripts\\run_weak_measurement_workflow_ab.py --mode layered_0pct
python scripts\\run_weak_measurement_workflow_ab.py --mode full_ab_workflow
```

`layered_0pct` 只运行 0% reference 和参数敏感性。`full_ab_workflow` 必须等待 gamma_i、post-selection strategy、w0、z_r 和 sigma_i branch 被确认。
"""
    (WORKFLOW_ROOT / "README.md").write_text(readme, encoding="utf-8")
    files = [
        ("scripts/run_weak_measurement_workflow_ab.py", "Python", "完整 workflow 入口"),
        ("tables/weak_fresnel_632p8nm_ab.csv", "CSV", "a/b fixed-632.8 Fresnel"),
        ("tables/weak_fresnel_vs_verified_v9_check.csv", "CSV", "0% v9 gate"),
        ("tables/layered_0pct_pshe_readout_632p8nm.csv", "CSV", "0% 三层级结果"),
        ("tables/weak_readout_632p8nm_ab.csv", "CSV", "post-selection readout"),
        ("tables/weak_fixed_632p8nm_candidate_metrics_ab.csv", "CSV", "fixed provisional metrics"),
        ("tables/weak_theta_lambda_candidate_metrics_ab.csv", "CSV", "theta-lambda provisional metrics"),
        ("tables/weak_best_candidates_ab.csv", "CSV", "provisional best candidates"),
        ("reports/analysis_weak_full_workflow_ab.md", "Markdown", "分析与边界"),
    ]
    lines = ["# 文件说明", "", "| 文件 | 类型 | 用途 |", "|---|---|---|"]
    lines.extend(f"| `{path}` | {kind} | {purpose} |" for path, kind, purpose in files)
    (WORKFLOW_ROOT / "file_description.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(WORKFLOW_ROOT.rglob("*")):
        if not path.is_file() or "matplotlib_cache" in path.parts or "__pycache__" in path.parts:
            continue
        rows.append(
            {
                "file_path": str(path.relative_to(WORKFLOW_ROOT)).replace("\\", "/"),
                "action": "generated_or_retained",
                "source_path": "read-only optic_data; archived v9; audited MATLAB formulas",
                "generated_by": Path(__file__).name,
                "parameter_status": PARAMETER_STATUS if path.name.startswith("weak_") else "reference_or_documentation",
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(WORKFLOW_ROOT / "manifest.csv", index=False, encoding="utf-8-sig")


def run_sanity_mode() -> float:
    ensure_environment()
    check, maximum = run_v9_fresnel_gate()
    check.to_csv(TABLES / "weak_fresnel_vs_verified_v9_check.csv", index=False, encoding="utf-8-sig")
    parameter_audit_table().to_csv(TABLES / "weak_parameter_audit_ab.csv", index=False, encoding="utf-8-sig")
    plot_fresnel_overlay(check)
    print(f"Fresnel v9 maximum difference: {maximum:.12e}")
    print("sanity_0pct: PASS")
    return maximum


def run_layered_0pct_mode() -> dict[str, float]:
    """Run only the verified 0% layered calculation and provisional parameter audit."""
    ensure_environment()
    check, maximum = run_v9_fresnel_gate()
    if not maximum < FRESNEL_TOL:
        raise RuntimeError(f"Fresnel v9 gate failed: {maximum:.12e} >= {FRESNEL_TOL:.1e}")
    check.to_csv(TABLES / "layered_0pct_fresnel_v9_check.csv", index=False, encoding="utf-8-sig")

    state = compute_axis_state("a", LAMBDA_NM, FIXED_THETA_DEG)
    layered = build_layered_0pct_table(state, FIXED_THETA_DEG)
    sensitivity, sensitivity_summary = build_weak_parameter_sensitivity(state, FIXED_THETA_DEG)
    parameter_audit = layered_parameter_audit_table(maximum)

    strict_theta_max_nm = float(np.nanmax(np.abs(layered["Theta_y_detector_nm"])))
    if not np.isclose(strict_theta_max_nm, 7839.375458256209, rtol=2e-10, atol=2e-6):
        raise AssertionError(
            "Strict gamma_i=45 deg detector angular contribution no longer matches the source audit: "
            f"{strict_theta_max_nm:.12e} nm"
        )
    if not (sensitivity["parameter_status"] == SENSITIVITY_STATUS).all():
        raise AssertionError("Sensitivity rows must remain NEED_USER_CONFIRM")

    layered.to_csv(TABLES / "layered_0pct_pshe_readout_632p8nm.csv", index=False, encoding="utf-8-sig")
    parameter_audit.to_csv(TABLES / "layered_0pct_parameter_audit_632p8nm.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(TABLES / "weak_parameter_sensitivity_0pct_632p8nm.csv", index=False, encoding="utf-8-sig")
    sensitivity_summary.to_csv(TABLES / "weak_parameter_sensitivity_summary_0pct_632p8nm.csv", index=False, encoding="utf-8-sig")

    plot_layered_0pct_outputs(layered)
    plot_parameter_sensitivity(sensitivity_summary)
    write_layered_reports(maximum, layered, sensitivity_summary)

    result = {
        "fresnel_v9_residual": maximum,
        "max_abs_intrinsic_spatial_nm": float(np.nanmax(np.abs(layered["delta_y_spatial_nm"]))),
        "max_abs_detector_plane_um": float(np.nanmax(np.abs(layered["detector_y_um"]))),
        "max_abs_weak_readout_m": float(np.nanmax(np.abs(layered["y_weak_m"]))),
        "strict_gamma45_Theta_max_nm": strict_theta_max_nm,
    }
    for key, value in result.items():
        print(f"{key}={value:.12e}")
    print("parameter_status=NEED_USER_CONFIRM")
    print("full_ab_workflow_executed=NO")
    return result


def run_full_mode() -> None:
    maximum = run_sanity_mode()
    fixed_states = {axis: compute_axis_state(axis, LAMBDA_NM, FIXED_THETA_DEG) for axis in ("a", "b")}
    fresnel_table = pd.concat([long_fresnel_table(axis, state, FIXED_THETA_DEG) for axis, state in fixed_states.items()], ignore_index=True)
    pshe_table = pd.concat([long_pshe_table(axis, state, FIXED_THETA_DEG) for axis, state in fixed_states.items()], ignore_index=True)
    readout_table = pd.concat([long_readout_table(axis, state, FIXED_THETA_DEG) for axis, state in fixed_states.items()], ignore_index=True)
    fixed_metrics = pd.concat(
        [
            build_candidate_metrics(
                axis, LAMBDA_NM, FIXED_THETA_DEG,
                state["axis_data"]["strain_percent"], state["weak"]["y_weak_m"], "weak_post_selection_readout_m",
            )
            for axis, state in fixed_states.items()
        ],
        ignore_index=True,
    )
    fresnel_table.to_csv(TABLES / "weak_fresnel_632p8nm_ab.csv", index=False, encoding="utf-8-sig")
    pshe_table.to_csv(TABLES / "weak_pshe_total_632p8nm_ab.csv", index=False, encoding="utf-8-sig")
    readout_table.to_csv(TABLES / "weak_readout_632p8nm_ab.csv", index=False, encoding="utf-8-sig")
    fixed_metrics.to_csv(TABLES / "weak_fixed_632p8nm_candidate_metrics_ab.csv", index=False, encoding="utf-8-sig")

    raw = load_grids(str(OPTIC_GRIDS_MAT))
    lambda_grid = np.asarray(raw["lambda_nm"], dtype=float).ravel()
    lambda_values = BASE_LAMBDA_SCREEN_NM[
        (BASE_LAMBDA_SCREEN_NM >= np.min(lambda_grid)) & (BASE_LAMBDA_SCREEN_NM <= np.max(lambda_grid))
    ]
    lambda_values = np.unique(np.r_[lambda_values, LAMBDA_NM])
    screen_frames = []
    for axis in ("a", "b"):
        for wavelength in lambda_values:
            state = compute_axis_state(axis, float(wavelength), SCREEN_THETA_DEG)
            screen_frames.append(
                build_candidate_metrics(
                    axis, float(wavelength), SCREEN_THETA_DEG,
                    state["axis_data"]["strain_percent"], state["weak"]["y_weak_m"], "weak_post_selection_readout_m",
                )
            )
    screen_metrics = pd.concat(screen_frames, ignore_index=True)
    screen_metrics.to_csv(TABLES / "weak_theta_lambda_candidate_metrics_ab.csv", index=False, encoding="utf-8-sig")

    best_rows = []
    for axis in ("a", "b"):
        fixed_valid = fixed_metrics.loc[(fixed_metrics["axis"] == axis) & np.isfinite(fixed_metrics["score"])].sort_values("score", ascending=False)
        global_valid = screen_metrics.loc[(screen_metrics["axis"] == axis) & np.isfinite(screen_metrics["score"])].sort_values("score", ascending=False)
        if fixed_valid.empty or global_valid.empty:
            raise RuntimeError(f"No numerically valid provisional candidate for {axis}-axis")
        fixed_row = fixed_valid.iloc[0].copy()
        fixed_row["candidate_type"] = "fixed_632p8nm"
        global_row = global_valid.iloc[0].copy()
        global_row["candidate_type"] = "theta_lambda_global"
        best_rows.extend([fixed_row, global_row])
    best = pd.DataFrame(best_rows)
    best.to_csv(TABLES / "weak_best_candidates_ab.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {"item": "fresnel_v9_max_abs_difference", "value": maximum, "unit": "1", "status": "PASS"},
            {"item": "input_unit_chain", "value": "S/cm -> S/m -> sheet S", "unit": "", "status": "CONFIRMED"},
            {"item": "thickness", "value": THICKNESS_M, "unit": "m", "status": "CONFIRMED_ONCE"},
            {"item": "main_PSHE", "value": "intrinsic_spatial_PSHE_displacement", "unit": "m", "status": "CONFIRMED"},
            {"item": "weak_output", "value": "post-selection readout", "unit": "m", "status": PARAMETER_STATUS},
            {"item": "lambda_screen_count", "value": len(lambda_values), "unit": "count", "status": PARAMETER_STATUS},
            {"item": "batch_conclusion_allowed", "value": False, "unit": "boolean", "status": "NO"},
        ]
    )
    summary.to_csv(TABLES / "weak_workflow_summary_ab.csv", index=False, encoding="utf-8-sig")
    parameter_audit_table().to_csv(TABLES / "weak_parameter_audit_ab.csv", index=False, encoding="utf-8-sig")

    zero_state = compute_axis_state("a", LAMBDA_NM, FIXED_THETA_DEG)
    plot_fresnel_four_elements(zero_state, FIXED_THETA_DEG)
    plot_axis_strain_curves(
        pshe_table, "delta_y_spatial_nm", "Intrinsic spatial PSHE displacement (nm)",
        "Intrinsic spatial PSHE, plus branch, 632.8 nm", FIGURES / "weak_pshe_intrinsic_spatial_plus_branch_632p8nm_ab.png",
    )
    plot_axis_strain_curves(
        readout_table, "weak_readout_um", "Post-selection readout (um)",
        "Provisional weak-measurement response, plus branch", FIGURES / "weak_readout_plus_branch_632p8nm_ab.png",
    )
    plot_best_theta_fit(fixed_metrics, readout_table)
    plot_sensitivity_map(screen_metrics, "a")
    plot_sensitivity_map(screen_metrics, "b")
    plot_best_comparison(best)
    write_documentation(maximum, best, len(lambda_values))
    write_readme_and_description()
    write_manifest()
    print(f"full_ab_workflow rows: fixed={len(fixed_metrics)}, theta_lambda={len(screen_metrics)}")
    print(f"parameter status: {PARAMETER_STATUS}")
    print("final conclusions allowed: NO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="a/b weak measurement workflow")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sanity_0pct":
        run_sanity_mode()
        write_readme_and_description()
        write_manifest()
    elif args.mode == "layered_0pct":
        run_layered_0pct_mode()
    else:
        run_full_mode()


if __name__ == "__main__":
    main()
