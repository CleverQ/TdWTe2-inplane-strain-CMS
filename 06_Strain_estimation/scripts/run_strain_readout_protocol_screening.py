from __future__ import annotations

import math
import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / "logs" / "matplotlib_cache")))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WORKFLOW_ROOT.parents[1]
TABLES = WORKFLOW_ROOT / "tables"
FIGURES = WORKFLOW_ROOT / "figures"
REPORTS = WORKFLOW_ROOT / "reports"
MANUSCRIPT = WORKFLOW_ROOT / "manuscript_text"
LOGS = WORKFLOW_ROOT / "logs"

REQUIRED_OUTPUTS = (
    "strain_readout_amplitude_candidates.csv",
    "strain_readout_feature_angle_candidates.csv",
    "strain_readout_ratio_differential_candidates.csv",
    "strain_readout_multipoint_fitting_candidates.csv",
    "strain_readout_best_protocols_summary.csv",
    "strain_readout_632p8nm_experiment_compatible_summary.csv",
    "strain_readout_amplitude_fit_examples.png",
    "strain_readout_feature_angle_fit_examples.png",
    "strain_readout_ratio_differential_fit_examples.png",
    "strain_readout_multipoint_prediction_examples.png",
    "strain_readout_best_protocol_comparison.png",
    "strain_readout_theta_lambda_gamma2_screening_map.png",
    "strain_readout_632p8nm_experiment_compatible_protocol.png",
    "analysis_weak_strain_readout_protocol_screening.md",
    "manuscript_text_weak_strain_readout_protocol_screening.md",
)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coefficient = np.polyfit(x, y, 1)
    predicted = np.polyval(coefficient, x)
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > np.finfo(float).tiny else 0.0
    return float(coefficient[0]), float(coefficient[1]), predicted, float(r2)


def linear_metrics(strain_percent: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    strain = np.asarray(strain_percent, dtype=float)
    response = np.asarray(values, dtype=float)
    if strain.ndim != 1 or response.ndim != 1 or len(strain) != len(response) or len(strain) < 3:
        raise ValueError("strain_percent and values must be equal-length one-dimensional arrays with >=3 points")
    if not np.isfinite(strain).all() or not np.isfinite(response).all():
        raise ValueError("linear_metrics requires finite inputs")
    slope, intercept, predicted, r2 = _linear_fit(strain, response)
    residual = response - predicted
    differences = np.diff(response)
    tolerance = max(np.finfo(float).eps, 1e-8 * float(np.max(np.abs(response))))
    nonzero = differences[np.abs(differences) > tolerance]
    if len(nonzero) == 0:
        monotonicity = 0.0
        sign_consistency = 0.0
        monotonicity_label = "flat"
    else:
        positive = float(np.mean(nonzero > 0.0))
        negative = float(np.mean(nonzero < 0.0))
        monotonicity = max(positive, negative)
        expected_sign = 1.0 if slope >= 0.0 else -1.0
        sign_consistency = float(np.mean(np.sign(nonzero) == expected_sign))
        monotonicity_label = "increase" if positive == 1.0 else "decrease" if negative == 1.0 else "mixed"
    dynamic_range = float(np.max(response) - np.min(response))
    sensitivity = abs(slope)
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    loocv = np.full_like(response, np.nan)
    for held_out in range(len(response)):
        mask = np.arange(len(response)) != held_out
        held_slope, held_intercept, _, _ = _linear_fit(strain[mask], response[mask])
        loocv[held_out] = held_slope * strain[held_out] + held_intercept
    loocv_rmse = float(np.sqrt(np.mean((loocv - response) ** 2)))
    formula = f"epsilon_percent = (readout - ({intercept:.16e})) / ({slope:.16e})" if abs(slope) > 0.0 else "undefined"
    return {
        "slope": slope,
        "intercept": intercept,
        "sensitivity_abs_per_percent": sensitivity,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "loocv_rmse_readout": loocv_rmse,
        "dynamic_range": dynamic_range,
        "monotonicity": monotonicity,
        "monotonicity_label": monotonicity_label,
        "sign_consistency": sign_consistency,
        "inversion_formula": formula,
    }


def denominator_gate(denominator: np.ndarray, threshold: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    values = np.abs(np.asarray(denominator, dtype=float))
    if values.ndim == 1:
        values = values[:, None]
    finite = np.isfinite(values)
    reference = float(np.nanpercentile(values[finite], 95.0)) if np.any(finite) else 0.0
    if not np.isfinite(reference) or reference <= np.finfo(float).tiny:
        relative = np.zeros(values.shape[0], dtype=float)
        return relative, np.zeros(values.shape[0], dtype=bool)
    row_minimum = np.min(np.where(finite, values, 0.0), axis=1)
    relative = row_minimum / reference
    accepted = np.all(finite, axis=1) & (relative >= float(threshold))
    return relative, accepted


def safe_enhancement(weak_readout: np.ndarray, intrinsic_spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weak = np.asarray(weak_readout, dtype=float)
    intrinsic = np.asarray(intrinsic_spatial, dtype=float)
    scale = float(np.nanmax(np.abs(intrinsic))) if intrinsic.size else 0.0
    threshold = max(1e-15, 1e-3 * scale)
    valid = np.isfinite(weak) & np.isfinite(intrinsic) & (np.abs(intrinsic) >= threshold)
    enhancement = np.full(np.broadcast_shapes(weak.shape, intrinsic.shape), np.nan, dtype=float)
    np.divide(np.abs(weak), np.abs(intrinsic), out=enhancement, where=valid)
    return enhancement, valid


def gamma_robustness(
    nominal: np.ndarray,
    shifted: Iterable[np.ndarray],
    dynamic_range: float | None = None,
) -> dict[str, float]:
    reference = np.asarray(nominal, dtype=float)
    variations = [np.asarray(item, dtype=float) for item in shifted]
    if not variations:
        return {"gamma2_error_normalized_rmse": 0.0, "robustness_score": 1.0}
    scale = float(dynamic_range) if dynamic_range is not None else float(np.ptp(reference))
    scale = max(abs(scale), 1e-30)
    normalized = [float(np.sqrt(np.mean((item - reference) ** 2))) / scale for item in variations]
    error = max(normalized)
    return {"gamma2_error_normalized_rmse": error, "robustness_score": 1.0 / (1.0 + error)}


def ratio_differential_series(y1: np.ndarray, y2: np.ndarray, floor_fraction: float = 0.05) -> dict[str, Any]:
    first = np.asarray(y1, dtype=float)
    second = np.asarray(y2, dtype=float)
    differential = first - second
    denominator = first + second
    scale = max(float(np.nanmax(np.abs(first)) + np.nanmax(np.abs(second))), 1e-30)
    ratio_valid = bool(np.isfinite(denominator).all() and np.min(np.abs(denominator)) >= floor_fraction * scale)
    ratio = differential / denominator if ratio_valid else np.full_like(differential, np.nan)
    return {
        "differential": differential,
        "ratio": ratio,
        "ratio_valid": ratio_valid,
        "minimum_abs_ratio_denominator": float(np.nanmin(np.abs(denominator))),
        "ratio_denominator_scale": scale,
    }


def _crossings(theta: np.ndarray, values: np.ndarray, target: float = 0.0) -> list[float]:
    shifted = np.asarray(values, dtype=float) - float(target)
    coordinates = np.asarray(theta, dtype=float)
    result: list[float] = []
    for index in range(len(coordinates) - 1):
        y0, y1 = shifted[index], shifted[index + 1]
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0.0:
            result.append(float(coordinates[index]))
        elif y0 * y1 < 0.0:
            fraction = -y0 / (y1 - y0)
            result.append(float(coordinates[index] + fraction * (coordinates[index + 1] - coordinates[index])))
    return result


def extract_feature_angles(theta_deg: np.ndarray, curves: np.ndarray) -> dict[str, np.ndarray]:
    theta = np.asarray(theta_deg, dtype=float)
    response = np.asarray(curves, dtype=float)
    if response.ndim == 1:
        response = response[:, None]
    if response.shape[0] != len(theta):
        raise ValueError("curves must have theta along axis 0")
    features = {name: np.full(response.shape[1], np.nan) for name in (
        "zero_crossing_deg", "extremum_angle_deg", "max_slope_angle_deg", "threshold_angle_deg"
    )}
    zero_reference: float | None = None
    threshold_reference: float | None = None
    threshold_anchor: float | None = None
    for column in range(response.shape[1]):
        values = response[:, column]
        derivative = np.gradient(values, theta)
        max_slope_index = int(np.nanargmax(np.abs(derivative)))
        extremum_index = int(np.nanargmax(np.abs(values)))
        features["max_slope_angle_deg"][column] = theta[max_slope_index]
        features["extremum_angle_deg"][column] = theta[extremum_index]
        zero_candidates = _crossings(theta, values, 0.0)
        if zero_candidates:
            if zero_reference is None:
                slopes = [abs(float(np.interp(item, theta, derivative))) for item in zero_candidates]
                zero_reference = zero_candidates[int(np.argmax(slopes))]
            selected = min(zero_candidates, key=lambda item: abs(item - float(zero_reference)))
            features["zero_crossing_deg"][column] = selected
        if threshold_reference is None:
            threshold_reference = 0.5 * float(values[extremum_index])
            threshold_anchor = float(theta[max_slope_index])
        threshold_candidates = _crossings(theta, values, float(threshold_reference))
        if threshold_candidates:
            features["threshold_angle_deg"][column] = min(
                threshold_candidates, key=lambda item: abs(item - float(threshold_anchor))
            )
    return features


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1e-15, scale, 1.0)
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    design = np.column_stack([np.ones(len(train)), train])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return np.column_stack([np.ones(len(test)), test]) @ coefficients


def ridge_loocv(features: np.ndarray, strain_percent: np.ndarray, alpha: float = 1e-3) -> dict[str, Any]:
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(strain_percent, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(target) or len(target) < 4:
        raise ValueError("features must be n-by-p and match at least four strain points")
    predictions = np.full_like(target, np.nan)
    for held_out in range(len(target)):
        mask = np.arange(len(target)) != held_out
        predictions[held_out] = _ridge_fit_predict(
            matrix[mask], target[mask], matrix[[held_out]], alpha=float(alpha)
        )[0]
    residual = predictions - target
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > np.finfo(float).tiny else 0.0
    return {
        "predicted_strain_percent": predictions,
        "loocv_rmse_percent": float(np.sqrt(np.mean(residual**2))),
        "loocv_mae_percent": float(np.mean(np.abs(residual))),
        "loocv_max_abs_error_percent": float(np.max(np.abs(residual))),
        "prediction_r2": float(r2),
    }


def _candidate_score(metrics: dict[str, Any], denominator_relative: float, robustness: float = 1.0) -> float:
    sensitivity = max(float(metrics.get("sensitivity_abs_per_percent", 0.0)), 0.0)
    r2 = max(min(float(metrics.get("r2", 0.0)), 1.0), 0.0)
    monotonicity = max(min(float(metrics.get("monotonicity", 0.0)), 1.0), 0.0)
    sign_consistency = max(min(float(metrics.get("sign_consistency", 0.0)), 1.0), 0.0)
    denominator_factor = min(max(math.log10(max(denominator_relative, 1e-12)) + 3.0, 0.0) / 3.0, 1.0)
    return sensitivity * r2 * monotonicity * sign_consistency * robustness * denominator_factor


def _common_linear_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "slope": metrics["slope"],
        "intercept": metrics["intercept"],
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "loocv_rmse_readout": metrics["loocv_rmse_readout"],
        "dynamic_range": metrics["dynamic_range"],
        "monotonicity": metrics["monotonicity"],
        "monotonicity_label": metrics["monotonicity_label"],
        "sign_consistency": metrics["sign_consistency"],
        "inversion_formula": metrics["inversion_formula"],
    }


def build_amplitude_candidates(
    axis: str,
    lambda_nm: float,
    theta_deg: np.ndarray,
    strain_percent: np.ndarray,
    gamma_2_deg: np.ndarray,
    weak_cube_m: np.ndarray,
    denominator_cube: np.ndarray,
    intrinsic_m: np.ndarray,
    top_per_gamma: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strain_array = np.asarray(strain_percent, dtype=float)
    centered = strain_array - np.mean(strain_array)
    slope_denominator = float(np.sum(centered**2))
    for gamma_index, gamma_value in enumerate(np.asarray(gamma_2_deg, dtype=float)):
        relative, accepted = denominator_gate(denominator_cube[gamma_index], threshold=1e-3)
        gamma_rows: list[dict[str, Any]] = []
        slopes = np.abs(np.sum(weak_cube_m[gamma_index] * centered[None, :], axis=1) / slope_denominator)
        ranked_indices = np.argsort(np.where(accepted, slopes, -np.inf))[::-1]
        selected_indices = ranked_indices[accepted[ranked_indices]]
        for theta_index in selected_indices[: max(int(top_per_gamma) * 3, int(top_per_gamma))]:
            theta_value = np.asarray(theta_deg, dtype=float)[theta_index]
            series = weak_cube_m[gamma_index, theta_index]
            metrics = linear_metrics(strain_percent, series)
            enhancement, valid = safe_enhancement(series, intrinsic_m[theta_index])
            row = {
                "protocol_id": f"A_{axis}_{lambda_nm:.3f}_{theta_value:.3f}_{gamma_value:.3f}",
                "protocol": "amplitude",
                "axis": axis,
                "lambda_nm": float(lambda_nm),
                "theta_i_deg": float(theta_value),
                "gamma_2_deg": float(gamma_value),
                "pre_selection_gamma_i_deg": 0.0,
                "sensitivity_abs_m_per_percent": metrics["sensitivity_abs_per_percent"],
                "weak_denominator_relative_min": float(relative[theta_index]),
                "enhancement_factor_median_valid": float(np.nanmedian(enhancement)) if np.any(valid) else np.nan,
                "enhancement_valid_fraction": float(np.mean(valid)),
                "gamma2_error_normalized_rmse": np.nan,
                "robustness_to_gamma2_error": np.nan,
                **_common_linear_fields(metrics),
            }
            row["screening_score"] = _candidate_score(metrics, row["weak_denominator_relative_min"])
            gamma_rows.append(row)
        gamma_rows.sort(key=lambda item: item["screening_score"], reverse=True)
        rows.extend(gamma_rows[: max(int(top_per_gamma), 1)])
    return pd.DataFrame(rows)


def build_feature_angle_candidates(
    axis: str,
    lambda_nm: float,
    theta_deg: np.ndarray,
    strain_percent: np.ndarray,
    gamma_2_deg: np.ndarray,
    weak_cube_m: np.ndarray,
    denominator_cube: np.ndarray,
) -> pd.DataFrame:
    feature_names = {
        "zero_crossing_deg": "zero_crossing",
        "extremum_angle_deg": "extremum",
        "max_slope_angle_deg": "max_slope",
        "threshold_angle_deg": "threshold",
    }
    rows: list[dict[str, Any]] = []
    for gamma_index, gamma_value in enumerate(np.asarray(gamma_2_deg, dtype=float)):
        relative, accepted = denominator_gate(denominator_cube[gamma_index], threshold=1e-3)
        features = extract_feature_angles(theta_deg, weak_cube_m[gamma_index])
        for source_name, feature_type in feature_names.items():
            values = features[source_name]
            finite = np.isfinite(values)
            if np.count_nonzero(finite) < 3:
                metrics = {
                    "slope": np.nan, "intercept": np.nan, "r2": 0.0, "rmse": np.nan,
                    "loocv_rmse_readout": np.nan, "dynamic_range": np.nan, "monotonicity": 0.0,
                    "monotonicity_label": "insufficient", "sign_consistency": 0.0,
                    "inversion_formula": "undefined", "sensitivity_abs_per_percent": 0.0,
                }
            else:
                metrics = linear_metrics(np.asarray(strain_percent)[finite], values[finite])
            denominator_relative = float(np.nanmin(relative[accepted])) if np.any(accepted) else 0.0
            row = {
                "protocol_id": f"B_{feature_type}_{axis}_{lambda_nm:.3f}_{gamma_value:.3f}",
                "protocol": "feature_angle",
                "feature_type": feature_type,
                "axis": axis,
                "lambda_nm": float(lambda_nm),
                "gamma_2_deg": float(gamma_value),
                "pre_selection_gamma_i_deg": 0.0,
                "feature_angles_deg": ";".join(f"{item:.8g}" for item in values),
                "finite_feature_fraction": float(np.mean(finite)),
                "sensitivity_abs_deg_per_percent": metrics["sensitivity_abs_per_percent"],
                "weak_denominator_relative_min": denominator_relative,
                **_common_linear_fields(metrics),
            }
            row["screening_score"] = _candidate_score(metrics, denominator_relative)
            rows.append(row)
    return pd.DataFrame(rows)


def _top_theta_indices(
    strain_percent: np.ndarray,
    values_by_theta: np.ndarray,
    accepted: np.ndarray,
    count: int,
) -> np.ndarray:
    strain = np.asarray(strain_percent, dtype=float)
    centered = strain - np.mean(strain)
    slopes = np.abs(np.sum(np.asarray(values_by_theta) * centered[None, :], axis=1) / np.sum(centered**2))
    ranked = np.argsort(np.where(accepted, slopes, -np.inf))[::-1]
    ranked = ranked[accepted[ranked]]
    return np.asarray(ranked[: max(int(count), 2)], dtype=int)


def build_ratio_differential_candidates(
    axis: str,
    lambda_nm: float,
    theta_deg: np.ndarray,
    strain_percent: np.ndarray,
    gamma_2_deg: np.ndarray,
    weak_cube_m: np.ndarray,
    denominator_cube: np.ndarray,
    top_theta_count: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    theta_values = np.asarray(theta_deg, dtype=float)
    for gamma_index, gamma_value in enumerate(np.asarray(gamma_2_deg, dtype=float)):
        relative, accepted = denominator_gate(denominator_cube[gamma_index], threshold=1e-3)
        selected = _top_theta_indices(strain_percent, weak_cube_m[gamma_index], accepted, top_theta_count)
        for first_position in range(len(selected)):
            for second_position in range(first_position + 1, len(selected)):
                first_index, second_index = int(selected[first_position]), int(selected[second_position])
                combined = ratio_differential_series(
                    weak_cube_m[gamma_index, first_index], weak_cube_m[gamma_index, second_index]
                )
                for readout_type, series in (("D", combined["differential"]), ("R", combined["ratio"])):
                    if readout_type == "R" and not combined["ratio_valid"]:
                        continue
                    metrics = linear_metrics(strain_percent, series)
                    denominator_relative = float(min(relative[first_index], relative[second_index]))
                    row = {
                        "protocol_id": f"C_{readout_type}_theta_{axis}_{lambda_nm:.3f}_{gamma_value:.3f}_{theta_values[first_index]:.3f}_{theta_values[second_index]:.3f}",
                        "protocol": "ratio_differential",
                        "readout_type": readout_type,
                        "pair_dimension": "theta",
                        "axis": axis,
                        "lambda_nm": float(lambda_nm),
                        "gamma_2_deg": float(gamma_value),
                        "coordinate_1": float(theta_values[first_index]),
                        "coordinate_2": float(theta_values[second_index]),
                        "theta_1_deg": float(theta_values[first_index]),
                        "theta_2_deg": float(theta_values[second_index]),
                        "weak_denominator_relative_min": denominator_relative,
                        "minimum_abs_ratio_denominator": combined["minimum_abs_ratio_denominator"],
                        "ratio_valid": combined["ratio_valid"],
                        "sensitivity_abs_per_percent": metrics["sensitivity_abs_per_percent"],
                        **_common_linear_fields(metrics),
                    }
                    row["screening_score"] = _candidate_score(metrics, denominator_relative)
                    rows.append(row)
    return pd.DataFrame(rows)


def build_multipoint_candidates(
    axis: str,
    lambda_nm: float,
    theta_deg: np.ndarray,
    strain_percent: np.ndarray,
    gamma_2_deg: np.ndarray,
    weak_cube_m: np.ndarray,
    denominator_cube: np.ndarray,
    point_count: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    theta_values = np.asarray(theta_deg, dtype=float)
    for gamma_index, gamma_value in enumerate(np.asarray(gamma_2_deg, dtype=float)):
        relative, accepted = denominator_gate(denominator_cube[gamma_index], threshold=1e-3)
        selected = _top_theta_indices(strain_percent, weak_cube_m[gamma_index], accepted, point_count)
        if len(selected) < 2:
            continue
        matrix = weak_cube_m[gamma_index, selected, :].T
        result = ridge_loocv(matrix, strain_percent, alpha=1e-3)
        denominator_relative = float(np.min(relative[selected]))
        stability = max(result["prediction_r2"], 0.0) / (1.0 + result["loocv_rmse_percent"])
        rows.append({
            "protocol_id": f"D_theta_{axis}_{lambda_nm:.3f}_{gamma_value:.3f}",
            "protocol": "multipoint_fitting",
            "fingerprint_dimension": "theta",
            "axis": axis,
            "lambda_nm": float(lambda_nm),
            "gamma_2_deg": float(gamma_value),
            "feature_coordinates": ";".join(f"{theta_values[index]:.8g}" for index in selected),
            "point_count": int(len(selected)),
            "weak_denominator_relative_min": denominator_relative,
            "loocv_rmse_percent": result["loocv_rmse_percent"],
            "loocv_mae_percent": result["loocv_mae_percent"],
            "loocv_max_abs_error_percent": result["loocv_max_abs_error_percent"],
            "prediction_r2": result["prediction_r2"],
            "predicted_strain_percent": ";".join(f"{item:.8g}" for item in result["predicted_strain_percent"]),
            "inversion_formula": "ridge regression on the listed Y_weak(theta) fingerprint",
            "screening_score": stability * min(max(denominator_relative / 1e-3, 0.0), 1.0),
        })
    return pd.DataFrame(rows)


THETA_GRID_DEG = np.round(np.arange(55.0, 85.0 + 1e-12, 0.1), 10)
GAMMA2_GRID_DEG = np.round(np.arange(60.0, 89.5 + 1e-12, 0.5), 10)
LAMBDA_GRID_NM = np.unique(np.r_[np.arange(400.0, 780.0 + 1e-12, 10.0), 632.8])
GAMMA2_ERROR_DEG = (-0.5, -0.1, 0.1, 0.5)
GRIDS_PATH = Path(__file__).resolve().parents[2] / "04_Optical_conductivity" / "data" / "processed" / "grids.mat"


def _load_base_workflow():
    path = Path(__file__).with_name("run_weak_measurement_workflow_ab.py")
    spec = importlib.util.spec_from_file_location("weak_base_workflow", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base workflow: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OPTIC_GRIDS_MAT = GRIDS_PATH
    return module


def _compute_block(workflow: Any, axis: str, lambda_nm: float, include_shifts: bool = True) -> dict[str, Any]:
    data = workflow.load_axis_sigma_s_cm(GRIDS_PATH, axis, float(lambda_nm))
    sheet = workflow.bulk_s_cm_to_sheet_s(data["sigma_bulk_s_cm"])
    fresnel, derivatives = workflow.fresnel_with_derivatives(THETA_GRID_DEG, float(lambda_nm), sheet)
    intrinsic = workflow.intrinsic_spatial_pshe(
        THETA_GRID_DEG, float(lambda_nm), fresnel, derivatives, gamma_i_deg=0.0
    )["delta_y_spatial_plus_m"]
    detector = workflow.detector_plane_propagated_displacement(
        THETA_GRID_DEG, float(lambda_nm), fresnel, derivatives, gamma_i_deg=0.0
    )["detector_y_plus_m"]
    weak = np.empty((len(GAMMA2_GRID_DEG), len(THETA_GRID_DEG), len(data["strain_percent"])))
    denominator = np.empty_like(weak)
    shifted: dict[float, np.ndarray] = {}
    for gamma_index, gamma_value in enumerate(GAMMA2_GRID_DEG):
        readout = workflow.weak_readout(
            THETA_GRID_DEG, float(lambda_nm), fresnel, derivatives,
            gamma_i_deg=0.0, gamma_2_deg=float(gamma_value),
        )
        weak[gamma_index] = np.asarray(readout["y_weak_m"], dtype=float)
        denominator[gamma_index] = np.asarray(readout["denominator_m_inv2"], dtype=float)
    if include_shifts:
        for delta in GAMMA2_ERROR_DEG:
            shifted_cube = np.empty_like(weak)
            for gamma_index, gamma_value in enumerate(GAMMA2_GRID_DEG):
                shifted_cube[gamma_index] = np.asarray(workflow.weak_readout(
                    THETA_GRID_DEG, float(lambda_nm), fresnel, derivatives,
                    gamma_i_deg=0.0, gamma_2_deg=float(gamma_value + delta),
                )["y_weak_m"], dtype=float)
            shifted[delta] = shifted_cube
    return {
        "axis": axis, "lambda_nm": float(lambda_nm), "strain_percent": np.asarray(data["strain_percent"], dtype=float),
        "weak_m": weak, "denominator": denominator, "intrinsic_m": np.asarray(intrinsic, dtype=float),
        "detector_m": np.asarray(detector, dtype=float), "shifted_weak_m": shifted,
    }


def _enrich_gamma_robustness(table: pd.DataFrame, block: dict[str, Any]) -> pd.DataFrame:
    if table.empty:
        return table
    output = table.copy()
    errors, scores = [], []
    feature_cache: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    for _, row in output.iterrows():
        gamma_index = int(np.argmin(np.abs(GAMMA2_GRID_DEG - float(row["gamma_2_deg"]))))
        if row["protocol"] == "amplitude":
            theta_index = int(np.argmin(np.abs(THETA_GRID_DEG - float(row["theta_i_deg"]))))
            nominal = block["weak_m"][gamma_index, theta_index]
            shifted = [cube[gamma_index, theta_index] for cube in block["shifted_weak_m"].values()]
        elif row["protocol"] == "ratio_differential":
            first = int(np.argmin(np.abs(THETA_GRID_DEG - float(row["theta_1_deg"]))))
            second = int(np.argmin(np.abs(THETA_GRID_DEG - float(row["theta_2_deg"]))))
            kind = row["readout_type"]
            nominal_parts = ratio_differential_series(block["weak_m"][gamma_index, first], block["weak_m"][gamma_index, second])
            nominal = nominal_parts["differential" if kind == "D" else "ratio"]
            shifted = []
            for cube in block["shifted_weak_m"].values():
                parts = ratio_differential_series(cube[gamma_index, first], cube[gamma_index, second])
                candidate = parts["differential" if kind == "D" else "ratio"]
                if np.isfinite(candidate).all():
                    shifted.append(candidate)
        elif row["protocol"] == "feature_angle":
            mapping = {"zero_crossing": "zero_crossing_deg", "extremum": "extremum_angle_deg", "max_slope": "max_slope_angle_deg", "threshold": "threshold_angle_deg"}
            nominal_key = (0.0, gamma_index)
            if nominal_key not in feature_cache:
                feature_cache[nominal_key] = extract_feature_angles(THETA_GRID_DEG, block["weak_m"][gamma_index])
            nominal = feature_cache[nominal_key][mapping[row["feature_type"]]]
            shifted = []
            for delta, cube in block["shifted_weak_m"].items():
                cache_key = (float(delta), gamma_index)
                if cache_key not in feature_cache:
                    feature_cache[cache_key] = extract_feature_angles(THETA_GRID_DEG, cube[gamma_index])
                shifted.append(feature_cache[cache_key][mapping[row["feature_type"]]])
            shifted = [item for item in shifted if np.isfinite(item).all()]
        else:
            coordinates = [float(item) for item in str(row["feature_coordinates"]).split(";")]
            indices = [int(np.argmin(np.abs(THETA_GRID_DEG - item))) for item in coordinates]
            nominal = block["weak_m"][gamma_index, indices].ravel()
            shifted = [cube[gamma_index, indices].ravel() for cube in block["shifted_weak_m"].values()]
        robust = gamma_robustness(nominal, shifted)
        errors.append(robust["gamma2_error_normalized_rmse"])
        scores.append(robust["robustness_score"])
    output["gamma2_error_normalized_rmse"] = errors
    output["robustness_to_gamma2_error"] = scores
    if "screening_score" in output:
        output["screening_score"] = output["screening_score"] * output["robustness_to_gamma2_error"]
    return output


def _cross_gamma_candidates(block: dict[str, Any]) -> pd.DataFrame:
    rows = []
    strain = block["strain_percent"]
    gamma_indices = [int(np.argmin(np.abs(GAMMA2_GRID_DEG - item))) for item in (60.0, 70.0, 80.0, 85.0, 89.5)]
    theta_index = int(np.nanargmax(np.ptp(block["weak_m"], axis=2).mean(axis=0)))
    for left, right in zip(gamma_indices[:-1], gamma_indices[1:]):
        parts = ratio_differential_series(block["weak_m"][left, theta_index], block["weak_m"][right, theta_index])
        for kind, series in (("D", parts["differential"]), ("R", parts["ratio"])):
            if kind == "R" and not parts["ratio_valid"]:
                continue
            metrics = linear_metrics(strain, series)
            rows.append({
                "protocol_id": f"C_{kind}_gamma_{block['axis']}_{block['lambda_nm']:.3f}_{GAMMA2_GRID_DEG[left]:.1f}_{GAMMA2_GRID_DEG[right]:.1f}",
                "protocol": "ratio_differential", "readout_type": kind, "pair_dimension": "gamma_2",
                "axis": block["axis"], "lambda_nm": block["lambda_nm"], "gamma_2_deg": np.nan,
                "coordinate_1": GAMMA2_GRID_DEG[left], "coordinate_2": GAMMA2_GRID_DEG[right],
                "theta_1_deg": THETA_GRID_DEG[theta_index], "theta_2_deg": THETA_GRID_DEG[theta_index],
                "weak_denominator_relative_min": np.nan, "minimum_abs_ratio_denominator": parts["minimum_abs_ratio_denominator"],
                "ratio_valid": parts["ratio_valid"], "sensitivity_abs_per_percent": metrics["sensitivity_abs_per_percent"],
                **_common_linear_fields(metrics), "screening_score": metrics["sensitivity_abs_per_percent"] * max(metrics["r2"], 0.0),
                "gamma2_error_normalized_rmse": np.nan, "robustness_to_gamma2_error": np.nan,
            })
    return pd.DataFrame(rows)


def _cross_lambda_candidates(blocks: dict[tuple[str, float], dict[str, Any]], axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = np.array(sorted(key[1] for key in blocks if key[0] == axis))
    targets = (400.0, 500.0, 600.0, 632.8, 700.0, 780.0)
    lambdas = [float(available[np.argmin(np.abs(available - item))]) for item in targets]
    gamma_index = int(np.argmin(np.abs(GAMMA2_GRID_DEG - 85.0)))
    theta_index = int(np.argmin(np.abs(THETA_GRID_DEG - 70.0)))
    strain = blocks[(axis, lambdas[0])]["strain_percent"]
    ratio_rows = []
    for first_lambda, second_lambda in zip(lambdas[:-1], lambdas[1:]):
        first = blocks[(axis, first_lambda)]["weak_m"][gamma_index, theta_index]
        second = blocks[(axis, second_lambda)]["weak_m"][gamma_index, theta_index]
        parts = ratio_differential_series(first, second)
        for kind, series in (("D", parts["differential"]), ("R", parts["ratio"])):
            if kind == "R" and not parts["ratio_valid"]:
                continue
            metrics = linear_metrics(strain, series)
            ratio_rows.append({
                "protocol_id": f"C_{kind}_lambda_{axis}_{first_lambda:.1f}_{second_lambda:.1f}",
                "protocol": "ratio_differential", "readout_type": kind, "pair_dimension": "lambda",
                "axis": axis, "lambda_nm": np.nan, "gamma_2_deg": GAMMA2_GRID_DEG[gamma_index],
                "coordinate_1": first_lambda, "coordinate_2": second_lambda,
                "theta_1_deg": THETA_GRID_DEG[theta_index], "theta_2_deg": THETA_GRID_DEG[theta_index],
                "weak_denominator_relative_min": np.nan, "minimum_abs_ratio_denominator": parts["minimum_abs_ratio_denominator"],
                "ratio_valid": parts["ratio_valid"], "sensitivity_abs_per_percent": metrics["sensitivity_abs_per_percent"],
                **_common_linear_fields(metrics), "screening_score": metrics["sensitivity_abs_per_percent"] * max(metrics["r2"], 0.0),
                "gamma2_error_normalized_rmse": np.nan, "robustness_to_gamma2_error": np.nan,
            })
    matrix = np.column_stack([blocks[(axis, item)]["weak_m"][gamma_index, theta_index] for item in lambdas])
    fit = ridge_loocv(matrix, strain, alpha=1e-3)
    multi = pd.DataFrame([{
        "protocol_id": f"D_lambda_{axis}_85_70", "protocol": "multipoint_fitting",
        "fingerprint_dimension": "lambda", "axis": axis, "lambda_nm": np.nan,
        "gamma_2_deg": GAMMA2_GRID_DEG[gamma_index], "feature_coordinates": ";".join(map(str, lambdas)),
        "point_count": len(lambdas), "weak_denominator_relative_min": np.nan,
        "loocv_rmse_percent": fit["loocv_rmse_percent"], "loocv_mae_percent": fit["loocv_mae_percent"],
        "loocv_max_abs_error_percent": fit["loocv_max_abs_error_percent"], "prediction_r2": fit["prediction_r2"],
        "predicted_strain_percent": ";".join(f"{item:.8g}" for item in fit["predicted_strain_percent"]),
        "inversion_formula": "ridge regression on the listed Y_weak(lambda) fingerprint",
        "screening_score": max(fit["prediction_r2"], 0.0) / (1.0 + fit["loocv_rmse_percent"]),
        "gamma2_error_normalized_rmse": np.nan, "robustness_to_gamma2_error": np.nan,
    }])
    return pd.DataFrame(ratio_rows), multi


def _rank_and_summarize(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    for protocol, table in tables.items():
        for axis in ("a", "b"):
            subset = table[table["axis"] == axis].copy()
            if subset.empty:
                continue
            best = subset.sort_values("protocol_stability_score", ascending=False).iloc[0].to_dict()
            best.update({"protocol_group": protocol, "selection_scope": "theory_visible_scan"})
            summaries.append(best)
    best_summary = pd.DataFrame(summaries)
    experiment_rows = []
    for protocol, table in tables.items():
        if "lambda_nm" not in table:
            continue
        fixed = table[np.isclose(pd.to_numeric(table["lambda_nm"], errors="coerce"), 632.8, atol=1e-9)].copy()
        for axis in ("a", "b"):
            subset = fixed[fixed["axis"] == axis]
            if not subset.empty:
                row = subset.sort_values("protocol_stability_score", ascending=False).iloc[0].to_dict()
                row.update({"protocol_group": protocol, "selection_scope": "632.8_nm_experiment_compatible"})
                experiment_rows.append(row)
    return best_summary, pd.DataFrame(experiment_rows)


def _add_comparable_stability_metrics(tables: dict[str, pd.DataFrame]) -> None:
    """Add unit-independent inversion-error and stability metrics for cross-protocol ranking."""
    for key, table in tables.items():
        if key == "multipoint":
            inverse_error = pd.to_numeric(table["loocv_rmse_percent"], errors="coerce")
            fit_quality = pd.to_numeric(table["prediction_r2"], errors="coerce").clip(0.0, 1.0)
            monotonicity = pd.Series(1.0, index=table.index)
            sign_consistency = pd.Series(1.0, index=table.index)
        else:
            slope = pd.to_numeric(table["slope"], errors="coerce").abs()
            inverse_error = pd.to_numeric(table["loocv_rmse_readout"], errors="coerce").abs() / slope.replace(0.0, np.nan)
            if key == "feature":
                # A perfectly linear sequence on the 0.1-deg theta grid must not be assigned zero uncertainty.
                # Use the uniform-bin standard uncertainty of the feature locator as a conservative floor.
                angular_resolution_floor = 0.1 / math.sqrt(12.0) / slope.replace(0.0, np.nan)
                inverse_error = np.maximum(inverse_error, angular_resolution_floor)
            fit_quality = pd.to_numeric(table["r2"], errors="coerce").clip(0.0, 1.0)
            monotonicity = pd.to_numeric(table["monotonicity"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            sign_consistency = pd.to_numeric(table["sign_consistency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        robustness = pd.to_numeric(table.get("robustness_to_gamma2_error", 1.0), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        denominator = pd.to_numeric(table.get("weak_denominator_relative_min", 1.0), errors="coerce").fillna(1e-3)
        denominator_safety = np.clip(denominator / 1e-2, 0.0, 1.0)
        table["inversion_loocv_rmse_percent"] = inverse_error
        table["protocol_stability_score"] = (
            fit_quality * monotonicity * sign_consistency * robustness * denominator_safety
            / (1.0 + inverse_error / 0.05)
        ).fillna(0.0)


def _setup_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 600,
    })


def _plot_protocol_examples(tables: dict[str, pd.DataFrame], blocks: dict[tuple[str, float], dict[str, Any]]) -> None:
    _setup_plot_style()
    plot_specs = [
        ("amplitude", "strain_readout_amplitude_fit_examples.png", "Weak readout (m)"),
        ("feature", "strain_readout_feature_angle_fit_examples.png", "Feature angle (deg)"),
        ("ratio", "strain_readout_ratio_differential_fit_examples.png", "Differential / ratio readout"),
        ("multipoint", "strain_readout_multipoint_prediction_examples.png", "Predicted strain (%)"),
    ]
    for key, filename, ylabel in plot_specs:
        table = tables[key]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        for ax, axis in zip(axes, ("a", "b")):
            subset = table[table["axis"] == axis].sort_values("protocol_stability_score", ascending=False)
            if subset.empty:
                continue
            row = subset.iloc[0]
            if key == "multipoint":
                predicted = np.array([float(item) for item in str(row["predicted_strain_percent"]).split(";")])
                actual = np.arange(len(predicted)) * 0.1
                ax.plot(actual, predicted, "o-", label="LOO prediction")
                ax.plot(actual, actual, "--", color="0.35", label="ideal")
                ax.set_xlabel("Actual strain (%)")
            elif key == "feature":
                values = np.array([float(item) for item in str(row["feature_angles_deg"]).split(";")])
                strain = np.arange(len(values)) * 0.1
                ax.plot(strain, values, "o", label="calculated")
                ax.plot(strain, row["intercept"] + row["slope"] * strain, "--", label="linear fit")
                ax.set_xlabel("Strain (%)")
            else:
                lambda_value = float(row["lambda_nm"])
                block = blocks[(axis, lambda_value)]
                gamma_index = int(np.argmin(np.abs(GAMMA2_GRID_DEG - float(row["gamma_2_deg"]))))
                if key == "amplitude":
                    theta_index = int(np.argmin(np.abs(THETA_GRID_DEG - float(row["theta_i_deg"]))))
                    values = block["weak_m"][gamma_index, theta_index]
                else:
                    first = int(np.argmin(np.abs(THETA_GRID_DEG - float(row["theta_1_deg"]))))
                    second = int(np.argmin(np.abs(THETA_GRID_DEG - float(row["theta_2_deg"]))))
                    parts = ratio_differential_series(block["weak_m"][gamma_index, first], block["weak_m"][gamma_index, second])
                    values = parts["differential" if row["readout_type"] == "D" else "ratio"]
                strain = block["strain_percent"]
                ax.plot(strain, values, "o", label="calculated")
                ax.plot(strain, row["intercept"] + row["slope"] * strain, "--", label="linear fit")
                ax.set_xlabel("Strain (%)")
            ax.set_ylabel(ylabel); ax.set_title(f"{axis}-axis"); ax.legend(frameon=False, fontsize=7)
        fig.savefig(FIGURES / filename, dpi=600, bbox_inches="tight"); plt.close(fig)


def _plot_summary(best: pd.DataFrame, experiment: pd.DataFrame, amplitude: pd.DataFrame) -> None:
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    labels = best["protocol_group"].astype(str) + "-" + best["axis"].astype(str)
    values = best["protocol_stability_score"].to_numpy(float)
    ax.bar(np.arange(len(values)), values, color=plt.cm.tab10(np.arange(len(values)) % 10))
    ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right"); ax.set_ylabel("Unit-independent stability score")
    fig.savefig(FIGURES / "strain_readout_best_protocol_comparison.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    pivot_source = amplitude.copy()
    pivot_source["score_log"] = np.log10(np.maximum(pivot_source["screening_score"], 1e-30))
    pivot = pivot_source.groupby(["lambda_nm", "gamma_2_deg"], as_index=False)["score_log"].max().pivot(index="gamma_2_deg", columns="lambda_nm", values="score_log")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()], cmap="viridis")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel(r"Post-selection angle $\gamma_2$ (deg)")
    fig.colorbar(image, ax=ax, label="log10(max amplitude score)")
    fig.savefig(FIGURES / "strain_readout_theta_lambda_gamma2_screening_map.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for ax, axis in zip(axes, ("a", "b")):
        subset = experiment[experiment["axis"] == axis]
        ax.bar(subset["protocol_group"].astype(str), subset["protocol_stability_score"].to_numpy(float))
        ax.tick_params(axis="x", rotation=30); ax.set_title(f"{axis}-axis, 632.8 nm"); ax.set_ylabel("Unit-independent stability score")
    fig.savefig(FIGURES / "strain_readout_632p8nm_experiment_compatible_protocol.png", dpi=600, bbox_inches="tight"); plt.close(fig)


def _write_three_layer_trace(best: pd.DataFrame, experiment: pd.DataFrame, blocks: dict[tuple[str, float], dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    selections = [("theory", best.sort_values("protocol_stability_score", ascending=False).iloc[0]),
                  ("632.8_nm", experiment.sort_values("protocol_stability_score", ascending=False).iloc[0])]
    for scope, selected in selections:
        axis, lambda_nm = str(selected["axis"]), float(selected["lambda_nm"])
        block = blocks[(axis, lambda_nm)]
        gamma_value = float(selected["gamma_2_deg"])
        gamma_index = int(np.argmin(np.abs(GAMMA2_GRID_DEG - gamma_value)))
        if selected["protocol_group"] == "multipoint":
            theta_values = [float(item) for item in str(selected["feature_coordinates"]).split(";")]
        else:
            theta_values = [float(selected.get("theta_i_deg", selected.get("theta_1_deg")))]
        for theta_value in theta_values:
            theta_index = int(np.argmin(np.abs(THETA_GRID_DEG - theta_value)))
            for strain_index, strain_value in enumerate(block["strain_percent"]):
                rows.append({
                    "selection_scope": scope, "protocol_id": selected["protocol_id"], "axis": axis,
                    "lambda_nm": lambda_nm, "theta_i_deg": THETA_GRID_DEG[theta_index], "gamma_i_deg": 0.0,
                    "gamma_2_deg": gamma_value, "strain_percent": strain_value,
                    "intrinsic_spatial_pshe_m": block["intrinsic_m"][theta_index, strain_index],
                    "detector_plane_propagated_displacement_m": block["detector_m"][theta_index, strain_index],
                    "weak_measurement_readout_m": block["weak_m"][gamma_index, theta_index, strain_index],
                })
    pd.DataFrame(rows).to_csv(TABLES / "strain_readout_best_protocol_three_layer_trace.csv", index=False, encoding="utf-8-sig")


def _write_reports(best: pd.DataFrame, experiment: pd.DataFrame, max_residual: float, tables: dict[str, pd.DataFrame]) -> None:
    theory = best.sort_values("protocol_stability_score", ascending=False).iloc[0]
    lab = experiment.sort_values("protocol_stability_score", ascending=False).iloc[0]
    stable = theory
    report = f"""# Weak-measurement strain readout protocol screening

## Scope and fixed settings

- Optical-conductivity input: all available 0–0.5% strain cases for the a and b axes.
- Three layers are kept distinct: intrinsic spatial PSHE, detector-plane propagated displacement, and weak-measurement readout.
- Fixed parameters: gamma_i = 0 deg, w0 = 14.5 um, z_r = 0.1391 m.
- Theta scan: 55–85 deg in 0.1-deg steps; gamma_2 scan: 60–89.5 deg in 0.5-deg steps.
- Journal-oriented wavelength scan: 400–780 nm in 10-nm steps plus 632.8 nm.
- v9 Fresnel gate maximum residual: {max_residual:.3e} (< 1e-12).
- Absolute detected intensity is unavailable because the script expression omits the common optical-intensity prefactor. The normalized weak denominator (minimum divided by the block 95th percentile) is therefore used as the intensity-surrogate gate, with threshold 1e-3.
- The selected theory and 632.8-nm protocols have separate intrinsic, detector-plane, and weak-readout columns in `tables/strain_readout_best_protocol_three_layer_trace.csv`.

## Results

The highest-ranked theory candidate by the unit-independent stability score is `{theory['protocol_group']}` for the {theory['axis']}-axis (protocol `{theory['protocol_id']}`, inversion LOO RMSE {theory['inversion_loocv_rmse_percent']:.4g}% strain). The highest-ranked 632.8-nm candidate is `{lab['protocol_group']}` for the {lab['axis']}-axis (protocol `{lab['protocol_id']}`, inversion LOO RMSE {lab['inversion_loocv_rmse_percent']:.4g}% strain). The score combines fit quality, monotonicity, sign consistency, gamma_2 robustness, denominator safety, and strain-domain LOO error. Raw sensitivities with unlike units are not compared across protocol families.

The four implemented families are: fixed-point amplitude regression; theta-feature angle regression (zero crossing, extremum, maximum slope, threshold); two-point differential/ratio regression across theta, gamma_2, or wavelength; and ridge-regression multi-point fingerprints with leave-one-strain-out validation.

The recommended scalar inversion uses `epsilon_percent = (readout - intercept) / slope`, with the numerical coefficients stored in the selected candidate row. Multi-point fingerprints use ridge regression on the listed Y_weak vector and are judged by leave-one-out error.

Near-orthogonal post-selection is not accepted merely because it produces a large amplitude. Candidates are penalized/rejected by the denominator surrogate and gamma_2 ±0.1/±0.5 deg robustness. Enhancement factors are evaluated only where the intrinsic spatial PSHE exceeds max(1e-15 m, 1e-3 of the block maximum). Thus large values caused only by a vanishing denominator or vanishing intrinsic reference are not treated as valid sensitivity.

## Interpretation and manuscript placement

For a theory paper, the multi-point or best robust differential protocol is preferred when its leave-one-out error and gamma-angle sensitivity remain lower than the best scalar protocol; the exact selected row is recorded in the summary table. At 632.8 nm, use the experiment-compatible summary rather than the unrestricted wavelength optimum. A compact best-protocol comparison and one representative inversion plot are suitable for the main text. The full theta-lambda-gamma_2 map, all four candidate tables, branch/denominator checks, and alternative feature definitions belong in supplementary material.

These results are protocol-screening outputs under the fixed model parameters, not a final experimental calibration. They do not modify the intrinsic spatial PSHE or the v9 Fresnel alignment.
"""
    (REPORTS / "analysis_weak_strain_readout_protocol_screening.md").write_text(report, encoding="utf-8")
    manuscript = f"""# Manuscript text: weak-measurement strain readout protocol screening

We screened four strain-readout strategies using the calculated weak-measurement centroid while retaining the intrinsic spatial PSHE and detector-plane propagated displacement as separate physical layers. The incident angle, visible wavelength, and post-selection angle were varied under a normalized weak-denominator gate, and robustness was evaluated against post-selection errors of ±0.1° and ±0.5°. Fixed-point amplitude and feature-angle calibrations were compared with two-point differential/ratio observables and multi-point ridge-regression fingerprints using leave-one-strain-out validation. The unrestricted theory scan selected `{theory['protocol_id']}`, whereas the 632.8-nm experimental subset selected `{lab['protocol_id']}`. Scalar protocols use $\\epsilon=(Q-b)/m$, where $m$ and $b$ are the fitted slope and intercept reported in the candidate table. Large apparent enhancement near orthogonal post-selection was rejected when accompanied by an insufficient weak denominator or poor angular robustness. Consequently, the selected protocols reflect invertibility and stability rather than amplification alone.
"""
    (MANUSCRIPT / "manuscript_text_weak_strain_readout_protocol_screening.md").write_text(manuscript, encoding="utf-8")


def run_screening() -> None:
    for directory in (TABLES, FIGURES, REPORTS, MANUSCRIPT, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    if not GRIDS_PATH.exists():
        raise FileNotFoundError(f"Read-only optical-conductivity grid is missing: {GRIDS_PATH}")
    workflow = _load_base_workflow()
    _, max_residual = workflow.run_v9_fresnel_gate()
    if max_residual >= 1e-12:
        raise RuntimeError(f"v9 Fresnel alignment failed: {max_residual:.3e}")
    blocks: dict[tuple[str, float], dict[str, Any]] = {}
    amplitude_tables, feature_tables, ratio_tables, multipoint_tables = [], [], [], []
    for axis in ("a", "b"):
        for lambda_nm in LAMBDA_GRID_NM:
            block = _compute_block(workflow, axis, float(lambda_nm), include_shifts=True)
            blocks[(axis, float(lambda_nm))] = {key: value for key, value in block.items() if key != "shifted_weak_m"}
            amplitude = build_amplitude_candidates(axis, lambda_nm, THETA_GRID_DEG, block["strain_percent"], GAMMA2_GRID_DEG, block["weak_m"], block["denominator"], block["intrinsic_m"], top_per_gamma=5)
            feature = build_feature_angle_candidates(axis, lambda_nm, THETA_GRID_DEG, block["strain_percent"], GAMMA2_GRID_DEG, block["weak_m"], block["denominator"])
            ratio = build_ratio_differential_candidates(axis, lambda_nm, THETA_GRID_DEG, block["strain_percent"], GAMMA2_GRID_DEG, block["weak_m"], block["denominator"], top_theta_count=5)
            multipoint = build_multipoint_candidates(axis, lambda_nm, THETA_GRID_DEG, block["strain_percent"], GAMMA2_GRID_DEG, block["weak_m"], block["denominator"], point_count=5)
            amplitude_tables.append(_enrich_gamma_robustness(amplitude, block))
            feature_tables.append(_enrich_gamma_robustness(feature, block))
            ratio_tables.append(_enrich_gamma_robustness(ratio, block))
            multipoint_tables.append(_enrich_gamma_robustness(multipoint, block))
            ratio_tables.append(_cross_gamma_candidates(block))
            print(f"completed axis={axis}, lambda={lambda_nm:.1f} nm", flush=True)
    for axis in ("a", "b"):
        cross_ratio, cross_multi = _cross_lambda_candidates(blocks, axis)
        ratio_tables.append(cross_ratio); multipoint_tables.append(cross_multi)
    tables = {
        "amplitude": pd.concat(amplitude_tables, ignore_index=True),
        "feature": pd.concat(feature_tables, ignore_index=True),
        "ratio": pd.concat(ratio_tables, ignore_index=True),
        "multipoint": pd.concat(multipoint_tables, ignore_index=True),
    }
    _add_comparable_stability_metrics(tables)
    output_names = {
        "amplitude": "strain_readout_amplitude_candidates.csv",
        "feature": "strain_readout_feature_angle_candidates.csv",
        "ratio": "strain_readout_ratio_differential_candidates.csv",
        "multipoint": "strain_readout_multipoint_fitting_candidates.csv",
    }
    for key, table in tables.items():
        table.sort_values("screening_score", ascending=False).to_csv(TABLES / output_names[key], index=False, encoding="utf-8-sig")
    best, experiment = _rank_and_summarize(tables)
    best.to_csv(TABLES / "strain_readout_best_protocols_summary.csv", index=False, encoding="utf-8-sig")
    experiment.to_csv(TABLES / "strain_readout_632p8nm_experiment_compatible_summary.csv", index=False, encoding="utf-8-sig")
    _write_three_layer_trace(best, experiment, blocks)
    _plot_protocol_examples(tables, blocks)
    _plot_summary(best, experiment, tables["amplitude"])
    _write_reports(best, experiment, max_residual, tables)
    print(f"screening complete; Fresnel residual={max_residual:.3e}", flush=True)


if __name__ == "__main__":
    run_screening()
