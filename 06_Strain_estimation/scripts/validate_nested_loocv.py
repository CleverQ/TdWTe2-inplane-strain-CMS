"""嵌套 LOOCV 冻结结果的折次、误差与参数稳定性验证接口。"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

AB_ROOT = Path(__file__).resolve().parents[2]
DATA = AB_ROOT / "data" / "pshe"
FLOW = AB_ROOT / "upstream" / "optical_pshe" / "current_pipeline" / "ab_pshe_output" / "weak_measurement_workflow"
SCREEN = FLOW / "scripts" / "run_strain_readout_protocol_screening.py"
GRIDS = Path(__file__).resolve().parents[2] / "04_Optical_conductivity" / "data" / "processed" / "grids.mat"
STRAINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
FIXED_LAMBDA = 700.0
FIXED_GAMMA = 82.0
FIXED_ANGLES = np.array([70.3, 70.4, 70.5, 70.6, 70.7], dtype=float)
ALPHA = 1e-3


def load_nested_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA / "v30A_full_nested_loocv_fold_results.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "v30A_full_nested_parameter_selection_by_fold.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "v30A_parameter_stability_analysis.csv", encoding="utf-8-sig"),
    )


def prediction_metrics(folds: pd.DataFrame) -> dict[str, float]:
    true_name = "true_strain_percent" if "true_strain_percent" in folds.columns else "held_out_strain_percent"
    pred_name = "predicted_strain_percent"
    residual = folds[pred_name].to_numpy(float) - folds[true_name].to_numpy(float)
    true = folds[true_name].to_numpy(float)
    return {"rmse_percent": float(np.sqrt(np.mean(residual ** 2))), "predictive_r2": float(1.0 - np.sum(residual ** 2) / np.sum((true - true.mean()) ** 2)), "max_abs_error_percent": float(np.max(np.abs(residual)))}


def load_authoritative_screening():
    """载入原 nested-LOOCV 所调用的上游候选协议实现，不依赖历史论文目录。"""
    if not SCREEN.is_file() or not GRIDS.is_file():
        raise FileNotFoundError("缺少 nested-LOOCV 上游筛选脚本或 grids.mat")
    spec = importlib.util.spec_from_file_location("canonical_nested_screening", SCREEN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load screening script: {SCREEN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ridge_fold(features: np.ndarray, targets: np.ndarray, train: np.ndarray, test: int, alpha: float = ALPHA) -> dict[str, object]:
    """直接保留 v30A 的 training-fold-only 标准化、截距不正则化和 solve 形式。"""
    x_train, y_train = np.asarray(features[train], dtype=float), np.asarray(targets[train], dtype=float)
    x_test = np.asarray(features[[test]], dtype=float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-15, scale, 1.0)
    z_train, z_test = (x_train - mean) / scale, (x_test - mean) / scale
    design = np.column_stack([np.ones(len(train)), z_train])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return {"mean": mean, "scale": scale, "coefficients": coefficients[1:], "intercept": float(coefficients[0]), "prediction": float((np.column_stack([np.ones(1), z_test]) @ coefficients)[0])}


def fixed_parameter_features() -> np.ndarray:
    """v30A 固定 λ=700 nm、γ2=82°、五角度 a-axis 特征矩阵。"""
    frame = pd.read_csv(DATA / "v58_ab_axis_weak_measurement_metrics.csv", encoding="utf-8-sig")
    selected = frame[(frame["axis"] == "a") & np.isclose(frame["lambda_nm"], FIXED_LAMBDA) & np.isclose(frame["gamma2_deg"], FIXED_GAMMA)].sort_values("theta_i_deg")
    if not np.allclose(selected["theta_i_deg"].to_numpy(float), FIXED_ANGLES):
        raise RuntimeError("v58 metric table does not contain the five fixed angles")
    return selected[[f"D_0p{index}_um" for index in range(6)]].to_numpy(float).T


def fixed_parameter_loocv() -> tuple[dict[str, object], pd.DataFrame]:
    features = fixed_parameter_features()
    rows, predictions = [], np.full(len(STRAINS), np.nan)
    for test in range(len(STRAINS)):
        train = np.array([index for index in range(len(STRAINS)) if index != test], dtype=int)
        fold = ridge_fold(features, STRAINS, train, test)
        predictions[test] = float(fold["prediction"])
        error = predictions[test] - STRAINS[test]
        rows.append({"outer_fold": test + 1, "held_out_strain_percent": STRAINS[test], "training_strains_percent": ";".join(f"{STRAINS[index]:.1f}" for index in train), "fixed_lambda_nm": FIXED_LAMBDA, "fixed_gamma2_deg": FIXED_GAMMA, "fixed_angles_deg": ";".join(f"{value:.1f}" for value in FIXED_ANGLES), "standardization_mean_um": json.dumps(fold["mean"].tolist()), "standardization_std_um": json.dumps(fold["scale"].tolist()), "ridge_alpha": ALPHA, "ridge_coefficients": json.dumps(fold["coefficients"].tolist()), "ridge_intercept": fold["intercept"], "predicted_strain_percent": predictions[test], "signed_error_percent": error, "absolute_error_percent": abs(error), "training_only_standardization": "yes", "training_only_ridge_fit": "yes"})
    errors = predictions - STRAINS
    summary = {"rmse_percent": float(np.sqrt(np.mean(errors ** 2))), "mae_percent": float(np.mean(np.abs(errors))), "max_abs_error_percent": float(np.max(np.abs(errors))), "prediction_r2": float(1.0 - np.sum(errors ** 2) / np.sum((STRAINS - STRAINS.mean()) ** 2)), "predicted_strain_percent": predictions.tolist()}
    return summary, pd.DataFrame(rows)


def _select_multipoint_from_training(screening, block: dict, train: np.ndarray) -> pd.DataFrame:
    """v30A inner training fold 的 five-angle 候选、γ2 扰动稳健性和可比稳定性指标。"""
    train_strain = np.asarray(block["strain_percent"], dtype=float)[train]
    weak_train = np.asarray(block["weak_m"], dtype=float)[:, :, train]
    denominator_train = np.asarray(block["denominator"], dtype=float)[:, :, train]
    candidates = screening.build_multipoint_candidates("a", float(block["lambda_nm"]), screening.THETA_GRID_DEG, train_strain, screening.GAMMA2_GRID_DEG, weak_train, denominator_train, point_count=5)
    if candidates.empty:
        raise RuntimeError(f"No valid multipoint candidates at lambda={block['lambda_nm']}")
    local = {"weak_m": weak_train, "shifted_weak_m": {delta: np.asarray(cube, dtype=float)[:, :, train] for delta, cube in block["shifted_weak_m"].items()}}
    candidates = screening._enrich_gamma_robustness(candidates, local)
    tables = {"multipoint": candidates}
    screening._add_comparable_stability_metrics(tables)
    return tables["multipoint"]


def nested_parameter_selection() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """完整保留 v30A：outer leave-one-state-out、inner candidate grid、score-first 排序、denominator gate 与 outer prediction。"""
    screening = load_authoritative_screening()
    workflow = screening._load_base_workflow()
    raw = workflow.load_axis_sigma_s_cm(GRIDS, "a", FIXED_LAMBDA)
    if not np.array_equal(np.asarray(raw["strain_percent"], dtype=float), STRAINS):
        raise RuntimeError("Authoritative a-axis strain grid differs from six-state design")
    blocks = {float(wavelength): screening._compute_block(workflow, "a", float(wavelength), include_shifts=True) for wavelength in np.asarray(screening.LAMBDA_GRID_NM, dtype=float)}
    selections, folds = [], []
    for test in range(len(STRAINS)):
        train = np.array([index for index in range(len(STRAINS)) if index != test], dtype=int)
        candidates = pd.concat([_select_multipoint_from_training(screening, block, train) for block in blocks.values()], ignore_index=True)
        ordered = candidates.sort_values(["protocol_stability_score", "loocv_rmse_percent", "prediction_r2", "lambda_nm", "gamma_2_deg"], ascending=[False, True, False, True, True], kind="mergesort")
        selected = ordered.iloc[0]
        top_score = float(selected["protocol_stability_score"])
        tied = ordered[np.isclose(ordered["protocol_stability_score"].to_numpy(float), top_score, rtol=0.0, atol=1e-14)]
        angles = np.array([float(value) for value in str(selected["feature_coordinates"]).split(";")], dtype=float)
        block = blocks[float(selected["lambda_nm"])]
        gamma_index = int(np.argmin(np.abs(np.asarray(screening.GAMMA2_GRID_DEG, dtype=float) - float(selected["gamma_2_deg"]))))
        theta_indices = [int(np.argmin(np.abs(np.asarray(screening.THETA_GRID_DEG, dtype=float) - angle))) for angle in angles]
        features = np.asarray(block["weak_m"], dtype=float)[gamma_index, theta_indices, :].T * 1e6
        fold = ridge_fold(features, STRAINS, train, test)
        denominator_train = np.asarray(block["denominator"], dtype=float)[gamma_index, theta_indices, :][:, train]
        relative_margin, accepted = screening.denominator_gate(denominator_train, threshold=1e-3)
        error = float(fold["prediction"]) - STRAINS[test]
        status = "pass" if len(tied) == 1 and bool(np.all(accepted)) else "conditional"
        selections.append({"outer_fold": test + 1, "held_out_strain_percent": STRAINS[test], "training_strains_percent": ";".join(f"{STRAINS[index]:.1f}" for index in train), "candidate_lambda_count": len(screening.LAMBDA_GRID_NM), "candidate_gamma2_count": len(screening.GAMMA2_GRID_DEG), "candidate_theta_count": len(screening.THETA_GRID_DEG), "selected_lambda_nm": float(selected["lambda_nm"]), "selected_gamma2_deg": float(selected["gamma_2_deg"]), "selected_five_angles_deg": ";".join(f"{value:.1f}" for value in angles), "inner_loocv_rmse_percent": float(selected["loocv_rmse_percent"]), "inner_prediction_r2": float(selected["prediction_r2"]), "relative_min_denominator_training": float(np.min(relative_margin)), "gamma2_error_normalized_rmse_training": float(selected["gamma2_error_normalized_rmse"]), "protocol_stability_score_training": top_score, "top_score_tie_count": len(tied), "test_label_used_in_selection": "no", "outer_fold_status": status})
        folds.append({"outer_fold": test + 1, "held_out_strain_percent": STRAINS[test], "training_strains_percent": ";".join(f"{STRAINS[index]:.1f}" for index in train), "selected_lambda_nm": float(selected["lambda_nm"]), "selected_gamma2_deg": float(selected["gamma_2_deg"]), "selected_five_angles_deg": ";".join(f"{value:.1f}" for value in angles), "standardization_mean_um": json.dumps(fold["mean"].tolist()), "standardization_std_um": json.dumps(fold["scale"].tolist()), "ridge_alpha": ALPHA, "ridge_coefficients": json.dumps(fold["coefficients"].tolist()), "ridge_intercept": fold["intercept"], "predicted_strain_percent": float(fold["prediction"]), "true_strain_percent": STRAINS[test], "signed_error_percent": error, "absolute_error_percent": abs(error), "test_label_used_before_prediction": "no", "outer_fold_status": status})
    fold_frame = pd.DataFrame(folds)
    metrics = prediction_metrics(fold_frame)
    summary = {"analysis": "full_nested_LOOCV_conditioned_on_authoritative_multipoint_family", "status": "pass" if (fold_frame["outer_fold_status"] == "pass").all() else "conditional", "rmse_percent": metrics["rmse_percent"], "mae_percent": float(fold_frame["absolute_error_percent"].mean()), "max_abs_error_percent": metrics["max_abs_error_percent"], "prediction_r2": metrics["predictive_r2"], "selected_parameter_combinations": "; ".join(f"{row['selected_lambda_nm']:.1f}/{row['selected_gamma2_deg']:.1f}/{row['selected_five_angles_deg']}" for row in selections), "conditional_fold_count": int((fold_frame["outer_fold_status"] == "conditional").sum())}
    return pd.DataFrame(selections), fold_frame, summary


def parameter_stability(selected_by_fold: pd.DataFrame) -> pd.DataFrame:
    """由每个 outer fold 的实际选定 λ、γ2 与五角度计算 v30A 参数稳定性汇总。"""
    rows = []
    for column in ("selected_lambda_nm", "selected_gamma2_deg"):
        values = selected_by_fold[column].to_numpy(float)
        rows.append({"parameter": column, "minimum": float(values.min()), "maximum": float(values.max()), "mean": float(values.mean()), "std_ddof0": float(values.std(ddof=0)), "unique_count": int(len(np.unique(values)))})
    angles = selected_by_fold["selected_five_angles_deg"].str.split(";", expand=True).astype(float)
    for index in range(angles.shape[1]):
        values = angles.iloc[:, index].to_numpy(float)
        rows.append({"parameter": f"selected_angle_{index + 1}_deg", "minimum": float(values.min()), "maximum": float(values.max()), "mean": float(values.mean()), "std_ddof0": float(values.std(ddof=0)), "unique_count": int(len(np.unique(values)))})
    return pd.DataFrame(rows)


def fixed_vs_nested_comparison() -> dict[str, object]:
    """组织固定协议与 nested 选择的逐折预测、误差和汇总，保留两者严格独立的评估流程。"""
    fixed_summary, fixed_folds = fixed_parameter_loocv()
    selected, nested_folds, nested_summary = nested_parameter_selection()
    return {"fixed_summary": fixed_summary, "fixed_folds": fixed_folds, "nested_selected_parameters": selected, "nested_folds": nested_folds, "nested_summary": nested_summary, "parameter_stability": parameter_stability(selected)}
