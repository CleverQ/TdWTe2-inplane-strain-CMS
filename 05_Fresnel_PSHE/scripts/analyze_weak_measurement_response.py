"""a/b 轴五角度弱测量响应与固定参数 ridge-LOOCV 的可复用科学核心。"""
from pathlib import Path
import importlib.util
import math
import numpy as np
import pandas as pd

AB_ROOT = Path(__file__).resolve().parents[2]
METRICS = AB_ROOT / "data" / "pshe" / "v58_ab_axis_weak_measurement_metrics.csv"
STRAINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
ANGLES = np.array([70.3, 70.4, 70.5, 70.6, 70.7], dtype=float)
LAMBDA_NM = 700.0
GAMMA2_DEG = 82.0
ALPHA = 1e-3
FLOW = AB_ROOT / "upstream" / "optical_pshe" / "current_pipeline" / "ab_pshe_output" / "weak_measurement_workflow"
SCREENING_SCRIPT = FLOW / "scripts" / "run_strain_readout_protocol_screening.py"
OPTICAL_GRIDS = Path(__file__).resolve().parents[2] / "04_Optical_conductivity" / "data" / "processed" / "grids.mat"
THETA_LAMBDA_TABLE = AB_ROOT / "upstream" / "optical_pshe" / "current_pipeline" / "ab_pshe_output" / "paper_package_theoretical_optimal" / "tables" / "table_weak_readout_theta_lambda_key_values.csv"


def response_matrix(axis: str = "a") -> np.ndarray:
    frame = pd.read_csv(METRICS, encoding="utf-8-sig")
    selected = frame[(frame["axis"] == axis) & np.isclose(frame["lambda_nm"], LAMBDA_NM) & np.isclose(frame["gamma2_deg"], GAMMA2_DEG)]
    value = "weak_readout_um" if "weak_readout_um" in selected.columns else selected.select_dtypes(include="number").columns[-1]
    matrix = selected.pivot(index="strain_percent", columns="theta_i_deg", values=value).reindex(index=STRAINS, columns=ANGLES)
    return matrix.to_numpy(float)


def ridge_loocv(features: np.ndarray, targets: np.ndarray = STRAINS, alpha: float = ALPHA) -> np.ndarray:
    predictions = np.empty(len(targets), dtype=float)
    for held_out in range(len(targets)):
        train = np.arange(len(targets)) != held_out
        mean, std = features[train].mean(0), features[train].std(0, ddof=0)
        standardized = (features - mean) / std
        design = np.c_[np.ones(train.sum()), standardized[train]]
        penalty = np.diag([0.0] + [alpha] * features.shape[1])
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ targets[train])
        predictions[held_out] = np.r_[1.0, standardized[held_out]] @ coef
    return predictions


def load_screening_workflow():
    """加载已记录的上游 PSHE 筛选实现；不涉及论文目录、Word 或报告。"""
    if not SCREENING_SCRIPT.is_file():
        raise FileNotFoundError(SCREENING_SCRIPT)
    spec = importlib.util.spec_from_file_location("canonical_pshe_screening", SCREENING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load screening workflow: {SCREENING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module._load_base_workflow()


def reconstruct_axis_response(axis: str) -> dict[str, object]:
    """保留 v58 的 a/b 轴、70.5° 对照、五角度重构和固定参数 ridge-LOOCV 科学数据流。"""
    screening, workflow = load_screening_workflow()
    raw = workflow.load_axis_sigma_s_cm(OPTICAL_GRIDS, axis, LAMBDA_NM)
    strain = np.asarray(raw["strain_percent"], dtype=float)
    if not np.array_equal(strain, STRAINS):
        raise RuntimeError(f"{axis} 轴六个权威应变状态不完整：{strain.tolist()}")
    block = screening._compute_block(workflow, axis, LAMBDA_NM, include_shifts=False)
    gamma_index = int(np.argmin(np.abs(screening.GAMMA2_GRID_DEG - GAMMA2_DEG)))
    if not math.isclose(float(screening.GAMMA2_GRID_DEG[gamma_index]), GAMMA2_DEG, abs_tol=1e-12):
        raise RuntimeError("无法定位 gamma2=82°")
    angle_indices = [int(np.argmin(np.abs(screening.THETA_GRID_DEG - value))) for value in ANGLES]
    selected_angles = np.asarray(screening.THETA_GRID_DEG[angle_indices], dtype=float)
    if not np.allclose(selected_angles, ANGLES, atol=1e-12):
        raise RuntimeError("无法定位五个入射角")
    displacement_um = np.asarray(block["weak_m"][gamma_index, angle_indices, :].T, dtype=float) * 1e6
    metrics: list[dict[str, object]] = []
    for column, theta in enumerate(ANGLES):
        linear = screening.linear_metrics(STRAINS, displacement_um[:, column])
        differences = np.diff(displacement_um[:, column])
        monotonicity = "increasing" if np.all(differences > 0) else "decreasing" if np.all(differences < 0) else "nonmonotonic"
        metrics.append({"axis": axis, "theta_i_deg": float(theta), "slope_um_per_percent": float(linear["slope"]), "r2": float(linear["r2"]), "total_change_um": float(displacement_um[-1, column] - displacement_um[0, column]), "min_adjacent_abs_um": float(np.min(np.abs(differences))), "monotonicity": monotonicity})
    slope_order = np.argsort(-np.abs([float(row["slope_um_per_percent"]) for row in metrics]))
    ridge_five = screening.ridge_loocv(displacement_um[:, slope_order], STRAINS, ALPHA)
    ridge_single: dict[float, dict[str, object]] = {}
    for theta in (70.4, 70.5):
        index = int(np.where(np.isclose(ANGLES, theta))[0][0])
        ridge_single[theta] = screening.ridge_loocv(displacement_um[:, [index]], STRAINS, ALPHA)
    return {"axis": axis, "d_um": displacement_um, "metrics": metrics, "order": ANGLES[slope_order], "ridge5": ridge_five, "ridge1": ridge_single, "average_abs_slope": float(np.mean(np.abs([row["slope_um_per_percent"] for row in metrics]))), "minimum_r2": float(np.min([row["r2"] for row in metrics])), "all_monotonic": all(row["monotonicity"] != "nonmonotonic" for row in metrics)}


def wavelength_angle_response_map() -> dict[str, object]:
    """保留 v58 Fig. 9 的 0%、a 轴、γ2=82° θi–λ 质心位移二维数据构造与工作点。"""
    frame = pd.read_csv(THETA_LAMBDA_TABLE, encoding="utf-8-sig")
    value_column = "weak_readout_um"
    if value_column not in frame.columns:
        raise KeyError(f"缺少 PSHE 位移列：{value_column}")
    subset = frame[(frame["case"] == "0%") & (frame["axis"] == "a") & np.isclose(frame["gamma_2_deg"], GAMMA2_DEG)].copy()
    response = subset.pivot(index="theta_i_deg", columns="lambda_nm", values=value_column).sort_index().sort_index(axis=1)
    values = response.to_numpy(float)
    if values.shape != (151, 77):
        raise RuntimeError("Fig. 9 数据网格尺寸变化")
    return {"wavelength_nm": response.columns.to_numpy(float), "incident_angle_deg": response.index.to_numpy(float), "centroid_displacement_um": values, "case": "0%", "axis": "a", "gamma2_deg": GAMMA2_DEG, "working_point": {"lambda_nm": LAMBDA_NM, "theta_i_deg": 70.5}, "display_range_um": (-250.0, 250.0)}


def fixed_parameter_scientific_tables() -> dict[str, pd.DataFrame]:
    """从 v58 原始计算结果构造长期使用的 a/b 曲线、五角度特征及 70.5°/五角度 LOOCV 数据表。"""
    a_data = reconstruct_axis_response("a")
    b_data = reconstruct_axis_response("b")
    response_rows = []
    for data in (a_data, b_data):
        matrix = np.asarray(data["d_um"], dtype=float)
        for column, metric in enumerate(data["metrics"]):
            row = {"axis": metric["axis"], "theta_i_deg": float(metric["theta_i_deg"]), "lambda_nm": LAMBDA_NM, "gamma2_deg": GAMMA2_DEG, "slope_um_per_percent": float(metric["slope_um_per_percent"]), "r2": float(metric["r2"]), "total_change_um": float(metric["total_change_um"]), "min_adjacent_abs_um": float(metric["min_adjacent_abs_um"]), "monotonicity": str(metric["monotonicity"]), "axis_average_abs_slope": float(data["average_abs_slope"]), "axis_minimum_r2": float(data["minimum_r2"]), "axis_all_monotonic": bool(data["all_monotonic"])}
            for index, strain in enumerate(STRAINS):
                row[f"D_{strain:.1f}_um"] = float(matrix[index, column])
            response_rows.append(row)
    prediction_rows = []
    for protocol, result in (("single_angle_70p5", a_data["ridge1"][70.5]), ("five_angle", a_data["ridge5"])):
        predictions = np.asarray(result["predicted_strain_percent"], dtype=float)
        for fold, (truth, prediction) in enumerate(zip(STRAINS, predictions), start=1):
            error = float(prediction - truth)
            prediction_rows.append({"protocol": protocol, "fold": fold, "held_out_strain_percent": float(truth), "predicted_strain_percent": float(prediction), "signed_error_percent": error, "absolute_error_percent": abs(error), "rmse_percent": float(result["loocv_rmse_percent"]), "prediction_r2": float(result["prediction_r2"]), "max_abs_error_percent": float(result["loocv_max_abs_error_percent"]), "ridge_alpha": ALPHA, "training_only_standardization": True})
    feature_matrix = pd.DataFrame(np.asarray(a_data["d_um"], dtype=float), index=STRAINS, columns=ANGLES)
    feature_matrix.index.name = "strain_percent"
    feature_matrix.columns.name = "theta_i_deg"
    return {"response_metrics": pd.DataFrame(response_rows), "five_angle_feature_matrix_a_axis": feature_matrix, "fixed_parameter_loocv": pd.DataFrame(prediction_rows), "a_axis": a_data, "b_axis": b_data}


def a_axis_protocol_selection_supported(a_data: dict[str, object], b_data: dict[str, object]) -> bool:
    """保留 v58 对 a/b 固定工作点比较的选择条件。"""
    a70, b70 = a_data["metrics"][2], b_data["metrics"][2]
    return bool(float(a_data["average_abs_slope"]) > float(b_data["average_abs_slope"]) and float(a_data["minimum_r2"]) > float(b_data["minimum_r2"]) and a_data["all_monotonic"] and not b_data["all_monotonic"] and float(a70["min_adjacent_abs_um"]) > float(b70["min_adjacent_abs_um"]))


def strain_response_curve_data(axis_data: dict[str, object], angle_index: int) -> dict[str, np.ndarray | float]:
    """返回 v58 科学曲线所用的 D–strain 点、线性拟合与对应入射角，不含出版版式。"""
    values = np.asarray(axis_data["d_um"], dtype=float)[:, angle_index]
    coefficients = np.polyfit(STRAINS, values, 1)
    return {"strain_percent": STRAINS.copy(), "centroid_displacement_um": values, "linear_fit_um": np.polyval(coefficients, STRAINS), "slope_um_per_percent": float(coefficients[0]), "intercept_um": float(coefficients[1]), "theta_i_deg": float(ANGLES[angle_index]), "lambda_nm": LAMBDA_NM, "gamma2_deg": GAMMA2_DEG}
