"""完整复 Fresnel 矩阵的 canonical 数据接口与连续相位分析。"""
from pathlib import Path
import numpy as np
import pandas as pd

AB_ROOT = Path(__file__).resolve().parents[2]
FRESNEL_CSV = AB_ROOT / "data" / "fresnel" / "v33_complex_Fresnel_matrix_summary.csv"
SUBSTITUTION_CSV = AB_ROOT / "data" / "fresnel" / "v33_conductivity_component_substitution_analysis.csv"


def load_fresnel_matrix() -> pd.DataFrame:
    return pd.read_csv(FRESNEL_CSV, encoding="utf-8-sig")


def unwrap_phase_deg(values: np.ndarray) -> np.ndarray:
    """保持原 Fresnel 连续相位定义：先转弧度 unwrap，再转回角度。"""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=float))))


def wavelength_states() -> list[str]:
    frame = load_fresnel_matrix()
    for name in ("state", "strain_state", "case"):
        if name in frame.columns:
            return [str(value) for value in frame[name].drop_duplicates()]
    return []


def load_component_substitution() -> pd.DataFrame:
    """读取 v33/v34 沿用的 baseline、full、四个单分量替换与 nonlinear-residual 光谱。"""
    return pd.read_csv(SUBSTITUTION_CSV, encoding="utf-8-sig")


def representative_substitution_metrics() -> pd.DataFrame:
    """直接保留 v34 的六个 representative wavelength/channel/component 选择及幅值差定义。"""
    substitution = load_component_substitution()
    rows = []
    selections = (
        ("a +0.5%", "r_pp", "replace_xx", 603.3),
        ("a +0.5%", "r_ps", "replace_yx", 753.4),
        ("a +0.5%", "r_sp", "replace_xy", 753.4),
        ("b +0.5%", "r_pp", "replace_xx", 431.35),
        ("b +0.5%", "r_ps", "replace_yx", 730.6),
        ("b +0.5%", "r_sp", "replace_xy", 729.65),
    )
    for state, element, scenario, wavelength in selections:
        selected = substitution[(substitution["state"] == state) & (substitution["scenario"] == scenario)]
        row = selected.iloc[(selected["lambda_nm"] - wavelength).abs().argmin()]
        baseline = substitution[(substitution["state"] == state) & (substitution["scenario"] == "baseline")]
        reference = baseline.iloc[(baseline["lambda_nm"] - row["lambda_nm"]).abs().argmin()]
        rows.append({"state": state, "channel": element, "replacement": scenario.replace("replace_", ""), "lambda_nm": float(row["lambda_nm"]), "absolute_amplitude_change": abs(float(row[f"abs_{element}"]) - float(reference[f"abs_{element}"]))})
    return pd.DataFrame(rows)


def spectral_residual_summary() -> pd.DataFrame:
    """从同一四分量替换表返回各 state/channel 的 nonlinear complex residual 幅值统计。"""
    substitution = load_component_substitution()
    residual = substitution[substitution["scenario"].eq("nonlinear_residual")].copy()
    rows = []
    for state, frame in residual.groupby("state", sort=False):
        for element in ("r_pp", "r_ps", "r_sp", "r_ss"):
            column = f"abs_{element}"
            if column not in frame:
                continue
            values = frame[column].dropna().to_numpy(float)
            if len(values):
                rows.append({"state": state, "channel": element, "wavelength_min_nm": float(frame["lambda_nm"].min()), "wavelength_max_nm": float(frame["lambda_nm"].max()), "nonlinear_residual_max": float(values.max()), "nonlinear_residual_mean": float(values.mean())})
    return pd.DataFrame(rows)
