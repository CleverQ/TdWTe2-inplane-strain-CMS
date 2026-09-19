from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
TABLES = WORKFLOW_ROOT / "tables"
FIGURES = WORKFLOW_ROOT / "figures"
REPORTS = WORKFLOW_ROOT / "reports"
MANUSCRIPT = WORKFLOW_ROOT / "manuscript_text"
LOGS = WORKFLOW_ROOT / "logs"
os.environ.setdefault("MPLCONFIGDIR", str(LOGS / "matplotlib_cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


THEORY_SOURCE = TABLES / "strain_readout_best_protocols_summary.csv"
EXPERIMENT_SOURCE = TABLES / "strain_readout_632p8nm_experiment_compatible_summary.csv"
RANKING_OUTPUT = TABLES / "strain_readout_final_ranking_fit_first.csv"
RECOMMENDATION_OUTPUT = TABLES / "strain_readout_final_recommendation_fit_first.csv"
FIGURE_OUTPUT = FIGURES / "strain_readout_final_ranking_fit_first.png"
REPORT_OUTPUT = REPORTS / "analysis_strain_readout_final_recommendation_fit_first.md"
MANUSCRIPT_OUTPUT = MANUSCRIPT / "manuscript_text_strain_readout_final_recommendation_fit_first.md"


PROTOCOL_LABELS = {
    "amplitude": "Amplitude",
    "feature": "Feature-angle",
    "ratio": "Ratio/differential",
    "multipoint": "Multi-point fitting",
}

FEASIBILITY = {
    "amplitude": (4, "high: one fixed theta and one scalar weak-readout measurement"),
    "feature": (3, "medium: requires a theta scan and robust feature localization"),
    "ratio": (3, "medium: requires two coordinated measurements"),
    "multipoint": (2, "medium-low: requires five-point acquisition and regression calibration"),
}


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_summary(source: Path, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"Empty summary table: {source}")
    required = {"protocol_id", "protocol_group", "axis", "lambda_nm"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns in {source.name}: {missing}")

    output = frame.copy()
    output["ranking_scope"] = scope
    output["protocol_label"] = output["protocol_group"].map(PROTOCOL_LABELS)
    output["fit_rmse_percent"] = _finite_numeric(output, "inversion_loocv_rmse_percent")
    output["fit_rmse_percent"] = output["fit_rmse_percent"].fillna(_finite_numeric(output, "loocv_rmse_percent"))
    output["fit_r2"] = _finite_numeric(output, "prediction_r2").fillna(_finite_numeric(output, "r2"))

    output["sensitivity_value"] = np.nan
    output["sensitivity_unit"] = "not scalar; vector fingerprint"
    amplitude = output["protocol_group"].eq("amplitude")
    feature = output["protocol_group"].eq("feature")
    ratio = output["protocol_group"].eq("ratio")
    output.loc[amplitude, "sensitivity_value"] = _finite_numeric(output, "sensitivity_abs_m_per_percent")[amplitude]
    output.loc[amplitude, "sensitivity_unit"] = "m per % strain"
    output.loc[feature, "sensitivity_value"] = _finite_numeric(output, "sensitivity_abs_deg_per_percent")[feature]
    output.loc[feature, "sensitivity_unit"] = "deg per % strain"
    output.loc[ratio, "sensitivity_value"] = _finite_numeric(output, "sensitivity_abs_per_percent")[ratio]
    output.loc[ratio, "sensitivity_unit"] = "readout unit per % strain"

    output["sensitivity_rank_within_protocol"] = output.groupby("protocol_group")["sensitivity_value"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    output["sensitivity_sort_within_protocol"] = output.groupby("protocol_group")["sensitivity_value"].transform(
        lambda values: values.rank(pct=True, ascending=True).fillna(0.0)
    )
    output["experimental_feasibility_rank"] = output["protocol_group"].map(lambda item: FEASIBILITY[item][0])
    output["experimental_feasibility"] = output["protocol_group"].map(lambda item: FEASIBILITY[item][1])
    output["monotonicity_fit_first"] = _finite_numeric(output, "monotonicity").fillna(1.0)
    output["sign_consistency_fit_first"] = _finite_numeric(output, "sign_consistency").fillna(1.0)
    output["weak_denominator_relative_min_fit_first"] = _finite_numeric(
        output, "weak_denominator_relative_min"
    ).fillna(0.0)
    output["robustness_to_gamma2_error_fit_first"] = _finite_numeric(
        output, "robustness_to_gamma2_error"
    ).fillna(0.0)

    output = output.sort_values(
        [
            "fit_rmse_percent",
            "fit_r2",
            "sensitivity_sort_within_protocol",
            "monotonicity_fit_first",
            "sign_consistency_fit_first",
            "weak_denominator_relative_min_fit_first",
            "robustness_to_gamma2_error_fit_first",
            "experimental_feasibility_rank",
        ],
        ascending=[True, False, False, False, False, False, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    output["overall_rank_fit_first"] = np.arange(1, len(output) + 1)
    output["ranking_rule"] = (
        "LOO inversion RMSE ascending; R2 descending; sensitivity ranked only within the same protocol; "
        "then monotonicity, sign consistency, denominator, gamma2 robustness, feasibility"
    )
    return output


def build_recommendations(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, subset in ranking.groupby("ranking_scope", sort=False):
        ordered = subset.sort_values("overall_rank_fit_first")
        primary = ordered.iloc[0].copy()
        primary["recommendation_role"] = "primary_fit_first"
        if scope == "theory_unrestricted_wavelength":
            primary["recommendation_reason"] = (
                "lowest strain-domain LOO RMSE and highest fit R2 among the retained four-protocol summaries"
            )
            secondary = ordered[ordered["protocol_group"].eq("amplitude")].iloc[0].copy()
            secondary["recommendation_role"] = "secondary_scalar_sensitivity"
            secondary["recommendation_reason"] = (
                "best direct scalar-amplitude alternative; sensitivity is reported in m per % strain"
            )
        else:
            primary["recommendation_reason"] = (
                "lowest 632.8-nm strain-domain LOO RMSE; also higher R2 than the close multi-point and feature alternatives"
            )
            secondary = ordered[ordered["protocol_group"].eq("multipoint")].iloc[0].copy()
            secondary["recommendation_role"] = "secondary_multivariate_validation"
            secondary["recommendation_reason"] = (
                "closest multi-point alternative for redundancy, but its LOO RMSE and prediction R2 are inferior at 632.8 nm"
            )
        rows.extend([primary, secondary])
    return pd.DataFrame(rows).reset_index(drop=True)


def _protocol_description(protocol: str) -> str:
    return {
        "amplitude": "fixed lambda, theta_i, and gamma_2; regress one Y_weak amplitude against strain",
        "feature": "scan theta_i and regress a zero, extremum, maximum-slope, or threshold angle against strain",
        "ratio": "combine two theta, lambda, or gamma_2 measurements using D=Y1-Y2 or R=(Y1-Y2)/(Y1+Y2)",
        "multipoint": "fit strain from a vector of Y_weak values using ridge regression and leave-one-strain-out validation",
    }[protocol]


def plot_ranking(ranking: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 600,
        }
    )
    colors = {
        "amplitude": "#0072B2",
        "feature": "#E69F00",
        "ratio": "#009E73",
        "multipoint": "#CC79A7",
    }
    scopes = ["theory_unrestricted_wavelength", "632.8_nm_experiment_compatible"]
    titles = ["Theory ranking: unrestricted wavelength", "Experiment-compatible ranking: 632.8 nm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), constrained_layout=True)
    for ax, scope, title in zip(axes, scopes, titles):
        subset = ranking[ranking["ranking_scope"].eq(scope)].sort_values("overall_rank_fit_first", ascending=False)
        labels = [f"{row.protocol_label}, {row.axis}-axis" for row in subset.itertuples()]
        positions = np.arange(len(subset))
        values = subset["fit_rmse_percent"].to_numpy(float)
        bar_colors = [colors[item] for item in subset["protocol_group"]]
        ax.barh(positions, values, color=bar_colors, alpha=0.86)
        ax.set_yticks(positions, labels)
        ax.set_xlabel("Inversion LOO RMSE (% strain; lower is better)")
        ax.set_title(title)
        for position, (_, row) in enumerate(subset.iterrows()):
            ax.text(
                row["fit_rmse_percent"] + 0.015 * max(values),
                position,
                f"R²={row['fit_r2']:.3f}",
                va="center",
                fontsize=7,
            )
        ax.set_xlim(0.0, max(values) * 1.30)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURE_OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_text_outputs(ranking: pd.DataFrame, recommendation: pd.DataFrame) -> None:
    theory = recommendation[
        recommendation["ranking_scope"].eq("theory_unrestricted_wavelength")
        & recommendation["recommendation_role"].eq("primary_fit_first")
    ].iloc[0]
    theory_secondary = recommendation[
        recommendation["ranking_scope"].eq("theory_unrestricted_wavelength")
        & recommendation["recommendation_role"].eq("secondary_scalar_sensitivity")
    ].iloc[0]
    experiment = recommendation[
        recommendation["ranking_scope"].eq("632.8_nm_experiment_compatible")
        & recommendation["recommendation_role"].eq("primary_fit_first")
    ].iloc[0]
    experiment_multi = recommendation[
        recommendation["ranking_scope"].eq("632.8_nm_experiment_compatible")
        & recommendation["recommendation_role"].eq("secondary_multivariate_validation")
    ].iloc[0]

    theory_coordinates = str(theory.get("feature_coordinates", ""))
    report = f"""# Final fit-first recommendation for weak-measurement strain readout

## Ranking rule

This re-ranking uses only the existing summary CSV files; no optical-conductivity, Fresnel, PSHE, or weak-readout quantity was recalculated. The lexicographic order is: (1) strain-domain inversion LOO RMSE ascending, (2) prediction R² or scalar-fit R² descending, (3) sensitivity descending only within the same protocol and unit, then (4) monotonicity, sign consistency, normalized weak denominator, gamma_2-error robustness, and experimental feasibility. Sensitivities with units of m/% strain, deg/% strain, and dimensionless readout/% strain are not compared numerically across protocol families.

## Four protocol families

1. **Amplitude method:** {_protocol_description('amplitude')}.
2. **Feature-angle method:** {_protocol_description('feature')}.
3. **Ratio/differential method:** {_protocol_description('ratio')}.
4. **Multi-point fingerprint fitting:** {_protocol_description('multipoint')}.

## Theory recommendation

The fit-first theory recommendation is `{theory['protocol_id']}`: a-axis, lambda={theory['lambda_nm']:.1f} nm, gamma_2={theory['gamma_2_deg']:.1f} deg, theta fingerprint {{{theory_coordinates}}} deg. Its inversion LOO RMSE is {theory['fit_rmse_percent']:.6f}% strain and prediction R² is {theory['fit_r2']:.6f}. Multi-point fitting ranks first because the five weak-readout coordinates jointly constrain strain and reduce dependence on any single local amplitude; leave-one-strain-out prediction directly tests inversion rather than only in-sample line fitting.

The secondary scalar option is `{theory_secondary['protocol_id']}` with inversion LOO RMSE {theory_secondary['fit_rmse_percent']:.6f}% strain, R²={theory_secondary['fit_r2']:.6f}, and amplitude sensitivity {theory_secondary['sensitivity_value']:.6e} {theory_secondary['sensitivity_unit']}. This sensitivity is used only after fit quality and only against other amplitude candidates.

## 632.8 nm experiment-compatible recommendation

The existing amplitude protocol remains the fit-first 632.8-nm recommendation: `{experiment['protocol_id']}` (a-axis, theta_i={experiment['theta_i_deg']:.1f} deg, gamma_2={experiment['gamma_2_deg']:.1f} deg). It has inversion LOO RMSE {experiment['fit_rmse_percent']:.6f}% strain and R²={experiment['fit_r2']:.6f}. The multi-point alternative `{experiment_multi['protocol_id']}` has RMSE {experiment_multi['fit_rmse_percent']:.6f}% strain and prediction R²={experiment_multi['fit_r2']:.6f}; therefore it does not displace the amplitude protocol under the requested fit-first rule. The amplitude sensitivity is {experiment['sensitivity_value']:.6e} {experiment['sensitivity_unit']}, and its inversion formula is `{experiment['inversion_formula']}`.

## Role of sensitivity and figure placement

Sensitivity is a secondary criterion: it resolves choices only after LOO RMSE and R², and only within protocols sharing the same readout unit. This prevents a deg/% feature sensitivity or a dimensionless ratio sensitivity from being treated as numerically larger than an m/% displacement sensitivity.

For the main text, use `figures/strain_readout_final_ranking_fit_first.png`, the multi-point prediction example for the unrestricted theory result, and the 632.8-nm amplitude calibration example. Place the complete four-protocol candidate tables, feature-angle alternatives, ratio/differential examples, theta-lambda-gamma_2 screening map, denominator checks, and gamma_2-error robustness results in the supplementary material.

All recommendations remain conditional on the fixed workflow settings and the normalized denominator surrogate; this re-ranking does not modify the archived Fresnel alignment or intrinsic PSHE results.
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    manuscript = f"""# Manuscript text: fit-first strain-readout recommendation

We ranked the four weak-measurement strain-readout protocols by inversion accuracy before sensitivity. The primary criteria were the strain-domain leave-one-out root-mean-square error and the corresponding prediction or regression R². Sensitivity was used only as a secondary criterion within protocols sharing the same physical unit. Under the unrestricted visible-wavelength scan, the a-axis multi-point fingerprint at {theory['lambda_nm']:.1f} nm, gamma_2={theory['gamma_2_deg']:.1f}°, and theta_i={{{theory_coordinates}}}° yielded the smallest inversion error ({theory['fit_rmse_percent']:.6f}% strain; R²={theory['fit_r2']:.6f}). At the experiment-compatible wavelength of 632.8 nm, the fixed-amplitude protocol at theta_i={experiment['theta_i_deg']:.1f}° and gamma_2={experiment['gamma_2_deg']:.1f}° remained the preferred calibration ({experiment['fit_rmse_percent']:.6f}% strain; R²={experiment['fit_r2']:.6f}). Multi-point fitting was favored for the unrestricted theoretical analysis because the joint weak-readout fingerprint constrains strain across several incident angles, whereas the 632.8-nm amplitude protocol retained a slightly lower leave-one-out error and a simpler single-point measurement geometry.
"""
    MANUSCRIPT_OUTPUT.write_text(manuscript, encoding="utf-8")


def main() -> None:
    for directory in (TABLES, FIGURES, REPORTS, MANUSCRIPT, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    theory = normalize_summary(THEORY_SOURCE, "theory_unrestricted_wavelength")
    experiment = normalize_summary(EXPERIMENT_SOURCE, "632.8_nm_experiment_compatible")
    ranking = pd.concat([theory, experiment], ignore_index=True)
    recommendation = build_recommendations(ranking)
    ranking.to_csv(RANKING_OUTPUT, index=False, encoding="utf-8-sig")
    recommendation.to_csv(RECOMMENDATION_OUTPUT, index=False, encoding="utf-8-sig")
    plot_ranking(ranking)
    write_text_outputs(ranking, recommendation)
    print(f"wrote {RANKING_OUTPUT}")
    print(f"wrote {RECOMMENDATION_OUTPUT}")
    print(f"wrote {FIGURE_OUTPUT}")
    print(f"wrote {REPORT_OUTPUT}")
    print(f"wrote {MANUSCRIPT_OUTPUT}")


if __name__ == "__main__":
    main()
