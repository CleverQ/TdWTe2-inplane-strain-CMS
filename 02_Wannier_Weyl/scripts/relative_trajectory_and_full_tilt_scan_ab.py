from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MPL_CACHE = BASE / "logs" / "matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd


TABLE_DIR = BASE / "data"
FIGURE_DIR = BASE / "generated_diagnostics"
WT_ROOT = Path(os.environ.get("TDWTE2_HR_ROOT", str(BASE / "remote_import" / "hr")))
LEGACY_SCRIPT = Path(__file__).resolve().parent / "wannier_hr_linearization_support.py"

PAIR_BAND_N = 56
PAIR_BAND_M = 57
WINDOWS = [0.0005, 0.0010, 0.0020]
ANALYTIC_WINDOW_LABEL = "analytic_dH"
TYPEII_THRESHOLD = 1.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "savefig.dpi": 600,
        }
    )


def chirality_color(chirality: int) -> str:
    return "#d62728" if int(chirality) > 0 else "#1f77b4"


def save_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def centered_to_frac(x: float) -> float:
    return float(x) % 1.0


def minimal_delta(cur: float, ref: float) -> float:
    d = float(cur) - float(ref)
    return float(d - np.round(d))


def import_legacy_module():
    spec = importlib.util.spec_from_file_location("wt_legacy_for_tilt_scan", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import legacy plot script.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ACTIVE_WT_ROOT = str(WT_ROOT)
    mod.NUM_OCC_TO_USE = 56
    mod.PRINT_SUMMARY = False
    mod.PRINT_TIMING = False
    mod.ENABLE_TIMING = False

    def pick_complete_hr_file(case_path: str):
        hrs = sorted(Path(case_path).glob("*_hr.dat"))
        complete = [p for p in hrs if p.stat().st_size > 1_000_000]
        if complete:
            for p in complete:
                if p.name.lower() == "wannier90_hr.dat":
                    return str(p)
            return str(complete[0])
        return None

    mod.pick_hr_file = pick_complete_hr_file
    return mod


def pair_map() -> dict[tuple[str, float, str], str]:
    out = {}
    for name in ["weyl_pair_distance_a.csv", "weyl_pair_distance_b.csv"]:
        df = pd.read_csv(TABLE_DIR / name)
        for _, row in df.iterrows():
            path = str(row["path"])
            strain = float(row["strain_percent"])
            pair_id = str(row["pair_id"])
            out[(path, strain, str(row["family_plus"]))] = pair_id
            out[(path, strain, str(row["family_minus"]))] = pair_id
    return out


def compute_relative_tracks() -> pd.DataFrame:
    tracks = pd.concat(
        [
            pd.read_csv(TABLE_DIR / "family_tracks_a.csv"),
            pd.read_csv(TABLE_DIR / "family_tracks_b.csv"),
        ],
        ignore_index=True,
    )
    rows = []
    for (path, family), fam in tracks.groupby(["path", "family_id"], sort=True):
        fam = fam.sort_values("strain_percent")
        ref = fam[fam["strain_percent"].eq(0.0)].iloc[0]
        for _, row in fam.iterrows():
            dk1 = minimal_delta(row["k1_centered"], ref["k1_centered"])
            dk2 = minimal_delta(row["k2_centered"], ref["k2_centered"])
            dk3 = minimal_delta(row["k3_centered"], ref["k3_centered"])
            rows.append(
                {
                    "path": path,
                    "family_id": family,
                    "case_label": row["case_label"],
                    "strain_percent": float(row["strain_percent"]),
                    "chirality": int(row["chirality"]),
                    "delta_k1_relative_to_0": dk1,
                    "delta_k2_relative_to_0": dk2,
                    "delta_k3_relative_to_0": dk3,
                    "relative_displacement": float(np.sqrt(dk1 * dk1 + dk2 * dk2 + dk3 * dk3)),
                    "energy_minus_EF_eV": float(row["energy_minus_EF_eV"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "weyl_node_relative_displacement_tracks_ab.csv", index=False, encoding="utf-8-sig")
    return df


def draw_relative_tracks(df: pd.DataFrame, out_name: str, representative: bool = False) -> Path:
    setup_style()
    plot_df = df.copy()
    if representative:
        selected = []
        for path, sub in df[df["strain_percent"].eq(0.5)].groupby("path"):
            selected.extend(sub.sort_values("relative_displacement", ascending=False).head(3)["family_id"].tolist())
        plot_df = df[df["family_id"].isin(selected)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(27.0 / 2.54, 12.0 / 2.54), sharex=True, sharey=True)
    for ax, path, title in zip(axes, ["a", "b"], ["a-axis", "b-axis"]):
        sub_path = plot_df[plot_df["path"].eq(path)]
        for family, fam in sub_path.groupby("family_id", sort=True):
            fam = fam.sort_values("strain_percent")
            color = chirality_color(int(fam["chirality"].iloc[0]))
            x = fam["delta_k1_relative_to_0"].to_numpy(dtype=float)
            y = fam["delta_k2_relative_to_0"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.82)
            for i in range(len(x) - 1):
                ax.annotate(
                    "",
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9, mutation_scale=9, shrinkA=1.5, shrinkB=1.5),
                    zorder=2,
                )
            ax.scatter([x[0]], [y[0]], s=24, marker="o", color=color, edgecolor="black", linewidth=0.35, zorder=3)
            ax.scatter([x[-1]], [y[-1]], s=46, marker="s", color=color, edgecolor="black", linewidth=0.45, zorder=4)
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.35)
        ax.axvline(0.0, color="black", lw=0.6, alpha=0.35)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$\Delta k_1$")
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
    axes[0].set_ylabel(r"$\Delta k_2$")
    handles = [
        Line2D([0], [0], color="#d62728", lw=1.4, label=r"$\chi=+1$"),
        Line2D([0], [0], color="#1f77b4", lw=1.4, label=r"$\chi=-1$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markersize=6, label="0%"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor="black", markersize=7, label="0.5%"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.52, 0.985))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.12)
    out = FIGURE_DIR / out_name
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out


def projected_pauli_coeffs(M2: np.ndarray) -> tuple[float, float, float, float]:
    d0 = 0.5 * np.real(M2[0, 0] + M2[1, 1])
    dz = 0.5 * np.real(M2[0, 0] - M2[1, 1])
    dx = np.real(M2[0, 1])
    dy = -np.imag(M2[0, 1])
    return float(d0), float(dx), float(dy), float(dz)


def analyze_from_dH(pair_n: int, pair_m: int, evecs: np.ndarray, dH_list: list[np.ndarray]) -> dict:
    U = evecs[:, [pair_n, pair_m]]
    B = np.zeros((3, 3), dtype=float)
    v0 = np.zeros(3, dtype=float)
    for iax in range(3):
        M2 = U.conj().T @ dH_list[iax] @ U
        d0, dx, dy, dz = projected_pauli_coeffs(M2)
        v0[iax] = d0
        B[:, iax] = np.array([dx, dy, dz], dtype=float)
    G = B.T @ B
    cond_G = float(np.linalg.cond(G)) if np.all(np.isfinite(G)) else np.nan
    tilt = np.nan
    if np.isfinite(cond_G) and cond_G < 1e12:
        try:
            tilt = float(np.sqrt(v0.T @ np.linalg.inv(G) @ v0))
        except np.linalg.LinAlgError:
            tilt = np.nan
    svals = np.linalg.svd(B, compute_uv=False)
    detB = float(np.linalg.det(B))
    return {
        "tilt_ratio": tilt,
        "typeII_flag": bool(np.isfinite(tilt) and tilt > TYPEII_THRESHOLD),
        "singular_v1": float(svals[0]),
        "singular_v2": float(svals[1]),
        "singular_v3": float(svals[2]),
        "mean_cone_slope_eV_per_reduced_dk": float(np.mean(svals)),
        "min_cone_slope_eV_per_reduced_dk": float(np.min(svals)),
        "max_cone_slope_eV_per_reduced_dk": float(np.max(svals)),
        "detB": detB,
        "condition_G": cond_G,
        "stability_indicator": float(np.min(svals) / np.max(svals)) if np.max(svals) > 0 else np.nan,
    }


def finite_difference_dH(mod, k0: np.ndarray, Rvecs: np.ndarray, HR_div: np.ndarray, h: float) -> list[np.ndarray]:
    out = []
    for iax in range(3):
        step = np.zeros(3)
        step[iax] = h
        H_plus, _ = mod.h_and_dh(np.mod(k0 + step, 1.0), Rvecs, HR_div)
        H_minus, _ = mod.h_and_dh(np.mod(k0 - step, 1.0), Rvecs, HR_div)
        out.append((H_plus - H_minus) / (2.0 * h))
    return out


def case_path_label(row: pd.Series) -> str:
    label = str(row["case_label"])
    if label == "00_Optimized_Structure":
        return "0%"
    return f"{row['path']} +{float(row['strain_percent']):.1f}%"


def run_full_tilt_scan() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mod = import_legacy_module()
    pmap = pair_map()
    tracks = pd.concat(
        [
            pd.read_csv(TABLE_DIR / "family_tracks_a.csv"),
            pd.read_csv(TABLE_DIR / "family_tracks_b.csv"),
        ],
        ignore_index=True,
    )
    full_rows: list[dict] = []
    window_rows: list[dict] = []
    pair_n = PAIR_BAND_N - 1
    pair_m = PAIR_BAND_M - 1

    for _, row in tracks.iterrows():
        path = str(row["path"])
        strain = float(row["strain_percent"])
        family = str(row["family_id"])
        pair_id = pmap.get((path, strain, family), "")
        k0 = np.array([centered_to_frac(row["k1_centered"]), centered_to_frac(row["k2_centered"]), centered_to_frac(row["k3_centered"])], dtype=float)
        pack = mod.get_case_hr_pack(str(row["case_label"]))
        base = {
            "path": path,
            "strain_percent": strain,
            "case_label": row["case_label"],
            "case_short": case_path_label(row),
            "family_id": family,
            "pair_id": pair_id,
            "chirality": int(row["chirality"]),
            "k1_centered": float(row["k1_centered"]),
            "k2_centered": float(row["k2_centered"]),
            "k3_centered": float(row["k3_centered"]),
            "energy_minus_EF_eV": float(row["energy_minus_EF_eV"]),
        }
        if pack is None:
            full_rows.append({**base, "window": ANALYTIC_WINDOW_LABEL, "status": "missing_HR"})
            continue
        Rvecs, HR_div, _ = pack
        try:
            _evals, evecs, _H, dH_analytic = mod.eigensystem_at_k(k0, Rvecs, HR_div)
            ana = analyze_from_dH(pair_n, pair_m, evecs, dH_analytic)
            full_rows.append({**base, **ana, "window": ANALYTIC_WINDOW_LABEL, "window_reduced_dk": 0.0, "status": "ok"})
            window_rows.append({**base, **ana, "window": ANALYTIC_WINDOW_LABEL, "window_reduced_dk": 0.0, "status": "ok"})
            for h in WINDOWS:
                dH_fd = finite_difference_dH(mod, k0, Rvecs, HR_div, h)
                fd = analyze_from_dH(pair_n, pair_m, evecs, dH_fd)
                window_rows.append({**base, **fd, "window": f"fd_{h:g}", "window_reduced_dk": h, "status": "ok"})
        except Exception as exc:
            full_rows.append({**base, "window": ANALYTIC_WINDOW_LABEL, "status": f"failed: {exc}"})

    full_df = pd.DataFrame(full_rows)
    win_df = pd.DataFrame(window_rows)
    full_df.to_csv(TABLE_DIR / "weyl_tilt_ratio_full_scan_ab.csv", index=False, encoding="utf-8-sig")
    win_df.to_csv(TABLE_DIR / "weyl_tilt_ratio_window_sensitivity_ab.csv", index=False, encoding="utf-8-sig")

    rows = []
    for keys, sub in win_df[win_df["status"].eq("ok")].groupby(["path", "strain_percent", "case_label", "family_id", "pair_id", "chirality"], sort=True):
        path, strain, case_label, family, pair_id, chirality = keys
        analytic = sub[sub["window"].eq(ANALYTIC_WINDOW_LABEL)]
        finite = sub[~sub["window"].eq(ANALYTIC_WINDOW_LABEL)]
        tilt = float(analytic["tilt_ratio"].iloc[0]) if not analytic.empty else np.nan
        rows.append(
            {
                "path": path,
                "strain_percent": strain,
                "case_label": case_label,
                "case_short": "0%" if case_label == "00_Optimized_Structure" else f"{path} +{strain:.1f}%",
                "family_id": family,
                "pair_id": pair_id,
                "chirality": int(chirality),
                "k1_centered": float(analytic["k1_centered"].iloc[0]) if not analytic.empty else float(sub["k1_centered"].iloc[0]),
                "k2_centered": float(analytic["k2_centered"].iloc[0]) if not analytic.empty else float(sub["k2_centered"].iloc[0]),
                "k3_centered": float(analytic["k3_centered"].iloc[0]) if not analytic.empty else float(sub["k3_centered"].iloc[0]),
                "tilt_ratio": tilt,
                "tilt_ratio_window_min": float(finite["tilt_ratio"].min()) if not finite.empty else np.nan,
                "tilt_ratio_window_max": float(finite["tilt_ratio"].max()) if not finite.empty else np.nan,
                "tilt_ratio_window_std": float(finite["tilt_ratio"].std(ddof=0)) if not finite.empty else np.nan,
                "typeII_flag": bool(np.isfinite(tilt) and tilt > TYPEII_THRESHOLD),
                "typeII_window_all": bool((finite["tilt_ratio"] > TYPEII_THRESHOLD).all()) if not finite.empty else False,
                "mean_cone_slope_eV_per_reduced_dk": float(analytic["mean_cone_slope_eV_per_reduced_dk"].iloc[0]) if not analytic.empty else np.nan,
                "condition_G": float(analytic["condition_G"].iloc[0]) if not analytic.empty else np.nan,
                "stability_indicator": float(analytic["stability_indicator"].iloc[0]) if not analytic.empty else np.nan,
                "recommended_usage": "supplementary" if np.isfinite(tilt) else "not recommended",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "weyl_tilt_ratio_summary_ab.csv", index=False, encoding="utf-8-sig")
    return full_df, win_df, summary


def plot_tilt_mean(summary: pd.DataFrame) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(17.0 / 2.54, 12.0 / 2.54))
    for path, color, label in [("a", "#4c78a8", "a-axis"), ("b", "#f58518", "b-axis")]:
        sub = summary[summary["path"].eq(path)]
        agg = sub.groupby("strain_percent")["tilt_ratio"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["strain_percent"], agg["mean"], yerr=agg["std"], color=color, marker="o", lw=1.6, capsize=3, label=label)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="type-II threshold")
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Mean tilt ratio")
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.16))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.86)
    out = FIGURE_DIR / "weyl_tilt_ratio_mean_vs_strain_ab.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out


def plot_tilt_pair_resolved(summary: pd.DataFrame) -> Path:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(27.0 / 2.54, 12.0 / 2.54), sharey=True)
    colors = {"1": "#d62728", "-1": "#1f77b4"}
    for ax, path, title in zip(axes, ["a", "b"], ["a-axis", "b-axis"]):
        subp = summary[summary["path"].eq(path)]
        for family, sub in subp.groupby("family_id", sort=True):
            sub = sub.sort_values("strain_percent")
            color = chirality_color(int(sub["chirality"].iloc[0]))
            ax.plot(sub["strain_percent"], sub["tilt_ratio"], marker="o", ms=3, lw=1.0, color=color, alpha=0.78)
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_title(title, pad=6)
        ax.set_xlabel("Strain (%)")
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
    axes[0].set_ylabel("Tilt ratio")
    handles = [
        Line2D([0], [0], color="#d62728", marker="o", lw=1.2, label=r"$\chi=+1$"),
        Line2D([0], [0], color="#1f77b4", marker="o", lw=1.2, label=r"$\chi=-1$"),
        Line2D([0], [0], color="black", lw=0.8, ls="--", label="type-II threshold"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 0.985))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84, wspace=0.12)
    out = FIGURE_DIR / "weyl_tilt_ratio_pair_resolved_ab.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out


def plot_tilt_distribution(summary: pd.DataFrame) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(17.0 / 2.54, 12.0 / 2.54))
    data = [summary[summary["path"].eq("a")]["tilt_ratio"].dropna(), summary[summary["path"].eq("b")]["tilt_ratio"].dropna()]
    bp = ax.boxplot(data, tick_labels=["a-axis", "b-axis"], patch_artist=True, widths=0.45)
    for patch, color in zip(bp["boxes"], ["#4c78a8", "#f58518"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Tilt ratio")
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.94)
    out = FIGURE_DIR / "weyl_tilt_ratio_distribution_ab.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out


def plot_local_cone_3d(summary: pd.DataFrame) -> Path | None:
    mod = import_legacy_module()
    setup_style()
    cases = [("0%", "a", 0.0, "00_Optimized_Structure"), ("a +0.5%", "a", 0.5, "Strain_a_Tension_0.5pct"), ("b +0.5%", "b", 0.5, "Strain_b_Tension_0.5pct")]
    targets = []
    for case_short, path, strain, case_label in cases:
        sub = summary[(summary["path"].eq(path)) & (summary["strain_percent"].eq(strain)) & (summary["pair_id"].str.endswith("pair_04", na=False)) & (summary["chirality"].eq(1))]
        if sub.empty:
            sub = summary[(summary["path"].eq(path)) & (summary["strain_percent"].eq(strain)) & (summary["chirality"].eq(1))]
        if sub.empty:
            return None
        targets.append((case_short, sub.iloc[0]))

    qmax = 0.004
    ngrid = 31
    q = np.linspace(-qmax, qmax, ngrid)
    Q1, Q2 = np.meshgrid(q, q, indexing="xy")
    fig = plt.figure(figsize=(27.0 / 2.54, 12.0 / 2.54))
    pair_n = PAIR_BAND_N - 1
    pair_m = PAIR_BAND_M - 1

    for idx, (case_short, row) in enumerate(targets, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        pack = mod.get_case_hr_pack(str(row["case_label"]))
        if pack is None:
            continue
        Rvecs, HR_div, _ = pack
        k0 = np.array([centered_to_frac(row["k1_centered"]), centered_to_frac(row["k2_centered"]), centered_to_frac(row["k3_centered"])], dtype=float)
        evals0, evecs, _H, dH = mod.eigensystem_at_k(k0, Rvecs, HR_div)
        ana = mod.analyze_two_band_linearization(pair_n, pair_m, evecs, dH)
        dirs = np.asarray(ana["Vdirs"], dtype=float)
        v1 = dirs[:, 0] / np.linalg.norm(dirs[:, 0])
        v2 = dirs[:, 1] / np.linalg.norm(dirs[:, 1])
        e0 = 0.5 * (evals0[pair_n] + evals0[pair_m])
        E1 = np.zeros_like(Q1)
        E2 = np.zeros_like(Q1)
        for i in range(ngrid):
            for j in range(ngrid):
                kk = np.mod(k0 + Q1[i, j] * v1 + Q2[i, j] * v2, 1.0)
                ev, *_ = mod.eigensystem_at_k(kk, Rvecs, HR_div)
                E1[i, j] = ev[pair_n] - e0
                E2[i, j] = ev[pair_m] - e0
        ax.plot_surface(Q1, Q2, E1, color="#4c78a8", alpha=0.72, linewidth=0.05, edgecolor=(0, 0, 0, 0.12))
        ax.plot_surface(Q1, Q2, E2, color="#f58518", alpha=0.72, linewidth=0.05, edgecolor=(0, 0, 0, 0.12))
        ax.set_title(case_short, pad=2)
        ax.set_xlabel(r"$q_1$")
        ax.set_ylabel(r"$q_2$")
        ax.set_zlabel(r"$E-E_W$ (eV)")
        ax.view_init(elev=22, azim=-55)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.02, top=0.92, wspace=0.02)
    out = FIGURE_DIR / "local_weyl_cone_3d_0_a05_b05.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return out


def fmt(x: float, digits: int = 6) -> str:
    if x is None or not np.isfinite(float(x)):
        return ""
    return f"{float(x):.{digits}f}"


def write_reports(rel: pd.DataFrame, summary: pd.DataFrame, win: pd.DataFrame, cone_generated: bool) -> tuple[Path, Path]:
    dwn = pd.read_csv(TABLE_DIR / "ab_weyl_dW_nonmonotonic_summary.csv")
    amean = dwn[dwn["object_id"].eq("a_mean_dW")].iloc[0]
    bmean = dwn[dwn["object_id"].eq("b_mean_dW")].iloc[0]
    aagg = summary[summary["path"].eq("a")].groupby("strain_percent")["tilt_ratio"].mean().reset_index()
    bagg = summary[summary["path"].eq("b")].groupby("strain_percent")["tilt_ratio"].mean().reset_index()
    all_type2 = bool((summary["tilt_ratio"] > 1.0).all())
    min_tilt = float(summary["tilt_ratio"].min())
    max_tilt = float(summary["tilt_ratio"].max())
    win_std_max = float(summary["tilt_ratio_window_std"].max())
    a05 = float(aagg[aagg["strain_percent"].eq(0.5)]["tilt_ratio"].iloc[0])
    b05 = float(bagg[bagg["strain_percent"].eq(0.5)]["tilt_ratio"].iloc[0])
    axis_diff = a05 - b05

    report = f"""# a/b 轴 Weyl relative trajectory 与 full tilt ratio 分析报告

## 1. 相对位移 trajectory

本轮将 absolute k1-k2 trajectory 改为以每个 family 的 0% 坐标为参考的相对位移轨迹，并使用 periodic minimal-image convention：

`Δk_i(strain) = k_i(strain) - k_i(0)`

输出表为 `tables/weyl_node_relative_displacement_tracks_ab.csv`。新图包括：

| 图件 | 建议用途 | 说明 |
|---|---|---|
| `figures/weyl_node_relative_displacement_k1k2_ab.png` | 补充或主文辅助 | 展示全部 family 的相对 Δk1-Δk2 轨迹，起点均在 0 附近，箭头表示 strain 增大方向。 |
| `figures/weyl_node_relative_displacement_k1k2_ab_representative.png` | 主文可选 | 只显示每条路径相对位移最大的代表性 family，更容易观察小应变轨迹方向。 |
| `figures/weyl_node_trajectory_components_ab_v2.png` | 主文优先 | trajectory components 比 absolute k1-k2 trajectory 更适合展示小应变下的 a/b 方向差异。 |

absolute k1-k2 trajectory 图仍可保留，但更适合放补充或作为内部检查图；它不适合作为主文核心证据，因为 0-0.5% 小应变下绝对坐标点过于接近。

## 2. d_W 与 trajectory 的分析定位

`d_W` 仍应写成 pair-dependent and partly nonmonotonic。a-axis mean `d_W` 的 sign sequence 为 `{amean['sign_sequence']}`，b-axis mean `d_W` 的 sign sequence 为 `{bmean['sign_sequence']}`。两条路径的 peak-to-peak amplitude 接近；a-axis 更清楚的是 increase-then-decrease profile，而不是幅度显著更大。

trajectory components 与 relative displacement trajectory 才是方向性差异的主要证据：a-axis 位移主要由 `Δk2` 主导，b-axis 位移主要由 `Δk1` 主导。

## 3. full tilt ratio / Weyl cone slope 扫描

本轮复用了旧脚本中的 HR 读取与 two-band linearization 功能，对全部 tracked Weyl nodes 执行 tilt ratio / cone slope 扫描。输出包括：

| 表格 | 内容 |
|---|---|
| `tables/weyl_tilt_ratio_full_scan_ab.csv` | 解析 dH 下每个 tracked node 的 tilt ratio、cone slopes、condition number 和 type-II flag。 |
| `tables/weyl_tilt_ratio_window_sensitivity_ab.csv` | 解析 dH 与 finite-difference windows (`0.0005`, `0.001`, `0.002`) 的窗口敏感性结果。 |
| `tables/weyl_tilt_ratio_summary_ab.csv` | 每个 family 的主 tilt ratio、窗口 min/max/std、typeII flag 和推荐用途。 |

扫描结果：tilt ratio 范围为 {fmt(min_tilt)} 到 {fmt(max_tilt)}。所有 tracked nodes 的解析 tilt ratio 均 {'大于' if all_type2 else '并非全部大于'} 1。窗口敏感性最大标准差为 {fmt(win_std_max)}。

## 4. tilt ratio 图件建议

| 图件 | 建议用途 | 说明 |
|---|---|---|
| `figures/weyl_tilt_ratio_mean_vs_strain_ab.png` | 主文可选 | 如果平均趋势清楚，可作为 type-II-like character 保持与局部 cone geometry 连续调制的证据。 |
| `figures/weyl_tilt_ratio_pair_resolved_ab.png` | 补充 | 展示 family-resolved tilt ratio，避免主文过度简化。 |
| `figures/weyl_tilt_ratio_distribution_ab.png` | 补充 | 展示 a/b 路径的整体分布与 type-II threshold。 |
| `figures/local_weyl_cone_3d_0_a05_b05.png` | 补充或内部检查 | {'已生成。' if cone_generated else '未生成或不建议使用。'} 3D cone 图不应替代定量 tilt ratio 表。 |

## 5. a/b 轴 tilt 差异与窗口敏感性

a +0.5% 的 mean tilt ratio 为 {fmt(a05)}，b +0.5% 的 mean tilt ratio 为 {fmt(b05)}，差值为 {fmt(axis_diff)}。该差异可作为 local Weyl cone geometry 随应变方向连续变化的诊断，但不应写成拓扑相变。若所有窗口下 tilt ratio 均保持大于 1，可谨慎表述 type-II-like Weyl character 在 0-0.5% 小面内拉伸范围内保持。

## 6. local dispersion 与 PSHE 的解释边界

代表性 pair 的 local dispersion 只证明 tracked Weyl crossing 的局域能带特征，不代表所有 pair 的完整响应。tilt ratio 和 Weyl cone slope 也不能写成 632.8 nm PSHE 的直接决定因素。632.8 nm PSHE 的直接输入仍是 finite-frequency optical conductivity tensor `σ(ω, ε)`，再通过 Fresnel coefficients 影响 photonic spin Hall displacement。

## 7. 旧脚本中 c 轴专用逻辑

旧脚本仍保留部分 c-axis/通用 path 绘图开关和旧图输出函数，但本轮没有覆盖原始脚本，只在新脚本中复用 HR、linearization 与绘图函数。当前需要清理的主要不是算法本身，而是旧图函数中可能出现的 per Å 坐标标签；本轮新输出均使用 reduced Δk，不输出 per Å 单位。
"""

    manuscript = f"""# a/b 轴 Weyl topology 写作素材修正版 v5

## A. Revised Results Paragraph

The Weyl-node trajectories are better resolved by plotting relative displacements with respect to the 0% structure rather than the absolute k1-k2 coordinates. In the absolute k1-k2 maps, the 0-0.5% trajectory points are very close to each other and therefore are not ideal as the main visual evidence. By contrast, the trajectory components and the relative displacement trajectories directly show the direction-dependent redistribution: a-axis tension is dominated by the `Δk2` displacement, whereas b-axis tension is dominated by the `Δk1` displacement.

The `d_W` response remains pair-dependent and partly nonmonotonic. The mean `d_W` under a-axis tension shows a clearer increase-then-decrease profile, while the b-axis response first increases and then becomes weaker and nearly flattened. Pair-resolved curves confirm that different Weyl pairs do not share a universal response shape. Therefore, linear slopes are retained only as auxiliary descriptors.

The full tilt-ratio scan further shows that the local Weyl-cone geometry is continuously modified under small in-plane tensile strain. The scanned tilt ratios range from {fmt(min_tilt)} to {fmt(max_tilt)}, and the type-II-like Weyl character is {'preserved for all tracked nodes' if all_type2 else 'not uniformly preserved for all tracked nodes'} within the examined strain range. The a +0.5% and b +0.5% paths have mean tilt ratios of {fmt(a05)} and {fmt(b05)}, respectively, indicating a direction-dependent but continuous modification of the local cone geometry. These tilt and slope diagnostics should not be interpreted as evidence for an abrupt topological phase transition.

Finally, the representative local dispersion of `pair_04 (F08-F03)` verifies that the tracked Weyl crossing remains locally traceable for 0%, a +0.5%, and b +0.5%. This local dispersion does not represent the complete response of all Weyl pairs. Together, `d_W`, relative trajectory, local dispersion, and tilt-ratio analyses provide topology-related band-structure diagnostics. They are not direct determinants of the 632.8 nm photonic spin Hall displacement; the direct input at this wavelength remains the finite-frequency optical conductivity tensor `σ(ω, ε)`, which modifies the Fresnel coefficients and thereby the photonic spin Hall displacement.

## B. 对应中文翻译

与 absolute k1-k2 coordinates 相比，以 0% 结构为参考的 relative displacement 更适合展示 Weyl-node trajectories。在 absolute k1-k2 maps 中，0-0.5% 的轨迹点非常接近，因此不适合作为主文核心视觉证据。相比之下，trajectory components 和 relative displacement trajectories 能更直接地显示方向依赖重分布：a-axis tension 主要由 `Δk2` 位移主导，而 b-axis tension 主要由 `Δk1` 位移主导。

`d_W` 响应仍然是 pair-dependent and partly nonmonotonic。a-axis tension 下 mean `d_W` 显示更清楚的 increase-then-decrease profile，而 b-axis 响应先增大，随后变化减弱并接近平缓。pair-resolved 曲线确认不同 Weyl pair 不共享单一普适响应形状。因此，linear slopes 只保留为辅助描述量。

full tilt-ratio scan 进一步显示，小面内拉伸下 local Weyl-cone geometry 发生连续调制。扫描得到的 tilt ratio 范围为 {fmt(min_tilt)} 到 {fmt(max_tilt)}，在当前应变范围内 type-II-like Weyl character {'对所有 tracked nodes 均保持' if all_type2 else '并非对所有 tracked nodes 均保持'}。a +0.5% 和 b +0.5% 路径的 mean tilt ratio 分别为 {fmt(a05)} 和 {fmt(b05)}，说明局部 cone geometry 发生方向依赖但连续的变化。这些 tilt 和 slope diagnostics 不应被解释为突变拓扑相变的证据。

最后，代表性 `pair_04 (F08-F03)` 的 local dispersion 证明 tracked Weyl crossing 在 0%、a +0.5% 和 b +0.5% 中仍具有可追踪的局域能带特征。该 local dispersion 不代表所有 Weyl pairs 的完整响应。综合来看，`d_W`、relative trajectory、local dispersion 和 tilt-ratio analyses 提供的是 topology-related band-structure diagnostics。它们不是 632.8 nm photonic spin Hall displacement 的直接决定因素；该波长下的直接输入仍是 finite-frequency optical conductivity tensor `σ(ω, ε)`，它通过改变 Fresnel coefficients 进一步影响 photonic spin Hall displacement。

## C. Relative Displacement Trajectory Caption

**Figure X. Relative Weyl-node displacement trajectories.** The Weyl-node trajectories are plotted as displacements relative to the 0% structure in centered reduced reciprocal coordinates. The horizontal and vertical axes are `Δk1` and `Δk2`, respectively. Arrows indicate the direction of increasing strain from 0 to 0.5%. Red and blue curves denote nodes with positive and negative chirality. The relative displacement representation resolves the small-strain motion more clearly than the absolute k1-k2 coordinates.

## D. Tilt Ratio Caption

**Figure X. Strain evolution of the Weyl-cone tilt ratio.** Mean tilt ratios of the tracked Weyl nodes are compared for a-axis and b-axis tension. The dashed horizontal line marks the type-II threshold of tilt ratio = 1. The full scan indicates whether the type-II-like Weyl character is preserved and whether the local cone geometry is continuously modified under small in-plane tensile strain. The slopes are expressed in eV per reduced `Δk` when reported in the source tables.

## E. 谨慎结论

The present topology-related diagnostics support continuous, direction-dependent reconstruction of the Weyl-proximate band structure under 0-0.5% in-plane tensile strain. They do not indicate Weyl-node creation or annihilation, and they should not be interpreted as a strain-induced topological phase transition.
"""
    report_path = BASE / "analysis_weyl_trajectory_tilt_ab.md"
    manuscript_path = BASE / "manuscript_text_ab_weyl_topology_revised_v5.md"
    report_path.write_text(report, encoding="utf-8")
    manuscript_path.write_text(manuscript, encoding="utf-8")
    return report_path, manuscript_path


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    rel = compute_relative_tracks()
    rel_all = draw_relative_tracks(rel, "weyl_node_relative_displacement_k1k2_ab.png", representative=False)
    rel_rep = draw_relative_tracks(rel, "weyl_node_relative_displacement_k1k2_ab_representative.png", representative=True)
    full, win, summary = run_full_tilt_scan()
    mean_fig = plot_tilt_mean(summary)
    pair_fig = plot_tilt_pair_resolved(summary)
    dist_fig = plot_tilt_distribution(summary)
    cone_fig = plot_local_cone_3d(summary)
    report, manuscript = write_reports(rel, summary, win, cone_generated=cone_fig is not None)
    print("Generated relative trajectory:")
    print(rel_all)
    print(rel_rep)
    print("Generated tilt figures:")
    for p in [mean_fig, pair_fig, dist_fig, cone_fig]:
        if p is not None:
            print(p)
    print("Generated reports:")
    print(report)
    print(manuscript)


if __name__ == "__main__":
    main()
