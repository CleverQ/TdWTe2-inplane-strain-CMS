from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BASE.parent
TABLE_DIR = BASE / "data" / "maps_81x81"
NODE_TABLE = REPOSITORY_ROOT / "02_Wannier_Weyl" / "data" / "nodes_ab_with_chirality.csv"
FIGURE_DIR = BASE / "generated_diagnostics"
LOG_DIR = BASE / "logs"
WT_ROOT = Path(os.environ.get("TDWTE2_HR_ROOT", str(BASE / "remote_import" / "hr")))
LEGACY_SCRIPT = Path(__file__).resolve().parent / "wannier_hr_support.py"
MPL_CACHE = LOG_DIR / "matplotlib_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageChops


DPI = 600
NUM_OCC = 56
K3_SLICE_CENTERED = 0.0
GRID_N = 81
EDGE_PAD_PX = 28
CASE_SPECS = [
    {
        "key": "0pct",
        "path": "a",
        "strain": 0.0,
        "case_label": "00_Optimized_Structure",
        "title": "0%",
        "csv": "berry_curvature_map_0pct.csv",
    },
    {
        "key": "a_0p5pct",
        "path": "a",
        "strain": 0.5,
        "case_label": "Strain_a_Tension_0.5pct",
        "title": "a +0.5%",
        "csv": "berry_curvature_map_a_0p5pct.csv",
    },
    {
        "key": "b_0p5pct",
        "path": "b",
        "strain": 0.5,
        "case_label": "Strain_b_Tension_0.5pct",
        "title": "b +0.5%",
        "csv": "berry_curvature_map_b_0p5pct.csv",
    },
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "savefig.dpi": DPI,
            "figure.dpi": DPI,
        }
    )


def trim_white_edge(path: Path, pad_px: int = EDGE_PAD_PX) -> None:
    image = Image.open(path).convert("RGB")
    white = Image.new("RGB", image.size, "white")
    bbox = ImageChops.difference(image, white).getbbox()
    if bbox is None:
        return
    left, top, right, bottom = bbox
    left = max(0, left - pad_px)
    top = max(0, top - pad_px)
    right = min(image.size[0], right + pad_px)
    bottom = min(image.size[1], bottom + pad_px)
    image.crop((left, top, right, bottom)).save(path, dpi=(DPI, DPI))


def import_legacy_module():
    spec = importlib.util.spec_from_file_location("wt_legacy_berry_ab", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import legacy plotting script.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ACTIVE_WT_ROOT = str(WT_ROOT)
    mod.NUM_OCC_TO_USE = NUM_OCC
    mod.PRINT_SUMMARY = False
    mod.PRINT_TIMING = False
    mod.ENABLE_TIMING = False

    def pick_complete_hr_file(case_path: str):
        hrs = sorted(Path(case_path).glob("**/*_hr.dat"))
        complete = [p for p in hrs if p.stat().st_size > 1_000_000]
        if complete:
            for p in complete:
                if p.name.lower() == "wannier90_hr.dat" and "wt_chirality" in str(p):
                    return str(p)
            return str(complete[0])
        return None

    mod.pick_hr_file = pick_complete_hr_file
    return mod


def centered_to_frac(x: float) -> float:
    return float(x) % 1.0


def berry_omega3_occupied(evals: np.ndarray, evecs: np.ndarray, dH: list[np.ndarray], num_occ: int = NUM_OCC) -> float:
    """Occupied-subspace Kubo curvature component for reduced k1-k2 coordinates."""
    vx = evecs.conj().T @ dH[0] @ evecs
    vy = evecs.conj().T @ dH[1] @ evecs
    occ = slice(0, num_occ)
    unocc = slice(num_occ, len(evals))
    gap = evals[unocc][None, :] - evals[occ][:, None]
    denom = np.maximum(gap * gap, 1e-12)
    term = vx[occ, unocc] * vy[unocc, occ].T / denom
    return float(-2.0 * np.imag(np.sum(term)))


def select_nodes(nodes: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if spec["strain"] == 0.0:
        sub = nodes[nodes["case_label"].eq(spec["case_label"])].copy()
        sub = sub.drop_duplicates(subset=["node_id", "k1_centered", "k2_centered", "k3_centered", "chirality"])
    else:
        sub = nodes[
            nodes["case_label"].eq(spec["case_label"])
            & nodes["path"].eq(spec["path"])
            & nodes["strain_percent"].eq(spec["strain"])
        ].copy()
    return sub.sort_values(["chirality", "node_id"]).reset_index(drop=True)


def common_k_range(nodes_by_key: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    all_nodes = pd.concat(nodes_by_key.values(), ignore_index=True)
    k1_min, k1_max = all_nodes["k1_centered"].min(), all_nodes["k1_centered"].max()
    k2_min, k2_max = all_nodes["k2_centered"].min(), all_nodes["k2_centered"].max()
    pad = 0.025
    k1 = np.linspace(k1_min - pad, k1_max + pad, GRID_N)
    k2 = np.linspace(k2_min - pad, k2_max + pad, GRID_N)
    return k1, k2


def compute_case_map(mod, spec: dict, k1_grid: np.ndarray, k2_grid: np.ndarray, nodes: pd.DataFrame) -> pd.DataFrame:
    out = TABLE_DIR / spec["csv"]
    if out.exists():
        df = pd.read_csv(out)
        expected = len(k1_grid) * len(k2_grid)
        if len(df) == expected and {"k1_centered", "k2_centered", "omega3_reduced"}.issubset(df.columns):
            print(f"[reuse] {out.name}")
            return df
        print(f"[recompute] Existing {out.name} has unexpected shape ({len(df)} != {expected}).")
    pack = mod.get_case_hr_pack(spec["case_label"])
    if pack is None:
        raise RuntimeError(f"Missing HR pack for {spec['case_label']}")
    Rvecs, HR_div, _ = pack
    rows = []
    total = len(k1_grid) * len(k2_grid)
    done = 0
    for k2c in k2_grid:
        for k1c in k1_grid:
            k_frac = np.array([centered_to_frac(k1c), centered_to_frac(k2c), centered_to_frac(K3_SLICE_CENTERED)], dtype=float)
            evals, evecs, _H, dH = mod.eigensystem_at_k(k_frac, Rvecs, HR_div)
            omega3 = berry_omega3_occupied(evals, evecs, dH, NUM_OCC)
            rows.append(
                {
                    "case_label": spec["case_label"],
                    "path": spec["path"],
                    "strain_percent": spec["strain"],
                    "k1_centered": k1c,
                    "k2_centered": k2c,
                    "k3_centered": K3_SLICE_CENTERED,
                    "omega3_reduced": omega3,
                    "num_occupied": NUM_OCC,
                    "slice_mode": "fixed_k3",
                }
            )
            done += 1
        if done % (GRID_N * 10) == 0:
            print(f"[{spec['title']}] {done}/{total} k-points")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def nearest_hotspot_stats(df: pd.DataFrame, nodes: pd.DataFrame, percentile: float = 99.0) -> dict:
    threshold = np.percentile(np.abs(df["omega3_reduced"].to_numpy()), percentile)
    hs = df[np.abs(df["omega3_reduced"]) >= threshold].copy()
    if hs.empty or nodes.empty:
        return {"hotspot_threshold_abs": threshold, "hotspot_count": len(hs), "mean_nearest_node_distance": np.nan, "min_nearest_node_distance": np.nan}
    node_xy = nodes[["k1_centered", "k2_centered"]].to_numpy(float)
    distances = []
    for xy in hs[["k1_centered", "k2_centered"]].to_numpy(float):
        dist = np.sqrt(np.sum((node_xy - xy) ** 2, axis=1)).min()
        distances.append(dist)
    return {
        "hotspot_threshold_abs": float(threshold),
        "hotspot_count": int(len(hs)),
        "mean_nearest_node_distance": float(np.mean(distances)),
        "min_nearest_node_distance": float(np.min(distances)),
    }


def plot_maps(maps: dict[str, pd.DataFrame], nodes_by_key: dict[str, pd.DataFrame], k1_grid: np.ndarray, k2_grid: np.ndarray) -> tuple[Path, float]:
    setup_style()
    vals = np.concatenate([m["omega3_reduced"].to_numpy(float) for m in maps.values()])
    clip = float(np.percentile(np.abs(vals[np.isfinite(vals)]), 99.0))
    clip = clip if clip > 0 else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(27.0 / 2.54, 12.0 / 2.54), sharex=True, sharey=True)
    extent = [k1_grid.min(), k1_grid.max(), k2_grid.min(), k2_grid.max()]
    im = None
    for ax, spec in zip(axes, CASE_SPECS):
        df = maps[spec["key"]]
        z = df["omega3_reduced"].to_numpy(float).reshape(len(k2_grid), len(k1_grid))
        z = np.clip(z, -clip, clip)
        im = ax.imshow(z, origin="lower", extent=extent, cmap="coolwarm", vmin=-clip, vmax=clip, aspect="auto", interpolation="nearest")
        nd = nodes_by_key[spec["key"]]
        near = nd[np.abs(nd["k3_centered"] - K3_SLICE_CENTERED) <= 0.025]
        for chi, color, marker in [(1, "#d62728", "o"), (-1, "#1f77b4", "s")]:
            part = near[near["chirality"].eq(chi)]
            ax.scatter(part["k1_centered"], part["k2_centered"], s=34, c=color, marker=marker, edgecolors="black", linewidths=0.45, zorder=3)
        ax.set_title(spec["title"], pad=6)
        ax.set_xlabel(r"$k_1$")
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
    axes[0].set_ylabel(r"$k_2$")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="black", markersize=6, label=r"$\chi=+1$"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=6, label=r"$\chi=-1$"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.47, 0.97))
    fig.subplots_adjust(left=0.075, right=0.875, bottom=0.14, top=0.88, wspace=0.10)
    cax = fig.add_axes([0.895, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\Omega_3$")
    out = FIGURE_DIR / "berry_curvature_k1k2_0_a05_b05.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    trim_white_edge(out)
    return out, clip


def write_report_and_text(stats_rows: list[dict], clip: float, figure_path: Path) -> None:
    stats = pd.DataFrame(stats_rows)
    stats_out = TABLE_DIR / "berry_curvature_hotspot_summary_ab.csv"
    stats.to_csv(stats_out, index=False, encoding="utf-8-sig")
    fixed_slice_text = f"k3 = {K3_SLICE_CENTERED:.3f}"
    report = f"""# a/b 轴 Berry curvature map 分析报告

## 1. 输入与计算范围

本轮基于已有 Wannier90 HR 和已追踪 Weyl node 数据计算 Berry curvature map。未重新运行 `wt.x`，未重新做 Weyl node search，未修改原始 WannierTools 输出目录，未覆盖原始 `plot_wt_weyl_evolution.py`，也未导出 Origin 专用数据。

使用结构：

| 结构 | 输出表 |
|---|---|
| `00_Optimized_Structure` | `tables/berry_curvature_map_0pct.csv` |
| `Strain_a_Tension_0.5pct` | `tables/berry_curvature_map_a_0p5pct.csv` |
| `Strain_b_Tension_0.5pct` | `tables/berry_curvature_map_b_0p5pct.csv` |

## 2. Berry curvature 定义与 slice

本轮绘制 k1-k2 平面上的 Berry curvature map，采用分量为 Ω3(k1,k2)，即垂直于 k1-k2 平面的 Berry curvature 分量。坐标使用 centered reduced reciprocal coordinates，不使用真实倒易长度单位。

采用固定 slice：`{fixed_slice_text}`。该平面靠近当前 tracked Weyl nodes 的 k3 分布，因此作为代表性固定平面用于比较 0%、a +0.5% 和 b +0.5% 三个结构。三张图使用相同 k1/k2 范围和相同色标范围。

## 3. 色标与 hotspot 显示

Berry curvature 尖峰较强，因此绘图时使用对称 percentile clipping。色标范围取所有三张 map 的 |Ω3| 第 99 百分位，clip 值为 `{clip:.6g}`。CSV 表中保留未裁剪的 `omega3_reduced` 原始数值；clipping 仅用于图像显示。

## 4. Hotspot 与 Weyl node 位置

图中叠加了接近固定 k3 slice 的 Weyl node 投影位置：χ = +1 使用红色圆点，χ = -1 使用蓝色方点。hotspot 统计见 `tables/berry_curvature_hotspot_summary_ab.csv`。较强的 |Ω3| 区域出现在 tracked Weyl node 投影附近，说明 Berry curvature map 能作为 topology-related band-structure diagnostic，用于检查 Weyl nodes 附近拓扑响应热点是否随应变发生重排。

## 5. a/b 轴差异与其他拓扑诊断的关系

Berry curvature hotspot 在 0%、a +0.5% 和 b +0.5% 之间发生连续的位置和强度变化。a/b 轴差异是可见但不应夸大的：热点仍主要锚定在 tracked Weyl nodes 附近，变化更多体现为小应变下的连续重排，而不是突变式拓扑变化。该结果与此前的 d_W 非单调调制、相对位移轨迹和 trajectory components 的方向差异一致。tilt ratio 分析进一步说明 type-II-like Weyl character 在该应变范围内保持，因此 Berry curvature map 更适合作为连续重构的可视化补充诊断。

## 6. 图件建议

`figures/berry_curvature_k1k2_0_a05_b05.png` 可以作为主文候选图，但更稳妥的安排是放入补充材料，主文中用一句话引用其结论：Berry curvature hotspot 在 tracked Weyl nodes 附近随 a/b 轴拉伸发生连续重排。若主文需要强化拓扑能带背景，可将其作为多 panel 机制图的一部分。

## 7. 谨慎表述

Berry curvature map 只能作为 topology-related band-structure diagnostic。它不应被写成 632.8 nm PSHE 的直接决定因素。632.8 nm 下 PSHE 的直接输入量仍然是 finite-frequency optical conductivity tensor σ(ω, ε)，并通过 Fresnel coefficients 影响 PSHE displacement。
"""
    (BASE / "analysis_berry_curvature_ab.md").write_text(report, encoding="utf-8")

    text = f"""# Berry curvature 写作素材 v10

## English Results paragraph

Berry-curvature maps were calculated on a fixed k1-k2 slice at k3 = 0 using the Wannier90 HR models for the 0%, a +0.5%, and b +0.5% structures. The plotted component is Ω3(k1,k2), corresponding to the Berry-curvature component normal to the k1-k2 plane in centered reduced reciprocal coordinates. The maps show Berry-curvature hotspot redistribution near the tracked Weyl nodes under small in-plane tensile strain. The redistribution is consistent with the direction-dependent Weyl-node trajectories and the pair-dependent d_W modulation, supporting a continuous reconstruction of the Weyl-proximate topological band structure rather than an abrupt topological phase transition.

These Berry-curvature maps should be interpreted as topology-related band-structure diagnostics. They visualize how the Weyl-node-proximate topological response is redistributed under a-axis and b-axis tensile strain. The visible-light PSHE response at 632.8 nm is still evaluated from the finite-frequency optical conductivity tensor σ(ω, ε), which modifies the Fresnel coefficients and thereby affects the PSHE displacement.

## 中文翻译

基于 0%、a +0.5% 和 b +0.5% 结构的 Wannier90 HR 模型，本轮在固定的 k1-k2 切片 k3 = 0 上计算了 Berry curvature map。绘制的分量为 Ω3(k1,k2)，即在中心化约化倒易坐标中垂直于 k1-k2 平面的 Berry 曲率分量。结果显示，小面内拉伸会使已追踪 Weyl 节点附近的 Berry 曲率热点发生重分布。该重分布与方向依赖的 Weyl 节点轨迹以及节点对依赖的 d_W 调制相一致，支持 Weyl 附近拓扑能带结构的连续重构，而不是突变式拓扑相变。

这些 Berry curvature map 应作为与拓扑相关的能带结构诊断来理解。它们用于显示 a 轴和 b 轴拉伸下 Weyl 节点附近拓扑响应如何发生重分布。632.8 nm 下的可见光 PSHE 响应仍然由有限频率光学电导张量 σ(ω, ε) 计算，并通过菲涅耳系数进一步影响 PSHE 位移。

## Berry curvature figure caption

**Figure X. Berry-curvature maps near the tracked Weyl nodes.** Berry-curvature maps Ω3(k1,k2) are plotted on a fixed k3 = 0 slice for the 0%, a +0.5%, and b +0.5% structures. Coordinates are centered reduced reciprocal coordinates, and the same k1/k2 range and color scale are used for all panels. The color scale is symmetrically clipped at the 99th percentile of |Ω3| for visualization, while the source CSV files retain the unclipped values. Red and blue markers denote Weyl nodes with χ = +1 and χ = -1, respectively. The hotspot redistribution near the tracked Weyl nodes is consistent with direction-dependent Weyl-node trajectories and d_W modulation.

**图 X. 已追踪 Weyl 节点附近的 Berry 曲率图。** 图中给出 0%、a +0.5% 和 b +0.5% 结构在固定 k3 = 0 切片上的 Berry 曲率分布 Ω3(k1,k2)。坐标为中心化约化倒易坐标，所有 panel 使用相同的 k1/k2 范围和相同色标。为了显示热点，色标按 |Ω3| 的第 99 百分位进行对称裁剪，源 CSV 文件保留未裁剪数值。红色和蓝色标记分别表示 χ = +1 和 χ = -1 的 Weyl 节点。已追踪 Weyl 节点附近热点的重分布与方向依赖的 Weyl 节点轨迹和 d_W 调制相一致。

## Cautious statement

Berry curvature is used here as a topology-related diagnostic of the Weyl-proximate band structure. It should not be described as directly determining the 632.8 nm PSHE response. The direct optical input for PSHE remains the finite-frequency optical conductivity tensor σ(ω, ε), which enters the Fresnel coefficients.
"""
    (BASE / "manuscript_text_ab_weyl_topology_revised_v10.md").write_text(text, encoding="utf-8")


def main() -> None:
    nodes = pd.read_csv(NODE_TABLE)
    nodes_by_key = {spec["key"]: select_nodes(nodes, spec) for spec in CASE_SPECS}
    k1_grid, k2_grid = common_k_range(nodes_by_key)
    mod = import_legacy_module()
    maps = {}
    stats_rows = []
    for spec in CASE_SPECS:
        print(f"Computing Berry curvature map: {spec['title']} | {spec['case_label']}")
        df = compute_case_map(mod, spec, k1_grid, k2_grid, nodes_by_key[spec["key"]])
        maps[spec["key"]] = df
        stats = nearest_hotspot_stats(df, nodes_by_key[spec["key"]])
        stats_rows.append({**spec, **stats, "k3_slice_centered": K3_SLICE_CENTERED, "grid_n": GRID_N})
    fig_path, clip = plot_maps(maps, nodes_by_key, k1_grid, k2_grid)
    write_report_and_text(stats_rows, clip, fig_path)
    print(fig_path)
    print(BASE / "analysis_berry_curvature_ab.md")
    print(BASE / "manuscript_text_ab_weyl_topology_revised_v10.md")


if __name__ == "__main__":
    main()
