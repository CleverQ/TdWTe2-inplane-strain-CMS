#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_optics_sigma.py —— 提取/汇总 Wannier90-Kubo 光学电导结果 + 两张全局汇总图（Re/Im）
作者：Wang（派生自 compare_kmesh_optics.py，去掉对比，保留并强化汇总绘图）

本版修改：
1) 提取后的光电导率数据不再写回原应变目录，而是统一写到新的同级目录 OUTPUT_ROOT 下；
2) OUTPUT_ROOT 的二级目录名仍使用应变组名称（如 ref_0.0 / zigzag_p0.001 / armchair_p0.001）；
3) 只提取和汇总 xy 平面内结果（S_xx, S_yy, S_xy, S_yx, A_xy, A_yx）；
4) 全局汇总图也只画 xy 平面内分量；
5) 绘图时横轴按当前数据范围自动设置，不再固定从 0 开始。
"""

import os
import re
import sys
import glob
import math
import argparse
from typing import List, Tuple, Optional, Dict, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd

# matplotlib 后端保护（无显示环境也能出图）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 目标应变目录（保持你的原逻辑）======
TARGET_DIRS = [
    "00_Optimized_Structure",
    "Strain_a_Tension_0.1pct",
    "Strain_a_Tension_0.2pct",
    "Strain_a_Tension_0.3pct",
    "Strain_a_Tension_0.4pct",
    "Strain_a_Tension_0.5pct",
    "Strain_b_Tension_0.1pct",
    "Strain_b_Tension_0.2pct",
    "Strain_b_Tension_0.3pct",
    "Strain_b_Tension_0.4pct",
    "Strain_b_Tension_0.5pct",
]

# ====== 新的同级输出总目录（可自行修改）======
OUTPUT_ROOT = "optics_sigma_xy"

# ====== 仅保留 xy 平面内分量 ======
ALLOWED_COMPONENTS = {
    "S_xx", "S_yy", "S_xy", "S_yx",
    "A_xy", "A_yx",
}

# ====== 分量友好名映射（仅保留 xy 平面）======
LABEL_MAP = {
    "S_xx": "σ_xx",
    "S_yy": "σ_yy",
    "S_xy": "σ_xy",
    "S_yx": "σ_yx",
    "A_xy": "κ_xy",
    "A_yx": "κ_yx",
}

# ====== 文件名正则（尽量宽松）======
KUBO_FILE_RE = re.compile(r"wannier90[-_ ]kubo[-_ ]([A-Za-z]_[xyz]{2})\.dat$", re.IGNORECASE)

# ====== 汇总图分量优先级（仅 xy 平面）======
COMP_PRIORITY = ["S_xx", "S_yy", "S_xy", "S_yx", "A_xy", "A_yx"]


# ---------- 工具函数 ----------

def find_dataset_dirs(root: str) -> List[str]:
    """
    在 root 下递归查找“包含至少一个 wannier90-kubo_*.dat 的目录”。
    这些目录被视作“数据子目录”。
    """
    ds = set()
    for path, _, files in os.walk(root):
        for fn in files:
            if KUBO_FILE_RE.search(fn):
                ds.add(path)
                break
    return sorted(ds)


def load_sigma_file(path: str) -> Optional[pd.DataFrame]:
    """
    读取单个 kubo 文件：返回 DataFrame[omega_eV, Re, Im]（按 ω 升序且去重）
    若失败返回 None。
    """
    if not os.path.isfile(path):
        return None
    try:
        arr = np.loadtxt(path, comments="#", dtype=float, ndmin=2)
    except Exception:
        return None
    if arr.size == 0 or arr.shape[1] < 3:
        return None
    df = pd.DataFrame({
        "omega_eV": arr[:, 0],
        "Re": arr[:, 1],
        "Im": arr[:, 2],
    })
    # 排序并去除重复 ω
    df.sort_values("omega_eV", inplace=True, kind="mergesort")
    df = df[~df["omega_eV"].duplicated(keep="first")].reset_index(drop=True)
    return df


def linear_extrapolate_to_zero(df: pd.DataFrame) -> Tuple[float, float, bool]:
    """
    估算 ω→0 极限值（Re, Im）。
    - 若恰好包含 ω=0，直接返回该点；
    - 否则用前两个点线性外推；若点数不足则返回 (nan, nan, False)
    返回：(Re0, Im0, has_exact_zero)
    """
    w = df["omega_eV"].to_numpy()
    re_ = df["Re"].to_numpy()
    im_ = df["Im"].to_numpy()
    if len(w) == 0:
        return math.nan, math.nan, False
    zero_mask = np.isclose(w, 0.0, atol=1e-12)
    if np.any(zero_mask):
        idx = int(np.where(zero_mask)[0][0])
        return float(re_[idx]), float(im_[idx]), True
    if len(w) >= 2:
        w1, w2 = w[0], w[1]
        re1, re2 = re_[0], re_[1]
        im1, im2 = im_[0], im_[1]
        if not np.isclose(w2, w1):
            re0 = re1 + (re2 - re1) * (0.0 - w1) / (w2 - w1)
            im0 = im1 + (im2 - im1) * (0.0 - w1) / (w2 - w1)
            return float(re0), float(im0), False
    return math.nan, math.nan, False


def spectral_weight(df: pd.DataFrame, wmax: Optional[float]) -> float:
    """
    计算 Re(σ) 的谱权重 ∫_0^{wmax} Re(σ) dω （梯形法）。
    - 若 wmax=None，使用该谱线自身最大 ω。
    - 若 wmax 超出数据范围，截断到可用范围。
    """
    if df is None or len(df) < 2:
        return math.nan
    w = df["omega_eV"].to_numpy()
    re_ = df["Re"].to_numpy()
    mask = w >= 0.0
    w = w[mask]
    re_ = re_[mask]
    if len(w) < 2:
        return math.nan
    w_hi = w[-1] if wmax is None else min(wmax, w[-1])
    if w_hi <= w[0]:
        return 0.0
    if w_hi < w[-1]:
        re_hi = np.interp(w_hi, w, re_)
        last_valid = w <= w_hi + 1e-12
        if w[last_valid][-1] < w_hi - 1e-12:
            w = np.append(w[last_valid], w_hi)
            re_ = np.append(re_[last_valid], re_hi)
        else:
            w = w[last_valid]
            re_ = re_[last_valid]
    return float(np.trapz(re_, w))


def build_output_dir(root: str, dataset_dir: str, output_root: str) -> str:
    """
    为某个原始数据子目录构造新的输出目录。

    规则：
    - 一级总目录：OUTPUT_ROOT
    - 二级目录：应变组名称（即 root 名称）
    - 若 dataset_dir 在 root 下还有更深层结构，则继续保留相对层级

    例如：
      root       = ref_0.0
      dataset_dir= ref_0.0/sub1/sub2
      输出为：
      OUTPUT_ROOT/ref_0.0/sub1/sub2
    """
    root_name = os.path.basename(os.path.normpath(root))
    rel_sub = os.path.relpath(dataset_dir, root)

    if rel_sub == ".":
        return os.path.join(output_root, root_name)
    return os.path.join(output_root, root_name, rel_sub)


def make_dataset_label(root: str, dataset_dir: str) -> str:
    """
    生成用于图例/汇总的标签。
    - 若 dataset_dir == root，则标签为 root 名称
    - 若有更深层级，则为 root_name/相对子路径
    """
    root_name = os.path.basename(os.path.normpath(root))
    rel_sub = os.path.relpath(dataset_dir, root)
    if rel_sub == ".":
        return root_name
    return f"{root_name}/{rel_sub}"


def infer_x_limits_from_dfs(dfs: List[pd.DataFrame], pad_ratio: float = 0.03) -> Tuple[float, float]:
    """
    根据若干 DataFrame 中的 omega_eV 数据自动推断横轴范围，并加少量边距。
    若范围退化（xmin≈xmax），自动补一个小宽度。
    """
    xs_min = []
    xs_max = []

    for df in dfs:
        if df is None or len(df) == 0:
            continue
        x = df["omega_eV"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        xs_min.append(float(np.min(x)))
        xs_max.append(float(np.max(x)))

    if not xs_min:
        return 0.0, 1.0

    xmin = min(xs_min)
    xmax = max(xs_max)

    if np.isclose(xmin, xmax):
        dx = max(abs(xmin) * 0.05, 0.1)
        return xmin - dx, xmax + dx

    dx = xmax - xmin
    pad = dx * pad_ratio
    return xmin - pad, xmax + pad


def summarize_component(df: pd.DataFrame, component_key: str, label: str,
                        dataset_dir: str, wmax_limit: Optional[float],
                        write_dir: str, write_plots: bool,
                        dataset_label: str) -> Dict[str, object]:
    """
    对单个分量做统计、输出标准化 CSV、可选 quicklook 图。
    返回一行 dict 用于汇总。
    """
    os.makedirs(write_dir, exist_ok=True)
    out_csv = os.path.join(write_dir, f"{component_key}.csv")
    df.to_csv(out_csv, index=False, float_format="%.8f")

    # 基本统计
    npts = len(df)
    w_min = float(df["omega_eV"].min())
    w_max = float(df["omega_eV"].max())
    if npts >= 3:
        dw = np.diff(df["omega_eV"].to_numpy())
        dw_med = float(np.median(dw))
    else:
        dw_med = math.nan

    re0, im0, has0 = linear_extrapolate_to_zero(df)

    re_array = df["Re"].to_numpy()
    im_array = df["Im"].to_numpy()
    abs_array = np.hypot(re_array, im_array)

    i_remax = int(np.nanargmax(re_array))
    i_absmax = int(np.nanargmax(abs_array))

    re_max = float(re_array[i_remax])
    re_max_at = float(df["omega_eV"].iat[i_remax])
    abs_max = float(abs_array[i_absmax])
    abs_max_at = float(df["omega_eV"].iat[i_absmax])

    sw = spectral_weight(df, wmax_limit)

    if write_plots:
        try:
            plt.figure(figsize=(6, 4))
            plt.plot(df["omega_eV"], df["Re"], label="Re")
            plt.plot(df["omega_eV"], df["Im"], label="Im", linestyle="--")
            xlo, xhi = infer_x_limits_from_dfs([df], pad_ratio=0.03)
            plt.xlim(xlo, xhi)
            plt.xlabel("Energy (eV)")
            plt.ylabel(f"{label} (arb.)")
            plt.title(f"{label} — {dataset_label}")
            plt.legend()
            plt.tight_layout()
            png = os.path.join(write_dir, f"{component_key}_quicklook.png")
            plt.savefig(png, dpi=200)
            plt.close()
        except Exception:
            pass

    return {
        "dataset_dir": dataset_label,
        "component": component_key,
        "label": label,
        "n_points": npts,
        "omega_min_eV": w_min,
        "omega_max_eV": w_max,
        "domega_median_eV": dw_med,
        "has_omega_0": int(has0),
        "Re_sigma_at_0": re0,
        "Im_sigma_at_0": im0,
        "Re_max": re_max,
        "omega_at_Re_max": re_max_at,
        "abs_sigma_max": abs_max,
        "omega_at_abs_max": abs_max_at,
        "spectral_weight_Re_trapz": sw,
    }


# ---------- 全局汇总图 ----------

def choose_components_for_panels(all_components: List[str], max_panels: int) -> List[str]:
    """按优先级从 all_components 中选最多 max_panels 个用于全局汇总图面板。"""
    uniq = list(dict.fromkeys(all_components))  # 去重保序
    chosen = []

    # 先按预设优先级
    for c in COMP_PRIORITY:
        if c in uniq and c not in chosen:
            chosen.append(c)
        if len(chosen) >= max_panels:
            return chosen

    # 不够再补剩余
    for c in uniq:
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= max_panels:
            break
    return chosen


def subplot_grid(n: int) -> Tuple[int, int]:
    """给 n 个面板返回合适的 (nrow, ncol)。"""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    return 3, 4  # 最多 12 面板


def make_global_summary_plots(comp_to_series: Dict[str, List[Tuple[str, pd.DataFrame]]],
                              max_panels: int,
                              output_root: str) -> None:
    """
    生成两张全局汇总图：
      - OUTPUT_ROOT/optics_summary_Re.png : 多面板，各分量的 Re(σ) 叠加多子目录曲线
      - OUTPUT_ROOT/optics_summary_Im.png : 同上但 Im(σ)
    comp_to_series: {component_key: [(dataset_label, df), ...], ...}
    """
    components_all = [c for c in comp_to_series.keys() if len(comp_to_series[c]) > 0]
    if not components_all:
        print("[提示] 无可绘制的分量，跳过全局汇总图。")
        return

    chosen = choose_components_for_panels(components_all, max_panels=max_panels)
    n = len(chosen)
    nrow, ncol = subplot_grid(n)

    # ---- Re 图
    fig_re, axes_re = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.8 * nrow), squeeze=False)
    # ---- Im 图
    fig_im, axes_im = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.8 * nrow), squeeze=False)

    for idx, comp in enumerate(chosen):
        r = idx // ncol
        c = idx % ncol
        axR = axes_re[r][c]
        axI = axes_im[r][c]

        label = LABEL_MAP.get(comp, comp)
        axR.set_title(f"{label} (Re)")
        axI.set_title(f"{label} (Im)")

        series = comp_to_series.get(comp, [])
        dfs_this_panel = []

        for ds_label, df in series:
            dfs_this_panel.append(df)
            axR.plot(df["omega_eV"].to_numpy(), df["Re"].to_numpy(), label=ds_label, lw=1.0)
            axI.plot(df["omega_eV"].to_numpy(), df["Im"].to_numpy(), label=ds_label, lw=1.0)

        xlo, xhi = infer_x_limits_from_dfs(dfs_this_panel, pad_ratio=0.03)

        for ax in (axR, axI):
            ax.set_xlabel("Energy (eV)")
            ax.set_xlim(xlo, xhi)
            ax.grid(alpha=0.25, linestyle=":")
        axR.set_ylabel("Re σ / κ (arb.)")
        axI.set_ylabel("Im σ / κ (arb.)")

    # 多余空白子图隐藏
    for k in range(n, nrow * ncol):
        r = k // ncol
        c = k % ncol
        fig_re.delaxes(axes_re[r][c])
        fig_im.delaxes(axes_im[r][c])

    # 统一总图例
    handles, labels = [], []
    for ax in fig_re.axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    if labels:
        uniq = []
        seen = set()
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        handles, labels = zip(*uniq)
        fig_re.legend(handles, labels, loc="upper center",
                      ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig_im.legend(handles, labels, loc="upper center",
                      ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig_re.tight_layout(rect=[0, 0, 1, 0.96])
    fig_im.tight_layout(rect=[0, 0, 1, 0.96])

    fig_re.savefig(os.path.join(output_root, "optics_summary_Re.png"), dpi=250)
    fig_im.savefig(os.path.join(output_root, "optics_summary_Im.png"), dpi=250)
    plt.close(fig_re)
    plt.close(fig_im)
    print(f"[OK] 全局汇总图已写出：{output_root}/optics_summary_Re.png / optics_summary_Im.png")


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="提取/汇总 Wannier90-Kubo 光学电导结果（仅 xy 平面）")
    ap.add_argument("--targets", nargs="*", help="目标目录列表（覆盖脚本内 TARGET_DIRS）")
    ap.add_argument("--wmax", type=float, default=None, help="谱权重积分上限（eV），默认用各谱线自身最大 ω")
    ap.add_argument("--plots", action="store_true", help="为每个数据子目录绘制 quicklook PNG 图")
    ap.add_argument("--max-panels", type=int, default=6, help="全局汇总图最多展示的分量个数（默认 6）")
    args = ap.parse_args()

    roots = args.targets if args.targets else [d for d in TARGET_DIRS if os.path.isdir(d)]
    if not roots:
        print("未找到可处理的目标目录（请检查 TARGET_DIRS 或使用 --targets）。")
        sys.exit(2)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    # 用于全局汇总图：comp_key -> [(dataset_label, df), ...]
    comp_to_series: DefaultDict[str, List[Tuple[str, pd.DataFrame]]] = defaultdict(list)

    for root in roots:
        dataset_dirs = find_dataset_dirs(root)
        if not dataset_dirs:
            print(f"[提示] {root} 下未发现 kubo 输出文件。")
            continue

        for d in dataset_dirs:
            out_dir = build_output_dir(root, d, OUTPUT_ROOT)
            os.makedirs(out_dir, exist_ok=True)

            # 数据子目录标签：保留应变组名 + 相对层级
            ds_label = make_dataset_label(root, d)

            # 收集所有 kubo 文件
            files = sorted(glob.glob(os.path.join(d, "wannier90-kubo_*.dat"))) + \
                    sorted(glob.glob(os.path.join(d, "wannier90-kubo *.dat"))) + \
                    sorted(glob.glob(os.path.join(d, "wannier90-kubo-*.dat")))

            stats_this_dir: List[Dict[str, object]] = []

            for fp in files:
                m = KUBO_FILE_RE.search(os.path.basename(fp))
                if not m:
                    continue

                comp_key = m.group(1)  # 例如 S_xx / A_xy
                if comp_key not in ALLOWED_COMPONENTS:
                    continue

                label = LABEL_MAP.get(comp_key, comp_key)

                df = load_sigma_file(fp)
                if df is None or len(df) < 2:
                    print(f"[跳过] 数据不足：{os.path.relpath(fp)}")
                    continue

                row = summarize_component(
                    df=df,
                    component_key=comp_key,
                    label=label,
                    dataset_dir=d,
                    wmax_limit=args.wmax,
                    write_dir=out_dir,
                    write_plots=args.plots,
                    dataset_label=ds_label,
                )
                stats_this_dir.append(row)
                summary_rows.append(row)

                # 用于全局汇总图
                comp_to_series[comp_key].append((ds_label, df))

            # 写该目录的局部统计
            if stats_this_dir:
                df_dir = pd.DataFrame(stats_this_dir)
                df_dir.to_csv(
                    os.path.join(out_dir, "_stats_per_dir.csv"),
                    index=False,
                    float_format="%.6f"
                )
                print(f"[OK] 目录统计：{os.path.relpath(out_dir)}/_stats_per_dir.csv")
            else:
                print(f"[提示] {os.path.relpath(d)} 无可用的 xy 平面分量。")

    # 跨目录汇总
    if summary_rows:
        df_all = pd.DataFrame(summary_rows)
        cols = [
            "dataset_dir", "component", "label",
            "n_points", "omega_min_eV", "omega_max_eV", "domega_median_eV",
            "has_omega_0", "Re_sigma_at_0", "Im_sigma_at_0",
            "Re_max", "omega_at_Re_max", "abs_sigma_max", "omega_at_abs_max",
            "spectral_weight_Re_trapz"
        ]
        df_all = df_all.reindex(columns=cols)
        df_all.to_csv(
            os.path.join(OUTPUT_ROOT, "optics_summary.csv"),
            index=False,
            float_format="%.6f"
        )
        print(f"[OK] 跨目录汇总已写出：{OUTPUT_ROOT}/optics_summary.csv")
    else:
        print("[提示] 未收集到任何 xy 平面谱线数据。")

    # 全局汇总图（两张）
    make_global_summary_plots(comp_to_series, max_panels=args.max_panels, output_root=OUTPUT_ROOT)


if __name__ == "__main__":
    main()