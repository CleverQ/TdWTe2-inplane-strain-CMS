#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_wt_weyl_evolution.py

功能：
  读取 run_wanniertools_weyl_standard_flow.py 生成的 WannierTools Weyl 节点结果，
  对 c 轴应变组的 N56 Weyl 节点进行统一后处理与作图。

脚本定位：
  后续 Weyl 点定位与 chirality 校验统一以 WannierTools 输出为准；
  本脚本只负责：
    1) 读取 WannierTools 输出；
    2) 匹配 WeylChirality_calc 输出的 chirality；
    3) 汇总 c 轴应变下 Weyl 节点数量、能量、动量位置；
    4) 对可追踪节点进行 family tracking；
    5) 绘制 Weyl 节点数量 vs c 轴应变；
    6) 绘制 Weyl 节点能量 E_W-E_F vs c 轴应变；
    7) 绘制 Weyl 点在 centered k1-k2 平面的运动轨迹与应变叠加图；
    8) 绘制代表性 Weyl 点对的间距 vs 应变；
    9) 绘制局部 k-line 能带图；
   10) 可选绘制单个 Weyl cone 的二维切线图与三维双能带曲面；
   11) 可选绘制成对 Weyl 点附近两个能量面相交的三维曲面图；
   12) 新增：在 selected_case_k1k2_maps 文件夹中绘制
       “只包含 -1% 到 +2% 区间内持续存在的 4 对 Weyl 点”的
       -2%、-1%、0%、+1%、+2% 五面板 k1-k2 图，且五幅图使用统一坐标轴范围。

推荐执行顺序：
  Step 1.
      python run_wanniertools_weyl_standard_flow.py
    作用：
      用 WannierTools 对每个应变态执行 FindNodes_calc + WeylChirality_calc。
      推荐当前主流程：
        NumOccupied = 56
        FindNodes Nk = 11 × 11 × 11
        Gap_threshold = 0.002 eV
        WeylChirality radius = 0.002

  Step 2.
      python plot_wt_weyl_evolution.py
    作用：
      读取 Step 1 的输出并作图。

主要输出：
  <OUT_ROOT>/
    wt_nodes_with_chirality.csv
    family_tracks_c.csv
    family_tracks_c_persistent_core.csv
    family_tracks_c_transient.csv
    family_summary.csv
    persistent_family_summary.csv
    transient_family_summary.csv
    selected_case_k1k2_maps/representative_k1k2_maps.png
    selected_case_k1k2_maps/core_4pair_minus2_to_plus2_k1k2_maps.png
    selected_case_k1k2_maps/core_4pair_minus2_to_plus2_nodes.csv
    timing_summary.csv
    timing_summary.txt
"""

from __future__ import annotations

import os
import re
import csv
import glob
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Set
from contextlib import contextmanager

import numpy as np

MPL_BACKEND = "Agg"                 # 服务器/无图形界面环境建议使用 Agg
if MPL_BACKEND:
    import matplotlib
    matplotlib.use(MPL_BACKEND)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ======================================================================
# 用户可修改：输入输出路径
# ======================================================================

ROOT_DIR = os.getcwd()               # 当前脚本运行目录；通常就是 Step5_Optics_2_MLWF
CASE_INPUT_ROOT = ROOT_DIR           # 原始 MLWF case 目录所在根目录；包含各应变子目录

# 如果留空，脚本会从 WT_FLOW_ROOT_CANDIDATES 中自动寻找最新的 WannierTools 输出目录。
# 推荐在正式作图时显式指定，避免误读验证目录。
WT_FLOW_ROOT = "baseline_N56_Nk11_gap0p002_r0p002"

# 当 WT_FLOW_ROOT = "" 时，按下列通配符自动寻找。
WT_FLOW_ROOT_CANDIDATES = [
    "baseline_N56_Nk11_gap0p002_r0p002",
    "wt_weyl_main_c_minus5_to_plus5_N56_gap0p002_Nk11",
    "wt_weyl_standard_flow_c_minus5_to_plus5_N56_gap0p002_Nk11",
    "wt_weyl_standard_flow*",
]

OUT_ROOT = os.path.join(ROOT_DIR, "wt_weyl_evolution_output_c_minus5_to_plus5")


# ======================================================================
# 用户可修改：目标 case 与 c 轴路径定义
# ======================================================================

TARGET_DIRS = [
    "Strain_c_Compression_5%",
    "Strain_c_Compression_4%",
    "Strain_c_Compression_3%",
    "Strain_c_Compression_2%",
    "Strain_c_Compression_1%",
    "00_Optimized_Structure",
    "Strain_c_Tension_1%",
    "Strain_c_Tension_2%",
    "Strain_c_Tension_3%",
    "Strain_c_Tension_4%",
    "Strain_c_Tension_5%",
]

PATH_CASES = {
    "c": TARGET_DIRS,
}

# 主演化图只画这些 case 中一直存在的 Weyl family。
# 这部分仍保留，适合画 -5% 到 +2% 内持续存在的 family。
PERSISTENT_CORE_CASES = [
    "Strain_c_Compression_5%",
    "Strain_c_Compression_4%",
    "Strain_c_Compression_3%",
    "Strain_c_Compression_2%",
    "Strain_c_Compression_1%",
    "00_Optimized_Structure",
    "Strain_c_Tension_1%",
    "Strain_c_Tension_2%",
]

PLOT_TRANSIENT_FAMILIES_SEPARATELY = True   # True=中途出现/消失的 family 单独作图
TRANSIENT_MIN_FAMILY_LENGTH = 2             # transient family 至少出现几个应变态才单独作图


# ======================================================================
# 用户可修改：数据选择与解释方式
# ======================================================================

NUM_OCC_TO_USE = 56                  # 只读取 NumOccupied=56 的 WannierTools 结果
REQUIRE_CHIRALITY_MATCH = True       # True=必须匹配到 WeylChirality_calc 输出
REQUIRE_NONZERO_CHIRALITY = True     # True=丢弃 chirality=0 的点
CHIRALITY_MATCH_K_TOL = 2.0e-4       # FindNodes 节点与 chirality 节点匹配的分数坐标距离阈值
ENERGY_RELATIVE_MODE = "auto"        # auto/raw_is_relative/csv_energy_rel

# 若只想快速生成统计图、节点位置图和 4 对 Weyl 点图，建议设为 False。
COMPUTE_TILT_FROM_HR = False         # True=从 HR 重新计算 tilt_ratio；会增加耗时
TILT_TYPE2_THRESHOLD = 1.0           # tilt_ratio > 1 判为 type-II-like


# ======================================================================
# 用户可修改：k 坐标显示方式
# ======================================================================

CENTER_K_COORDINATES = True          # True=把 [0,1) 分数坐标转换到 [-0.5,0.5)，使 0 点在图中心
AUTO_ZOOM_CENTERED_KMAP = True       # True=centered k 图自动缩放到节点附近
CENTERED_KMAP_MARGIN = 0.025         # 自动缩放边距
CENTERED_KMAP_MIN_HALF_WIDTH = 0.080 # 自动缩放最小半宽
CENTERED_KMAP_MAX_HALF_WIDTH = 0.500 # 自动缩放最大半宽


# ======================================================================
# 用户可修改：新增 4 对 Weyl 点论文图设置
# ======================================================================

PLOT_SELECTED_CORE_4PAIR_K1K2_MAPS = True

# 用这些 case 来定义“论文主图中要追踪的 4 对 Weyl 点”。
# 物理含义：
#   这 8 个节点在 -1%、0%、+1%、+2% 中都存在；
#   它们就是 0% 附近稳定存在的 4 对 N56 Weyl 点。
CORE_4PAIR_SELECTION_CASES = [
    "Strain_c_Compression_1%",
    "00_Optimized_Structure",
    "Strain_c_Tension_1%",
    "Strain_c_Tension_2%",
]

# 最终绘图面板：只显示上述 8 个 family 在这些 case 中的位置。
# 注意：
#   -2% 实际有 20 个节点，但此图只显示属于上述 8 个 family 的那部分节点。
CORE_4PAIR_PANEL_CASES = [
    "Strain_c_Compression_2%",
    "Strain_c_Compression_1%",
    "00_Optimized_Structure",
    "Strain_c_Tension_1%",
    "Strain_c_Tension_2%",
]

CORE_4PAIR_EXPECTED_NODE_COUNT = 8
CORE_4PAIR_EXPECTED_PAIR_COUNT = 4

# 4 对 Weyl 点图是否把五个面板使用完全统一坐标轴范围。
CORE_4PAIR_USE_SHARED_AXIS = True

# 坐标轴范围：
#   "auto"  : 根据五个面板中的 8 个 family 自动设置统一范围；
#   "fixed" : 使用 CORE_4PAIR_FIXED_XLIM / CORE_4PAIR_FIXED_YLIM。
CORE_4PAIR_AXIS_MODE = "auto"
CORE_4PAIR_FIXED_XLIM = (-0.18, 0.18)
CORE_4PAIR_FIXED_YLIM = (-0.16, 0.16)
CORE_4PAIR_AXIS_MARGIN = 0.025
CORE_4PAIR_MIN_HALF_WIDTH = 0.080

# 是否在每个点旁标 family_id。
# 论文主图通常建议 False，避免文字拥挤。
CORE_4PAIR_ANNOTATE_FAMILY_ID = False

# 是否在 -2% 面板中若 family 缺失则用最邻近同手性节点补位。
# 默认 False，避免引入非 family-tracking 确认的节点。
CORE_4PAIR_ALLOW_NEAREST_FALLBACK = False
CORE_4PAIR_FALLBACK_K_TOL = 0.04


# ======================================================================
# 用户可修改：颜色、图例与 chirality 显示
# ======================================================================

USE_DISTINCT_FAMILY_COLORS = True
FAMILY_COLOR_CMAP = "tab20"
FAMILY_COLOR_CMAP_LARGE = "hsv"
MAX_LEGEND_ITEMS = 40

# 节点位置图、单应变空间位置图、点对图默认用红蓝表示 chirality。
USE_CHIRALITY_COLOR_FOR_NODE_MAPS = True
USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS = True

POS_COLOR = "tab:red"                # chirality = +1
NEG_COLOR = "tab:blue"               # chirality = -1
ZERO_COLOR = "tab:gray"              # chirality = 0
REF_COLOR = "black"

MARKER_POS = "o"
MARKER_NEG = "^"
MARKER_ZERO = "s"

UNIFIED_NODE_COLOR = "tab:purple"
UNIFIED_NODE_MARKER = "o"


# ======================================================================
# 用户可修改：family 追踪参数
# ======================================================================

MATCH_REQUIRE_SAME_CHIRALITY = True
MATCH_K_WEIGHT = 1.0
MATCH_E_WEIGHT = 0.20
MATCH_GAP_WEIGHT = 0.02
MATCH_CHIRALITY_PENALTY = 100.0
MAX_MATCH_COST = 10.0

MIN_FAMILY_LENGTH_FOR_PAIR_PLOT = 2


# ======================================================================
# 用户可修改：作图开关
# ======================================================================

WRITE_OUTPUT_FILES = True
PRINT_SUMMARY = True

PLOT_COUNT_VS_STRAIN = True
PLOT_CHIRALITY_COUNT_VS_STRAIN = False
PLOT_ENERGY_SCATTER = True

# 快速出论文图时，这些 3D 或多轨迹图建议关闭。
PLOT_K_TRAJECTORY_3D_RAW = False
PLOT_K_TRAJECTORY_3D_UNWRAPPED = False
PLOT_K_COMPONENTS_VS_STRAIN = False
PLOT_K_PROJECTIONS_2D = False
PLOT_K1K2_TRAJECTORY = False

PLOT_ENERGY_VS_STRAIN = False
PLOT_GAP_VS_STRAIN = False
PLOT_TILT_RATIO_VS_STRAIN = False
PLOT_CHIRALITY_PAIR_COMPARISON = False

PLOT_CASE_NODE_MAPS = False
PLOT_SELECTED_CASE_K1K2_MAPS = True
PLOT_SELECTED_CASE_3D_MAPS = False
PLOT_OVERLAY_NODE_MAPS = False
PLOT_PAIR_DISTANCE_VS_STRAIN = True
PLOT_PAIR_DISTANCE_CART_FIGURE = False  # False=不输出 1/Angstrom 或 perA 版本的距离图/数据
PLOT_SCHEMATIC_LIKE = False

REPRESENTATIVE_CASES = [
    "Strain_c_Compression_3%",
    "Strain_c_Compression_2%",
    "Strain_c_Compression_1%",
    "00_Optimized_Structure",
    "Strain_c_Tension_1%",
    "Strain_c_Tension_2%",
    "Strain_c_Tension_3%",
]


# ======================================================================
# 用户可修改：Weyl cone、局部 k-line 与成对能量面图
# ======================================================================

# 快速出论文节点图时建议全部关闭。
PLOT_WEYL_CONE_3D = False
PLOT_WEYL_CONE_CUTS = False
PLOT_PAIR_KLINE_BANDS = False
PLOT_PAIR_ENERGY_SURFACE_3D = False

CONE_SELECTED_CASES = [
    "Strain_c_Compression_2%",
    "00_Optimized_Structure",
    "Strain_c_Tension_2%",
]

CONE_NODE_SELECTION_MODE = "closest_to_ef"
CONE_MANUAL_SELECTION = {
    "c": [],
}
CONE_MAX_NODES_PER_CASE = 2

CONE_PLANE_LIST = [(0, 1)]
CONE_QMAX_FRAC = 0.006
CONE_GRID_N = 45
CONE_SHIFT_TO_NODE_ENERGY = True
CONE_VIEW_ELEV = 24
CONE_VIEW_AZIM = -55

CUT_LINE_QMAX_FRAC = 0.010
CUT_LINE_NPTS = 121

PAIR_PLOT_SELECTED_CASES = [
    "Strain_c_Compression_2%",
    "00_Optimized_Structure",
    "Strain_c_Tension_2%",
]
PAIR_MAX_PAIRS_PER_CASE = 4
PAIR_KLINE_MARGIN_FRAC = 0.025
PAIR_KLINE_NPTS = 181
PAIR_SURFACE_EXTRA_FRAC = 0.020
PAIR_SURFACE_PERP_HALF_FRAC = 0.030
PAIR_SURFACE_GRID_N = 55
PAIR_SURFACE_SHIFT_TO_EF = True
PAIR_SURFACE_VIEW_ELEV = 25
PAIR_SURFACE_VIEW_AZIM = -55


# ======================================================================
# 用户可修改：Origin 数据表格导出设置
# ======================================================================

EXPORT_ORIGIN_TABLES = True

# True=Origin 数据输出与对应 Python 绘图开关同步：
#   PLOT_PAIR_DISTANCE_VS_STRAIN=True 才输出 pair distance Origin 表；
#   PLOT_PAIR_KLINE_BANDS=True 才输出 k-line Origin 表。
# False=即使不画对应 Python 图，也允许单独导出 Origin 表。
ORIGIN_OUTPUT_SYNC_WITH_PLOTS = False

ORIGIN_OUT_DIR = os.path.join(OUT_ROOT, "origin_data")

# False=Origin 导出表格不输出 1/Angstrom 或 perA 相关列，只保留 centered fractional k 坐标。
EXPORT_ORIGIN_PER_A_COLUMNS = False

EXPORT_ORIGIN_PAIR_DISTANCE_TABLES = True
EXPORT_ORIGIN_PAIR_DISTANCE_LONG_TABLE = True
EXPORT_ORIGIN_PAIR_DISTANCE_WIDE_TABLE = True

EXPORT_ORIGIN_KLINE_TABLES = True

# 当 ORIGIN_OUTPUT_SYNC_WITH_PLOTS=True 时，k-line 表格只在 PLOT_PAIR_KLINE_BANDS=True 时导出；
# 当 ORIGIN_OUTPUT_SYNC_WITH_PLOTS=False 时，可使用 ORIGIN_KLINE_SELECTED_CASES 单独导出。
ORIGIN_KLINE_SELECTED_CASES = [
    "Strain_c_Compression_2%",
    "00_Optimized_Structure",
    "Strain_c_Tension_2%",
]

ORIGIN_KLINE_MAX_PAIRS_PER_CASE = 4

# True 时额外生成 .dat 制表符分隔文件，Origin 读取通常更稳定。
EXPORT_ORIGIN_TSV_DAT = True

# True 时为每条 k-line 单独输出一张表；同时也会输出 all_kline_data_long.csv 总表。
EXPORT_ORIGIN_INDIVIDUAL_KLINE_TABLES = True

# k-line 能量参考：
#   "fermi"      : E-E_F
#   "node_mean"  : 以两个 Weyl 节点平均能量为零点
#   "raw"        : 直接输出 HR 本征值
ORIGIN_KLINE_ENERGY_REFERENCE = "fermi"


# ======================================================================
# 用户可修改：schematic-like 图参数
# ======================================================================

SCHEMATIC_PROJECTION = ("k1_unwrapped", "k2_unwrapped", "k1", "k2")
SCHEMATIC_CASES_C = REPRESENTATIVE_CASES


# ======================================================================
# 用户可修改：计时开关
# ======================================================================

ENABLE_TIMING = True
PRINT_TIMING = True
WRITE_TIMING_SUMMARY = True


# ======================================================================
# 用户可修改：图像样式
# ======================================================================

DPI = 220
FIGSIZE = (7.8, 5.8)
FIGSIZE_TALL = (8.2, 8.8)
FIGSIZE_WIDE = (10.8, 4.8)
FIGSIZE_CONE = (7.8, 6.2)
FIGSIZE_TRIPLE = (14.0, 4.2)
FIGSIZE_PANEL = (15.0, 4.2)
FIGSIZE_CORE_4PAIR = (15.0, 4.2)


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class NodeRow:
    case_dir: str
    case_label: str
    direction: str
    eps_percent: float

    num_occupied: int
    pair_n_1based: int
    pair_m_1based: int

    kx_cart: float
    ky_cart: float
    kz_cart: float

    k1_frac: float
    k2_frac: float
    k3_frac: float

    gap_eV: float
    energy_raw_eV: float
    energy_rel_ef_eV: float
    fermi_energy_eV: float

    chirality: int
    chirality_match_distance: float

    tilt_ratio: float
    type_label: str
    singular_v1: float
    singular_v2: float
    singular_v3: float
    vx0: float
    vy0: float
    vz0: float


@dataclass
class TrackRow:
    path: str
    family_id: str
    case_label: str
    direction: str
    eps_percent: float

    num_occupied: int
    pair_n_1based: int
    pair_m_1based: int

    chirality: int

    k1_frac: float
    k2_frac: float
    k3_frac: float

    k1_unwrapped: float
    k2_unwrapped: float
    k3_unwrapped: float

    energy_rel_ef_eV: float
    gap_eV: float
    tilt_ratio: float
    type_label: str

    source_case_dir: str


# ======================================================================
# 计时工具
# ======================================================================

TIMING_RECORDS: List[dict] = []


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f} s"
    if sec < 3600:
        return f"{sec / 60:.2f} min"
    return f"{sec / 3600:.2f} h"


@contextmanager
def timed_section(name: str):
    if not ENABLE_TIMING:
        yield
        return

    t0 = time.perf_counter()
    if PRINT_TIMING:
        print(f"[TIMER START] {name}")
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        dt = time.perf_counter() - t0
        TIMING_RECORDS.append({
            "section": name,
            "elapsed_seconds": f"{dt:.6f}",
            "elapsed_human": format_seconds(dt),
            "status": status,
        })
        if PRINT_TIMING:
            print(f"[TIMER END  ] {name} | {format_seconds(dt)} | {status}")


def write_timing_summary(out_root: str):
    if not WRITE_TIMING_SUMMARY or not WRITE_OUTPUT_FILES:
        return

    ensure_dir(out_root)

    csv_path = os.path.join(out_root, "timing_summary.csv")
    txt_path = os.path.join(out_root, "timing_summary.txt")

    if TIMING_RECORDS:
        fields = ["section", "elapsed_seconds", "elapsed_human", "status"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in TIMING_RECORDS:
                w.writerow(r)

    total = sum(float(r["elapsed_seconds"]) for r in TIMING_RECORDS)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Timing summary for plot_wt_weyl_evolution.py\n")
        f.write("=" * 72 + "\n")
        for r in TIMING_RECORDS:
            f.write(f"{r['section']:<70s} {r['elapsed_human']:>12s}  {r['status']}\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'TOTAL':<70s} {format_seconds(total):>12s}\n")


# ======================================================================
# 基础工具
# ======================================================================

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def to_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(round(float(s)))
    except Exception:
        return default


def read_csv_dicts(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def write_dataclass_csv(path: str, rows: List):
    if not WRITE_OUTPUT_FILES:
        return
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_dict_csv(path: str, rows: List[dict]):
    if not WRITE_OUTPUT_FILES:
        return
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_origin_table(base_path_no_ext: str, rows: List[dict]):
    """
    为 Origin 输出同内容的 .csv 与 .dat 表格。

    base_path_no_ext:
      不带扩展名的路径。例如：
        /path/origin_data/kline/all_kline_data_long

    输出：
      base_path_no_ext.csv
      base_path_no_ext.dat
    """
    if not WRITE_OUTPUT_FILES or not EXPORT_ORIGIN_TABLES:
        return

    ensure_dir(os.path.dirname(base_path_no_ext))

    if not rows:
        with open(base_path_no_ext + ".csv", "w", encoding="utf-8", newline="") as f:
            f.write("")
        if EXPORT_ORIGIN_TSV_DAT:
            with open(base_path_no_ext + ".dat", "w", encoding="utf-8", newline="") as f:
                f.write("")
        return

    fields = list(rows[0].keys())

    def clean_value(v):
        if v is None:
            return ""
        if isinstance(v, float) and not np.isfinite(v):
            return ""
        return v

    with open(base_path_no_ext + ".csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: clean_value(row.get(k)) for k in fields})

    if EXPORT_ORIGIN_TSV_DAT:
        with open(base_path_no_ext + ".dat", "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(fields) + "\n")
            for row in rows:
                vals = [str(clean_value(row.get(k))) for k in fields]
                f.write("\t".join(vals) + "\n")


def wrap_frac_value(x: float) -> float:
    y = float(x) % 1.0
    if y < 0:
        y += 1.0
    return y


def center_frac_value(x: float) -> float:
    """
    把 [0,1) 周期分数坐标转换为 [-0.5,0.5) centered 坐标。
    例如：
      0.98 -> -0.02
      0.02 -> +0.02
    """
    return ((float(x) + 0.5) % 1.0) - 0.5


def center_frac_vector(k: Iterable[float]) -> np.ndarray:
    return np.array([center_frac_value(x) for x in k], dtype=float)


def minimal_image_delta(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    d = a - b
    d = d - np.round(d)
    return d


def periodic_distance(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(minimal_image_delta(a, b)))


def sanitize_name(s: str) -> str:
    return str(s).replace("%", "pct").replace("+", "p").replace("-", "m").replace("/", "_").replace(" ", "_")


def family_sort_key(fid: str):
    m = re.search(r"(\d+)", str(fid))
    if m:
        return int(m.group(1))
    return 10**9


def parse_case_meta(case_label: str) -> Tuple[str, float]:
    if case_label == "00_Optimized_Structure":
        return "ref", 0.0

    m = re.match(
        r"^Strain_([abc])_(Tension|Compression)_([0-9]+(?:\.[0-9]+)?)(?:pct|%)$",
        case_label,
        re.I,
    )
    if m:
        direction = m.group(1).lower()
        eps = float(m.group(3))
        if m.group(2).lower() == "compression":
            eps = -eps
        return direction, eps

    return "default", np.nan


def make_distinct_color_map(keys: Iterable[str]) -> Dict[str, object]:
    keys = sorted(list(set(keys)), key=family_sort_key)
    n = len(keys)
    if n == 0:
        return {}

    cmap_name = FAMILY_COLOR_CMAP if n <= 20 else FAMILY_COLOR_CMAP_LARGE
    cmap = plt.get_cmap(cmap_name)

    out = {}
    for i, key in enumerate(keys):
        if cmap_name == "tab20":
            out[key] = cmap(i % 20)
        else:
            out[key] = cmap(i / max(1, n - 1))
    return out


def node_chirality_color(chirality: int) -> str:
    if chirality > 0:
        return POS_COLOR
    if chirality < 0:
        return NEG_COLOR
    return ZERO_COLOR


def node_chirality_marker(chirality: int) -> str:
    if chirality > 0:
        return MARKER_POS
    if chirality < 0:
        return MARKER_NEG
    return MARKER_ZERO


def add_chirality_legend(ax):
    handles = [
        Line2D([0], [0], marker=MARKER_POS, color="w", markerfacecolor=POS_COLOR,
               markeredgecolor="k", markersize=8, label=r"$\chi=+1$"),
        Line2D([0], [0], marker=MARKER_NEG, color="w", markerfacecolor=NEG_COLOR,
               markeredgecolor="k", markersize=8, label=r"$\chi=-1$"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)


def add_family_legend(ax, fontsize: int = 7):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen = set()
    new_handles = []
    new_labels = []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        new_handles.append(h)
        new_labels.append(lab)

    if len(new_handles) <= 12:
        ax.legend(new_handles, new_labels, loc="best", fontsize=fontsize)
    else:
        ax.legend(
            new_handles,
            new_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=max(5, fontsize - 1),
            frameon=True,
        )


def node_plot_coord(nd: NodeRow, attr: str) -> float:
    val = getattr(nd, attr)
    if CENTER_K_COORDINATES and attr in ("k1_frac", "k2_frac", "k3_frac"):
        return center_frac_value(val)
    return val


def track_plot_coord(r: TrackRow, attr: str) -> float:
    val = getattr(r, attr)
    if CENTER_K_COORDINATES and attr in ("k1_frac", "k2_frac", "k3_frac"):
        return center_frac_value(val)
    return val


def set_k_axis_limits_2d(ax, xs: List[float], ys: List[float]):
    if not xs or not ys:
        if CENTER_K_COORDINATES:
            ax.set_xlim(-CENTERED_KMAP_MIN_HALF_WIDTH, CENTERED_KMAP_MIN_HALF_WIDTH)
            ax.set_ylim(-CENTERED_KMAP_MIN_HALF_WIDTH, CENTERED_KMAP_MIN_HALF_WIDTH)
        else:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        return

    if CENTER_K_COORDINATES and AUTO_ZOOM_CENTERED_KMAP:
        max_abs = max(np.max(np.abs(xs)), np.max(np.abs(ys)))
        half = min(
            CENTERED_KMAP_MAX_HALF_WIDTH,
            max(CENTERED_KMAP_MIN_HALF_WIDTH, float(max_abs) + CENTERED_KMAP_MARGIN),
        )
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    elif CENTER_K_COORDINATES:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)


def set_k_axis_limits_3d(ax, xs: List[float], ys: List[float], zs: List[float]):
    if not xs or not ys or not zs:
        if CENTER_K_COORDINATES:
            half = CENTERED_KMAP_MIN_HALF_WIDTH
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_zlim(-half, half)
        else:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_zlim(-0.02, 1.02)
        return

    if CENTER_K_COORDINATES and AUTO_ZOOM_CENTERED_KMAP:
        max_abs = max(np.max(np.abs(xs)), np.max(np.abs(ys)), np.max(np.abs(zs)))
        half = min(
            CENTERED_KMAP_MAX_HALF_WIDTH,
            max(CENTERED_KMAP_MIN_HALF_WIDTH, float(max_abs) + CENTERED_KMAP_MARGIN),
        )
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)
    elif CENTER_K_COORDINATES:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
    else:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_zlim(-0.02, 1.02)


def find_latest_wt_flow_root() -> str:
    if WT_FLOW_ROOT:
        path = WT_FLOW_ROOT
        if not os.path.isabs(path):
            path = os.path.join(ROOT_DIR, path)
        if not os.path.isfile(os.path.join(path, "all_findnodes_unique_nodes.csv")):
            raise FileNotFoundError(f"WT_FLOW_ROOT 中找不到 all_findnodes_unique_nodes.csv: {path}")
        return path

    candidates = []
    for pat in WT_FLOW_ROOT_CANDIDATES:
        fullpat = pat if os.path.isabs(pat) else os.path.join(ROOT_DIR, pat)
        for d in glob.glob(fullpat):
            fp = os.path.join(d, "all_findnodes_unique_nodes.csv")
            if os.path.isdir(d) and os.path.isfile(fp):
                candidates.append(d)

    if not candidates:
        raise FileNotFoundError(
            "未找到包含 all_findnodes_unique_nodes.csv 的 wt_weyl_standard_flow* 目录。请设置 WT_FLOW_ROOT。"
        )

    candidates = sorted(
        set(candidates),
        key=lambda x: os.path.getmtime(os.path.join(x, "all_findnodes_unique_nodes.csv")),
        reverse=True,
    )
    return candidates[0]


# ======================================================================
# 读取 WannierTools 结果并匹配 chirality
# ======================================================================

def load_chirality_rows(wt_root: str, case_label: str, num_occ: int) -> List[dict]:
    csv_path = os.path.join(wt_root, case_label, f"wt_chirality_N{num_occ}", "chirality_result_rows.csv")
    if os.path.isfile(csv_path):
        rows = read_csv_dicts(csv_path)
        out = []
        for r in rows:
            out.append({
                "k1": wrap_frac_value(to_float(r.get("k1"))),
                "k2": wrap_frac_value(to_float(r.get("k2"))),
                "k3": wrap_frac_value(to_float(r.get("k3"))),
                "kx": to_float(r.get("kx")),
                "ky": to_float(r.get("ky")),
                "kz": to_float(r.get("kz")),
                "chirality": to_int(r.get("chirality"), 0),
            })
        return out

    wt_out = os.path.join(wt_root, case_label, f"wt_chirality_N{num_occ}", "WT.out")
    if not os.path.isfile(wt_out):
        return []

    out = []
    inside = False
    with open(wt_out, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if "Start of chirality of Weyl points calculating" in s:
                inside = True
                continue
            if "End of chirality of Weyl points calculating" in s:
                inside = False
                continue
            if not inside or (not s) or s.startswith("#") or s.startswith(">") or s.startswith("<"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 7:
                continue
            vals = [to_float(x) for x in parts[:7]]
            if any(np.isnan(v) for v in vals):
                continue
            out.append({
                "k1": wrap_frac_value(vals[0]),
                "k2": wrap_frac_value(vals[1]),
                "k3": wrap_frac_value(vals[2]),
                "kx": vals[3],
                "ky": vals[4],
                "kz": vals[5],
                "chirality": int(round(vals[6])),
            })
    return out


def attach_chirality_to_node(node_k: Tuple[float, float, float], chi_rows: List[dict]) -> Tuple[int, float]:
    if not chi_rows:
        return 0, np.nan

    best = None
    best_d = 1e99
    for r in chi_rows:
        kr = (r["k1"], r["k2"], r["k3"])
        d = periodic_distance(node_k, kr)
        if d < best_d:
            best_d = d
            best = r

    if best is None:
        return 0, np.nan

    if best_d > CHIRALITY_MATCH_K_TOL:
        return 0, best_d

    return int(best["chirality"]), float(best_d)


def normalize_energy_fields(row: dict) -> Tuple[float, float]:
    energy_raw = to_float(row.get("energy_raw_eV"))
    energy_rel_csv = to_float(row.get("energy_rel_ef_eV"))

    mode = ENERGY_RELATIVE_MODE.lower()
    if mode == "raw_is_relative":
        return energy_raw, energy_raw
    if mode == "csv_energy_rel":
        return energy_raw, energy_rel_csv

    if np.isfinite(energy_raw) and abs(energy_raw) < 2.0 and np.isfinite(energy_rel_csv) and abs(energy_rel_csv) > 2.0:
        return energy_raw, energy_raw

    if np.isfinite(energy_rel_csv):
        return energy_raw, energy_rel_csv

    return energy_raw, energy_raw


def load_nodes_from_wt(wt_root: str) -> List[NodeRow]:
    csv_path = os.path.join(wt_root, "all_findnodes_unique_nodes.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 all_findnodes_unique_nodes.csv: {csv_path}")

    rows = read_csv_dicts(csv_path)

    chi_cache: Dict[Tuple[str, int], List[dict]] = {}
    nodes: List[NodeRow] = []

    for r in rows:
        case_label = r.get("case_label") or r.get("case_dir") or ""
        if case_label not in TARGET_DIRS:
            continue

        num_occ = to_int(r.get("num_occupied"))
        if num_occ != NUM_OCC_TO_USE:
            continue

        direction, eps = parse_case_meta(case_label)
        energy_raw, energy_rel = normalize_energy_fields(r)

        k1 = wrap_frac_value(to_float(r.get("k1_frac")))
        k2 = wrap_frac_value(to_float(r.get("k2_frac")))
        k3 = wrap_frac_value(to_float(r.get("k3_frac")))

        key = (case_label, num_occ)
        if key not in chi_cache:
            chi_cache[key] = load_chirality_rows(wt_root, case_label, num_occ)

        chirality, chi_d = attach_chirality_to_node((k1, k2, k3), chi_cache[key])

        if REQUIRE_CHIRALITY_MATCH and (not np.isfinite(chi_d) or chi_d > CHIRALITY_MATCH_K_TOL):
            continue

        if REQUIRE_NONZERO_CHIRALITY and chirality == 0:
            continue

        node = NodeRow(
            case_dir=r.get("case_dir") or case_label,
            case_label=case_label,
            direction=direction,
            eps_percent=float(eps),

            num_occupied=num_occ,
            pair_n_1based=to_int(r.get("pair_n_1based"), num_occ),
            pair_m_1based=to_int(r.get("pair_m_1based"), num_occ + 1),

            kx_cart=to_float(r.get("kx_cart")),
            ky_cart=to_float(r.get("ky_cart")),
            kz_cart=to_float(r.get("kz_cart")),

            k1_frac=float(k1),
            k2_frac=float(k2),
            k3_frac=float(k3),

            gap_eV=to_float(r.get("gap_eV")),
            energy_raw_eV=float(energy_raw),
            energy_rel_ef_eV=float(energy_rel),
            fermi_energy_eV=to_float(r.get("fermi_energy_eV")),

            chirality=int(chirality),
            chirality_match_distance=float(chi_d),

            tilt_ratio=to_float(r.get("tilt_ratio")),
            type_label=str(r.get("type_label") or ""),
            singular_v1=to_float(r.get("singular_v1")),
            singular_v2=to_float(r.get("singular_v2")),
            singular_v3=to_float(r.get("singular_v3")),
            vx0=to_float(r.get("vx0")),
            vy0=to_float(r.get("vy0")),
            vz0=to_float(r.get("vz0")),
        )
        nodes.append(node)

    nodes.sort(key=lambda x: (x.eps_percent, x.chirality, x.k1_frac, x.k2_frac, x.k3_frac))
    return nodes


# ======================================================================
# Wannier HR 读取与局域线性化
# ======================================================================

def pick_win_file(case_path: str) -> Optional[str]:
    wins = sorted(glob.glob(os.path.join(case_path, "*.win")))
    if not wins:
        return None
    for w in wins:
        if os.path.basename(w).lower() == "wannier90.win":
            return w
    return wins[0]


def pick_hr_file(case_path: str) -> Optional[str]:
    hrs = sorted(glob.glob(os.path.join(case_path, "*_hr.dat")))
    if not hrs:
        return None
    for h in hrs:
        if os.path.basename(h).lower() == "wannier90_hr.dat":
            return h
    return hrs[0]


def parse_unit_cell_cart_from_win(win_path: Optional[str]) -> Optional[np.ndarray]:
    if win_path is None or not os.path.isfile(win_path):
        return None

    with open(win_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    begin = None
    end = None
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("begin unit_cell_cart"):
            begin = i
        elif s.startswith("end unit_cell_cart") and begin is not None:
            end = i
            break

    if begin is None or end is None:
        return None

    block = [x.strip() for x in lines[begin + 1:end] if x.strip()]
    if len(block) < 3:
        return None

    unit = "ang"
    start = 0
    if block[0].lower() in ("ang", "angstrom", "bohr"):
        unit = block[0].lower()
        start = 1

    if len(block) < start + 3:
        return None

    lat = np.zeros((3, 3), dtype=float)
    for i in range(3):
        vals = block[start + i].split()
        lat[i, :] = [float(vals[0]), float(vals[1]), float(vals[2])]

    if unit == "bohr":
        lat *= 0.529177210903

    return lat


def reciprocal_lattice_from_real(real_lat: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(real_lat).T


def frac_k_to_cart(k_frac: np.ndarray, recip_lat: np.ndarray) -> np.ndarray:
    return np.asarray(k_frac, dtype=float) @ recip_lat


def read_wannier90_hr(hr_path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    with open(hr_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise RuntimeError(f"{hr_path} 内容过短，无法解析。")

    num_wann = int(lines[1].strip())
    nrpts = int(lines[2].strip())

    ndeglines = int(math.ceil(nrpts / 15.0))
    deg_tokens = []
    for i in range(3, 3 + ndeglines):
        deg_tokens.extend(lines[i].split())

    if len(deg_tokens) < nrpts:
        raise RuntimeError(f"{hr_path} degeneracy 数据不足。")

    degs = np.array([int(x) for x in deg_tokens[:nrpts]], dtype=int)

    data_start = 3 + ndeglines
    expected = nrpts * num_wann * num_wann
    data_lines = lines[data_start:data_start + expected]

    if len(data_lines) < expected:
        raise RuntimeError(f"{hr_path} hr 数据行数不足。")

    Rvecs = np.zeros((nrpts, 3), dtype=int)
    HR_div = np.zeros((nrpts, num_wann, num_wann), dtype=np.complex128)

    idx = 0
    for ir in range(nrpts):
        deg = degs[ir]
        for _ in range(num_wann * num_wann):
            parts = data_lines[idx].split()
            idx += 1

            rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
            m = int(parts[3]) - 1
            n = int(parts[4]) - 1
            re = float(parts[5])
            im = float(parts[6])

            if m == 0 and n == 0:
                Rvecs[ir, :] = [rx, ry, rz]

            HR_div[ir, m, n] = (re + 1j * im) / deg

    return num_wann, Rvecs, HR_div


def h_and_dh(k_frac: np.ndarray, Rvecs: np.ndarray, HR_div: np.ndarray):
    phase_arg = 2.0 * np.pi * (Rvecs @ k_frac)
    phase = np.exp(1j * phase_arg)

    H = np.tensordot(phase, HR_div, axes=(0, 0))

    dH = []
    for iax in range(3):
        coef = 1j * 2.0 * np.pi * Rvecs[:, iax] * phase
        dH_i = np.tensordot(coef, HR_div, axes=(0, 0))
        dH.append(dH_i)

    return H, dH


def eigensystem_at_k(k_frac: np.ndarray, Rvecs: np.ndarray, HR_div: np.ndarray):
    H, dH = h_and_dh(k_frac, Rvecs, HR_div)
    evals, evecs = np.linalg.eigh(H)
    return evals.real, evecs, H, dH


def projected_pauli_coeffs(M2: np.ndarray) -> Tuple[float, float, float, float]:
    d0 = 0.5 * np.real(M2[0, 0] + M2[1, 1])
    dz = 0.5 * np.real(M2[0, 0] - M2[1, 1])
    dx = np.real(M2[0, 1])
    dy = -np.imag(M2[0, 1])
    return float(d0), float(dx), float(dy), float(dz)


def analyze_two_band_linearization(pair_n: int, pair_m: int, evecs: np.ndarray, dH_list: List[np.ndarray]) -> Dict[str, object]:
    U = evecs[:, [pair_n, pair_m]]

    B = np.zeros((3, 3), dtype=float)
    v0 = np.zeros(3, dtype=float)

    for iax in range(3):
        M2 = U.conj().T @ dH_list[iax] @ U
        d0, dx, dy, dz = projected_pauli_coeffs(M2)
        v0[iax] = d0
        B[:, iax] = np.array([dx, dy, dz], dtype=float)

    detB = float(np.linalg.det(B))
    if detB > 1e-12:
        linear_chi = +1
    elif detB < -1e-12:
        linear_chi = -1
    else:
        linear_chi = 0

    _, svals, Vt = np.linalg.svd(B, full_matrices=True)
    Vdirs = Vt.T

    G = B.T @ B
    tilt_ratio = np.nan
    try:
        if np.linalg.cond(G) < 1e12:
            tilt_ratio = float(np.sqrt(v0.T @ np.linalg.inv(G) @ v0))
    except np.linalg.LinAlgError:
        pass

    type_label = "type-II-like" if np.isfinite(tilt_ratio) and tilt_ratio > TILT_TYPE2_THRESHOLD else "type-I-like"

    return {
        "linear_chirality": linear_chi,
        "B": B,
        "v0": v0,
        "svals": svals,
        "Vdirs": Vdirs,
        "tilt_ratio": tilt_ratio,
        "type_label": type_label,
    }


CASE_HR_CACHE: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = {}
ACTIVE_WT_ROOT = ""


def get_case_hr_pack(case_label: str) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    读取指定 case 的 wannier90_hr.dat，并尽量读取晶格。

    k-line 导出只需要 HR 文件；不再依赖 1/Å 坐标，因此即使没有 .win 晶格，
    也可以正常输出 fractional Δk 的 Origin 表格。

    搜索优先级：
      1) CASE_INPUT_ROOT/<case>
      2) ROOT_DIR/<case>
      3) ACTIVE_WT_ROOT/<case>
      4) ACTIVE_WT_ROOT/<case>/wt_find_Nxx
      5) ACTIVE_WT_ROOT/<case>/wt_chirality_Nxx
    """
    if case_label in CASE_HR_CACHE:
        return CASE_HR_CACHE[case_label]

    with timed_section(f"load HR model | {case_label}"):
        candidate_dirs = []

        def add_dir(d: str):
            if d and d not in candidate_dirs:
                candidate_dirs.append(d)

        add_dir(os.path.join(CASE_INPUT_ROOT, case_label))
        add_dir(os.path.join(ROOT_DIR, case_label))

        if ACTIVE_WT_ROOT:
            wt_case_dir = os.path.join(ACTIVE_WT_ROOT, case_label)
            add_dir(wt_case_dir)
            add_dir(os.path.join(wt_case_dir, f"wt_find_N{NUM_OCC_TO_USE}"))
            add_dir(os.path.join(wt_case_dir, f"wt_chirality_N{NUM_OCC_TO_USE}"))

            # 兜底：用户如果修改过 NUM_OCC_TO_USE 或目录名，扫描 wt_find_N*/wt_chirality_N*。
            for d in sorted(glob.glob(os.path.join(wt_case_dir, "wt_find_N*"))):
                add_dir(d)
            for d in sorted(glob.glob(os.path.join(wt_case_dir, "wt_chirality_N*"))):
                add_dir(d)

        hr_path = None
        hr_dir = None
        for d in candidate_dirs:
            h = pick_hr_file(d)
            if h is not None:
                hr_path = h
                hr_dir = d
                break

        if hr_path is None:
            print(f"[WARN] 未找到 HR 文件: {case_label}")
            print("       已搜索目录:")
            for d in candidate_dirs:
                print(f"       - {d}")
            CASE_HR_CACHE[case_label] = None
            return None

        # 晶格只用于旧的 1/Å 输出；当前 Origin 表默认不需要 perA。
        # 这里仍尽量读取，以兼容脚本中保留的 3D 曲面等旧功能。
        real_lat = None
        for d in candidate_dirs:
            win_path = pick_win_file(d)
            real_lat = parse_unit_cell_cart_from_win(win_path)
            if real_lat is not None:
                break

        recip_lat = reciprocal_lattice_from_real(real_lat) if real_lat is not None else None
        _, Rvecs, HR_div = read_wannier90_hr(hr_path)
        CASE_HR_CACHE[case_label] = (Rvecs, HR_div, recip_lat)

        if PRINT_SUMMARY:
            lat_msg = "with lattice" if recip_lat is not None else "without lattice; fractional k output only"
            print(f"[HR] {case_label}: {hr_path} ({lat_msg})")

        return CASE_HR_CACHE[case_label]


def enrich_nodes_with_hr_quantities(nodes: List[NodeRow]) -> List[NodeRow]:
    if not COMPUTE_TILT_FROM_HR and (not PLOT_WEYL_CONE_3D) and (not PLOT_WEYL_CONE_CUTS) and (not PLOT_PAIR_KLINE_BANDS) and (not PLOT_PAIR_ENERGY_SURFACE_3D):
        return nodes

    out = []
    for nd in nodes:
        pack = get_case_hr_pack(nd.case_label)
        if pack is None:
            out.append(nd)
            continue

        Rvecs, HR_div, _ = pack
        k0 = np.array([nd.k1_frac, nd.k2_frac, nd.k3_frac], dtype=float)
        pair_n = nd.pair_n_1based - 1
        pair_m = nd.pair_m_1based - 1

        try:
            _, evecs, _, dH = eigensystem_at_k(k0, Rvecs, HR_div)
            ana = analyze_two_band_linearization(pair_n, pair_m, evecs, dH)
            nd.tilt_ratio = float(ana["tilt_ratio"]) if np.isfinite(ana["tilt_ratio"]) else np.nan
            nd.type_label = str(ana["type_label"])
            svals = ana["svals"]
            nd.singular_v1 = float(svals[0])
            nd.singular_v2 = float(svals[1])
            nd.singular_v3 = float(svals[2])
            v0 = ana["v0"]
            nd.vx0 = float(v0[0])
            nd.vy0 = float(v0[1])
            nd.vz0 = float(v0[2])
        except Exception as e:
            print(f"[WARN] tilt 计算失败: {nd.case_label} k=({nd.k1_frac:.4f},{nd.k2_frac:.4f},{nd.k3_frac:.4f}) | {e}")

        out.append(nd)

    return out


# ======================================================================
# family 追踪
# ======================================================================

def node_cost(prev: NodeRow, cur: NodeRow) -> float:
    dk = periodic_distance(
        (prev.k1_frac, prev.k2_frac, prev.k3_frac),
        (cur.k1_frac, cur.k2_frac, cur.k3_frac),
    )
    de = abs(prev.energy_rel_ef_eV - cur.energy_rel_ef_eV)
    if not np.isfinite(de):
        de = 0.0
    dg = abs(prev.gap_eV - cur.gap_eV)
    if not np.isfinite(dg):
        dg = 0.0

    chi_penalty = 0.0
    if prev.chirality != cur.chirality:
        if MATCH_REQUIRE_SAME_CHIRALITY:
            return 1e9
        chi_penalty = MATCH_CHIRALITY_PENALTY

    return MATCH_K_WEIGHT * dk + MATCH_E_WEIGHT * (de / 0.1) + MATCH_GAP_WEIGHT * (dg / 1e-3) + chi_penalty


def solve_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        rr, cc = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rr, cc)]
    except Exception:
        pairs = []
        used_r = set()
        used_c = set()
        flat = []
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                flat.append((cost[i, j], i, j))
        flat.sort(key=lambda x: x[0])
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
        return pairs


def make_track_row(path_name: str, family_id: str, nd: NodeRow, k_unwrapped: np.ndarray) -> TrackRow:
    return TrackRow(
        path=path_name,
        family_id=family_id,
        case_label=nd.case_label,
        direction=nd.direction,
        eps_percent=nd.eps_percent,

        num_occupied=nd.num_occupied,
        pair_n_1based=nd.pair_n_1based,
        pair_m_1based=nd.pair_m_1based,

        chirality=nd.chirality,

        k1_frac=nd.k1_frac,
        k2_frac=nd.k2_frac,
        k3_frac=nd.k3_frac,

        k1_unwrapped=float(k_unwrapped[0]),
        k2_unwrapped=float(k_unwrapped[1]),
        k3_unwrapped=float(k_unwrapped[2]),

        energy_rel_ef_eV=nd.energy_rel_ef_eV,
        gap_eV=nd.gap_eV,
        tilt_ratio=nd.tilt_ratio,
        type_label=nd.type_label,

        source_case_dir=nd.case_dir,
    )


def track_one_path(nodes: List[NodeRow], path_name: str, case_sequence: List[str]) -> List[TrackRow]:
    nodes_by_case = {c: [] for c in case_sequence}
    for nd in nodes:
        if nd.case_label in nodes_by_case:
            nodes_by_case[nd.case_label].append(nd)

    existing_cases = [c for c in case_sequence if nodes_by_case.get(c)]
    if not existing_cases:
        return []

    first_case = existing_cases[0]
    first_nodes = sorted(
        nodes_by_case[first_case],
        key=lambda x: (x.chirality, x.k1_frac, x.k2_frac, x.k3_frac)
    )

    family_last: Dict[str, NodeRow] = {}
    family_last_raw_k: Dict[str, np.ndarray] = {}
    family_last_unwrapped_k: Dict[str, np.ndarray] = {}
    rows: List[TrackRow] = []

    for i, nd in enumerate(first_nodes, 1):
        fid = f"F{i}"
        kraw = np.array([nd.k1_frac, nd.k2_frac, nd.k3_frac], dtype=float)
        kunwrap = center_frac_vector(kraw) if CENTER_K_COORDINATES else kraw.copy()

        family_last[fid] = nd
        family_last_raw_k[fid] = kraw
        family_last_unwrapped_k[fid] = kunwrap.copy()
        rows.append(make_track_row(path_name, fid, nd, kunwrap))

    next_family_index = len(first_nodes) + 1

    for case in existing_cases[1:]:
        cur_nodes = sorted(nodes_by_case[case], key=lambda x: (x.chirality, x.k1_frac, x.k2_frac, x.k3_frac))
        fids = list(family_last.keys())
        prev_nodes = [family_last[fid] for fid in fids]

        if not cur_nodes:
            continue

        cost = np.zeros((len(prev_nodes), len(cur_nodes)), dtype=float)
        for i, p in enumerate(prev_nodes):
            for j, c in enumerate(cur_nodes):
                cost[i, j] = node_cost(p, c)

        assignments = solve_assignment(cost)

        used_cur = set()
        updated = {}

        for i, j in assignments:
            if cost[i, j] > MAX_MATCH_COST:
                continue

            fid = fids[i]
            nd = cur_nodes[j]
            used_cur.add(j)

            kraw = np.array([nd.k1_frac, nd.k2_frac, nd.k3_frac], dtype=float)
            prev_raw = family_last_raw_k[fid]
            prev_unwrapped = family_last_unwrapped_k[fid]
            delta = minimal_image_delta(kraw, prev_raw)
            kunwrap = prev_unwrapped + delta

            rows.append(make_track_row(path_name, fid, nd, kunwrap))

            updated[fid] = nd
            family_last_raw_k[fid] = kraw
            family_last_unwrapped_k[fid] = kunwrap

        for j, nd in enumerate(cur_nodes):
            if j in used_cur:
                continue

            fid = f"F{next_family_index}"
            next_family_index += 1

            kraw = np.array([nd.k1_frac, nd.k2_frac, nd.k3_frac], dtype=float)
            kunwrap = center_frac_vector(kraw) if CENTER_K_COORDINATES else kraw.copy()

            rows.append(make_track_row(path_name, fid, nd, kunwrap))

            updated[fid] = nd
            family_last_raw_k[fid] = kraw
            family_last_unwrapped_k[fid] = kunwrap.copy()

        family_last.update(updated)

    rows.sort(key=lambda x: (x.family_id, x.eps_percent, x.case_label))
    return rows


def group_tracks_by_family(rows: List[TrackRow]) -> Dict[str, List[TrackRow]]:
    d: Dict[str, List[TrackRow]] = {}
    for r in rows:
        d.setdefault(r.family_id, []).append(r)
    for fid in d:
        d[fid].sort(key=lambda x: x.eps_percent)
    return d


def family_ids_present_in_all_cases(rows: List[TrackRow], required_cases: List[str]) -> Set[str]:
    fams = group_tracks_by_family(rows)
    required = set(required_cases)
    keep = set()
    for fid, frs in fams.items():
        cases = set(r.case_label for r in frs)
        if required.issubset(cases):
            keep.add(fid)
    return keep


def split_persistent_and_transient(rows: List[TrackRow], required_cases: List[str]) -> Tuple[List[TrackRow], List[TrackRow], Set[str]]:
    keep_fids = family_ids_present_in_all_cases(rows, required_cases)
    persistent = [r for r in rows if r.family_id in keep_fids and r.case_label in required_cases]

    fams = group_tracks_by_family(rows)
    transient_fids = []
    for fid, frs in fams.items():
        if fid not in keep_fids and len(frs) >= TRANSIENT_MIN_FAMILY_LENGTH:
            transient_fids.append(fid)

    transient = [r for r in rows if r.family_id in set(transient_fids)]
    return persistent, transient, keep_fids


def summarize_families(track_rows: List[TrackRow]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[TrackRow]] = {}
    for r in track_rows:
        groups.setdefault((r.path, r.family_id), []).append(r)

    out = []
    for (path, fid), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda x: x.eps_percent)
        chis = sorted(set(r.chirality for r in rows))
        pairs = sorted(set((r.pair_n_1based, r.pair_m_1based) for r in rows))
        energies = [r.energy_rel_ef_eV for r in rows if np.isfinite(r.energy_rel_ef_eV)]
        tilts = [r.tilt_ratio for r in rows if np.isfinite(r.tilt_ratio)]
        gaps = [r.gap_eV for r in rows if np.isfinite(r.gap_eV)]

        out.append({
            "path": path,
            "family_id": fid,
            "length": len(rows),
            "start_case": rows[0].case_label,
            "end_case": rows[-1].case_label,
            "eps_min": min(r.eps_percent for r in rows),
            "eps_max": max(r.eps_percent for r in rows),
            "chirality_set": ";".join(str(x) for x in chis),
            "pair_set": ";".join(f"{a}-{b}" for a, b in pairs),
            "mean_energy_rel_ef_eV": float(np.mean(energies)) if energies else np.nan,
            "mean_abs_energy_rel_ef_eV": float(np.mean(np.abs(energies))) if energies else np.nan,
            "min_gap_eV": float(np.min(gaps)) if gaps else np.nan,
            "mean_tilt_ratio": float(np.mean(tilts)) if tilts else np.nan,
        })

    return out


# ======================================================================
# 统计图
# ======================================================================

def case_order_pairs() -> List[Tuple[str, float]]:
    out = []
    for c in TARGET_DIRS:
        _, eps = parse_case_meta(c)
        out.append((c, eps))
    out.sort(key=lambda x: x[1])
    return out


def plot_count_vs_strain(nodes: List[NodeRow], out_png: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = []
    yt = []
    for c, eps in case_order_pairs():
        sub = [nd for nd in nodes if nd.case_label == c]
        xs.append(eps)
        yt.append(len(sub))

    ax.plot(xs, yt, "o-", linewidth=1.8, markersize=6, label="chirality-confirmed N56 Weyl nodes")
    ax.set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    ax.set_ylabel("number of Weyl nodes")
    ax.set_title("Weyl node count vs c-axis strain")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def plot_chirality_count_vs_strain(nodes: List[NodeRow], out_png: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = []
    yp = []
    yn = []
    yz = []
    yt = []
    for c, eps in case_order_pairs():
        sub = [nd for nd in nodes if nd.case_label == c]
        xs.append(eps)
        yp.append(sum(1 for nd in sub if nd.chirality > 0))
        yn.append(sum(1 for nd in sub if nd.chirality < 0))
        yz.append(sum(1 for nd in sub if nd.chirality == 0))
        yt.append(len(sub))

    ax.plot(xs, yt, "k-o", linewidth=1.6, markersize=5, label="total")
    ax.plot(xs, yp, "o--", color=POS_COLOR, linewidth=1.4, markersize=5, label=r"$\chi=+1$")
    ax.plot(xs, yn, "^--", color=NEG_COLOR, linewidth=1.4, markersize=5, label=r"$\chi=-1$")
    if any(v > 0 for v in yz):
        ax.plot(xs, yz, "s:", color=ZERO_COLOR, linewidth=1.2, markersize=5, label=r"$\chi=0$")
    ax.set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    ax.set_ylabel("node count")
    ax.set_title("Chirality-resolved Weyl node count vs c-axis strain")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def plot_energy_scatter(nodes: List[NodeRow], out_png: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    xs_pos, ys_pos = [], []
    xs_neg, ys_neg = [], []

    for nd in nodes:
        if nd.chirality > 0:
            xs_pos.append(nd.eps_percent)
            ys_pos.append(nd.energy_rel_ef_eV)
        elif nd.chirality < 0:
            xs_neg.append(nd.eps_percent)
            ys_neg.append(nd.energy_rel_ef_eV)

    if USE_CHIRALITY_COLOR_FOR_NODE_MAPS:
        if xs_pos:
            ax.scatter(xs_pos, ys_pos, c=POS_COLOR, marker=MARKER_POS, s=52,
                       edgecolors="k", linewidths=0.25, label=r"$\chi=+1$")
        if xs_neg:
            ax.scatter(xs_neg, ys_neg, c=NEG_COLOR, marker=MARKER_NEG, s=52,
                       edgecolors="k", linewidths=0.25, label=r"$\chi=-1$")
    else:
        xs = xs_pos + xs_neg
        ys = ys_pos + ys_neg
        ax.scatter(xs, ys, c=UNIFIED_NODE_COLOR, marker=UNIFIED_NODE_MARKER, s=52,
                   edgecolors="k", linewidths=0.25, label="individual Weyl nodes")

    ax.axhline(0.0, color="black", linewidth=0.8, label=r"$E_F$")
    ax.set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    ax.set_ylabel(r"$E_W-E_F$ (eV)")
    ax.set_title(r"Energy distribution of Weyl nodes vs c-axis strain")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# ======================================================================
# 轨迹图与标量图
# ======================================================================

def set_axis_labels_3d(ax):
    label_suffix = "centered frac" if CENTER_K_COORDINATES else "frac"
    ax.set_xlabel(f"k1 ({label_suffix})")
    ax.set_ylabel(f"k2 ({label_suffix})")
    ax.set_zlabel(f"k3 ({label_suffix})")


def plot_3d_trajectory(rows: List[TrackRow], out_png: str, unwrapped: bool, title: str):
    fams = group_tracks_by_family(rows)
    color_map = make_distinct_color_map(fams.keys())

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    all_x, all_y, all_z = [], [], []

    for fid, frs in fams.items():
        if len(frs) < 2:
            continue

        color = color_map.get(fid, UNIFIED_NODE_COLOR)

        if unwrapped:
            xs = [r.k1_unwrapped for r in frs]
            ys = [r.k2_unwrapped for r in frs]
            zs = [r.k3_unwrapped for r in frs]
        else:
            xs = [track_plot_coord(r, "k1_frac") for r in frs]
            ys = [track_plot_coord(r, "k2_frac") for r in frs]
            zs = [track_plot_coord(r, "k3_frac") for r in frs]

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)

        ax.plot(xs, ys, zs, "-", linewidth=1.35, color=color, alpha=0.85, label=fid)
        ax.scatter(xs, ys, zs, marker=UNIFIED_NODE_MARKER, s=34, color=color, edgecolors="k", linewidths=0.2)

    set_axis_labels_3d(ax)
    set_k_axis_limits_3d(ax, all_x, all_y, all_z)
    ax.set_title(title)
    add_family_legend(ax, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_k_components(rows: List[TrackRow], out_png: str, path_name: str, title_prefix: str):
    fams = group_tracks_by_family(rows)
    color_map = make_distinct_color_map(fams.keys())

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_TALL, sharex=True)

    comps = [
        ("k1_unwrapped", "k1"),
        ("k2_unwrapped", "k2"),
        ("k3_unwrapped", "k3"),
    ]

    for ax, (attr, ylabel) in zip(axes, comps):
        for fid, frs in fams.items():
            if len(frs) < 2:
                continue
            color = color_map.get(fid, UNIFIED_NODE_COLOR)
            xs = [r.eps_percent for r in frs]
            ys = [getattr(r, attr) for r in frs]
            ax.plot(xs, ys, marker=UNIFIED_NODE_MARKER, linestyle="-", linewidth=1.2,
                    markersize=4, color=color, alpha=0.9, label=fid)
        ax.set_ylabel(ylabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else ylabel + " (unwrapped frac)")
        ax.grid(True)

    axes[-1].set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=True)
    fig.suptitle(f"{title_prefix} path {path_name}: Weyl-node k components vs strain")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_vs_strain(
    rows: List[TrackRow],
    out_png: str,
    path_name: str,
    attr: str,
    ylabel: str,
    title_tail: str,
    semilogy: bool = False,
    title_prefix: str = "",
):
    fams = group_tracks_by_family(rows)
    color_map = make_distinct_color_map(fams.keys())

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for fid, frs in fams.items():
        if len(frs) < 2:
            continue
        color = color_map.get(fid, UNIFIED_NODE_COLOR)
        xs = np.array([r.eps_percent for r in frs], dtype=float)
        ys = np.array([getattr(r, attr) for r in frs], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            continue

        if semilogy:
            ax.semilogy(xs[mask], np.abs(ys[mask]), marker=UNIFIED_NODE_MARKER,
                        linestyle="-", linewidth=1.25, markersize=4, color=color,
                        alpha=0.9, label=fid)
        else:
            ax.plot(xs[mask], ys[mask], marker=UNIFIED_NODE_MARKER,
                    linestyle="-", linewidth=1.25, markersize=4, color=color,
                    alpha=0.9, label=fid)

    if attr == "energy_rel_ef_eV":
        ax.axhline(0.0, color="black", linewidth=0.8, label=r"$E_F$")
    ax.set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} path {path_name}: {title_tail}")
    ax.grid(True)
    add_family_legend(ax, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_k_projections_2d(rows: List[TrackRow], out_png: str, path_name: str, title_prefix: str):
    fams = group_tracks_by_family(rows)
    color_map = make_distinct_color_map(fams.keys())

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_TRIPLE, constrained_layout=True)

    pairs = [
        ("k1_unwrapped", "k2_unwrapped", "k1", "k2"),
        ("k1_unwrapped", "k3_unwrapped", "k1", "k3"),
        ("k2_unwrapped", "k3_unwrapped", "k2", "k3"),
    ]

    for ax, (xattr, yattr, xlabel, ylabel) in zip(axes, pairs):
        all_x, all_y = [], []
        for fid, frs in fams.items():
            if len(frs) < 2:
                continue
            color = color_map.get(fid, UNIFIED_NODE_COLOR)
            xs = [getattr(r, xattr) for r in frs]
            ys = [getattr(r, yattr) for r in frs]
            all_x.extend(xs)
            all_y.extend(ys)
            ax.plot(xs, ys, "-", linewidth=1.15, color=color, alpha=0.85, label=fid)
            ax.scatter(xs, ys, marker=UNIFIED_NODE_MARKER, s=28, color=color, edgecolors="k", linewidths=0.2)
        ax.set_xlabel(xlabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else xlabel + " (unwrapped frac)")
        ax.set_ylabel(ylabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else ylabel + " (unwrapped frac)")
        set_k_axis_limits_2d(ax, all_x, all_y)
        ax.grid(True)
        ax.set_title(f"{xlabel}-{ylabel}")
        add_family_legend(ax, fontsize=5)

    fig.suptitle(f"{title_prefix} path {path_name}: 2D projections of Weyl-node trajectories")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_k1k2_trajectory(rows: List[TrackRow], out_png: str, path_name: str, title_prefix: str):
    fams = group_tracks_by_family(rows)
    color_map = make_distinct_color_map(fams.keys())

    fig, ax = plt.subplots(figsize=FIGSIZE)

    all_x, all_y = [], []
    last_sc = None

    for fid, frs in fams.items():
        if len(frs) < 2:
            continue

        color = color_map.get(fid, UNIFIED_NODE_COLOR)

        xs = np.array([r.k1_unwrapped for r in frs], dtype=float)
        ys = np.array([r.k2_unwrapped for r in frs], dtype=float)
        eps = np.array([r.eps_percent for r in frs], dtype=float)

        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())

        ax.plot(xs, ys, "-", color=color, linewidth=1.25, alpha=0.90, label=fid)
        last_sc = ax.scatter(xs, ys, c=eps, cmap="viridis", marker=UNIFIED_NODE_MARKER,
                             s=58, edgecolors="k", linewidths=0.25)

    ax.set_xlabel("k1 (centered/unwrapped frac)" if CENTER_K_COORDINATES else "k1 (unwrapped frac)")
    ax.set_ylabel("k2 (centered/unwrapped frac)" if CENTER_K_COORDINATES else "k2 (unwrapped frac)")
    ax.set_title(f"{title_prefix} path {path_name}: Weyl-node trajectories in k1-k2 plane")
    set_k_axis_limits_2d(ax, all_x, all_y)
    ax.grid(True)
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=ax)
        cbar.set_label(r"$\epsilon_c$ (%)")
    add_family_legend(ax, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 新增：只画 -1%~+2% 区间内 4 对 Weyl 点的 -2%~+2% 面板图
# ======================================================================

def select_core_4pair_family_ids(rows: List[TrackRow]) -> List[str]:
    """
    自动选择“在 CORE_4PAIR_SELECTION_CASES 中一直存在”的 family。
    目标是得到 8 个 family，即 4 对 Weyl 点。

    若选出的 family 多于 8：
      优先按照 0% 结构中 |E_W-E_F| 更小的 family 排序，取前 8 个。
    若少于 8：
      不强行补点，只给出 warning。
    """
    fams = group_tracks_by_family(rows)
    required = set(CORE_4PAIR_SELECTION_CASES)

    candidates = []
    for fid, frs in fams.items():
        cases = set(r.case_label for r in frs)
        if not required.issubset(cases):
            continue

        ref_rows = [r for r in frs if r.case_label == "00_Optimized_Structure"]
        if ref_rows:
            score_energy = abs(ref_rows[0].energy_rel_ef_eV)
            score_k = np.linalg.norm([ref_rows[0].k1_unwrapped, ref_rows[0].k2_unwrapped, ref_rows[0].k3_unwrapped])
        else:
            score_energy = np.mean([abs(r.energy_rel_ef_eV) for r in frs if np.isfinite(r.energy_rel_ef_eV)])
            score_k = np.mean([np.linalg.norm([r.k1_unwrapped, r.k2_unwrapped, r.k3_unwrapped]) for r in frs])

        chi = frs[0].chirality if frs else 0
        candidates.append((score_energy, score_k, chi, family_sort_key(fid), fid))

    candidates.sort(key=lambda x: (x[0], x[1], x[3]))

    selected = [x[-1] for x in candidates]

    if len(selected) > CORE_4PAIR_EXPECTED_NODE_COUNT:
        selected = selected[:CORE_4PAIR_EXPECTED_NODE_COUNT]

    return selected


def get_trackrow_by_case_and_family(rows: List[TrackRow], case_label: str, family_id: str) -> Optional[TrackRow]:
    matches = [r for r in rows if r.case_label == case_label and r.family_id == family_id]
    if not matches:
        return None
    matches.sort(key=lambda r: abs(r.energy_rel_ef_eV))
    return matches[0]


def fallback_nearest_trackrow_for_case(
    rows: List[TrackRow],
    target_case: str,
    reference_row: TrackRow,
    used_keys: Set[Tuple[str, float, float, float]],
) -> Optional[TrackRow]:
    """
    可选兜底：
      如果某个 selected family 在 -2% 中缺失，则找 target_case 中与 reference_row 最近、
      且 chirality 相同的节点。
    默认不启用，避免混入未追踪确认的节点。
    """
    if not CORE_4PAIR_ALLOW_NEAREST_FALLBACK:
        return None

    candidates = []
    ref_k = np.array([reference_row.k1_frac, reference_row.k2_frac, reference_row.k3_frac], dtype=float)

    for r in rows:
        if r.case_label != target_case:
            continue
        if r.chirality != reference_row.chirality:
            continue
        key = (r.case_label, round(r.k1_frac, 8), round(r.k2_frac, 8), round(r.k3_frac, 8))
        if key in used_keys:
            continue
        cur_k = np.array([r.k1_frac, r.k2_frac, r.k3_frac], dtype=float)
        d = periodic_distance(ref_k, cur_k)
        if d <= CORE_4PAIR_FALLBACK_K_TOL:
            candidates.append((d, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def compute_shared_axis_limits_for_trackrows(rows: List[TrackRow]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [r.k1_unwrapped for r in rows if np.isfinite(r.k1_unwrapped)]
    ys = [r.k2_unwrapped for r in rows if np.isfinite(r.k2_unwrapped)]

    if CORE_4PAIR_AXIS_MODE.lower() == "fixed":
        return CORE_4PAIR_FIXED_XLIM, CORE_4PAIR_FIXED_YLIM

    if not xs or not ys:
        return (-CORE_4PAIR_MIN_HALF_WIDTH, CORE_4PAIR_MIN_HALF_WIDTH), (-CORE_4PAIR_MIN_HALF_WIDTH, CORE_4PAIR_MIN_HALF_WIDTH)

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = max(
        0.5 * (xmax - xmin),
        0.5 * (ymax - ymin),
        CORE_4PAIR_MIN_HALF_WIDTH,
    ) + CORE_4PAIR_AXIS_MARGIN

    return (cx - half, cx + half), (cy - half, cy + half)


def plot_selected_core_4pair_k1k2_maps(rows: List[TrackRow], out_dir: str, path_name: str = "c"):
    """
    在 selected_case_k1k2_maps 目录下输出：
      1) core_4pair_minus2_to_plus2_k1k2_maps.png
      2) core_4pair_minus2_to_plus2_nodes.csv

    图的含义：
      只显示 -1%、0%、+1%、+2% 区间内一直存在的 8 个 family，
      即 4 对稳定 N56 Weyl 点；
      在 -2% 面板中也只显示这 8 个 family 中能追踪到的点，
      不显示 -2% 压缩诱导的额外 transient Weyl 点。
    """
    ensure_dir(out_dir)

    selected_fids = select_core_4pair_family_ids(rows)
    selected_set = set(selected_fids)

    print("\n================ Core 4-pair k1-k2 map selection ================")
    print(f"selection cases: {CORE_4PAIR_SELECTION_CASES}")
    print(f"panel cases    : {CORE_4PAIR_PANEL_CASES}")
    print(f"selected family 数: {len(selected_fids)}")
    print("selected families:", ", ".join(selected_fids))
    if len(selected_fids) != CORE_4PAIR_EXPECTED_NODE_COUNT:
        print(
            f"[WARN] 期望 {CORE_4PAIR_EXPECTED_NODE_COUNT} 个 family "
            f"({CORE_4PAIR_EXPECTED_PAIR_COUNT} 对 Weyl 点)，实际选到 {len(selected_fids)} 个。"
        )
    print("=================================================================")

    # 整理用于绘图的 track rows。
    panel_rows: Dict[str, List[TrackRow]] = {case: [] for case in CORE_4PAIR_PANEL_CASES}
    export_rows = []
    used_keys: Set[Tuple[str, float, float, float]] = set()

    for case in CORE_4PAIR_PANEL_CASES:
        for fid in selected_fids:
            r = get_trackrow_by_case_and_family(rows, case, fid)

            # 若 -2% 中某个 family 没有直接追踪到，可选最邻近同手性节点兜底。
            if r is None and CORE_4PAIR_ALLOW_NEAREST_FALLBACK:
                ref = None
                for ref_case in CORE_4PAIR_SELECTION_CASES:
                    ref = get_trackrow_by_case_and_family(rows, ref_case, fid)
                    if ref is not None:
                        break
                if ref is not None:
                    r = fallback_nearest_trackrow_for_case(rows, case, ref, used_keys)

            if r is None:
                continue

            key = (r.case_label, round(r.k1_frac, 8), round(r.k2_frac, 8), round(r.k3_frac, 8))
            used_keys.add(key)
            panel_rows[case].append(r)

            export_rows.append({
                "path": path_name,
                "family_id": fid,
                "case_label": r.case_label,
                "eps_percent": r.eps_percent,
                "chirality": r.chirality,
                "k1_frac": r.k1_frac,
                "k2_frac": r.k2_frac,
                "k3_frac": r.k3_frac,
                "k1_centered_unwrapped": r.k1_unwrapped,
                "k2_centered_unwrapped": r.k2_unwrapped,
                "k3_centered_unwrapped": r.k3_unwrapped,
                "energy_rel_ef_eV": r.energy_rel_ef_eV,
                "gap_eV": r.gap_eV,
            })

    write_dict_csv(os.path.join(out_dir, "core_4pair_minus2_to_plus2_nodes.csv"), export_rows)

    all_plot_rows = []
    for case in CORE_4PAIR_PANEL_CASES:
        all_plot_rows.extend(panel_rows[case])

    if CORE_4PAIR_USE_SHARED_AXIS:
        xlim, ylim = compute_shared_axis_limits_for_trackrows(all_plot_rows)
    else:
        xlim, ylim = None, None

    fig, axes = plt.subplots(
        1,
        len(CORE_4PAIR_PANEL_CASES),
        figsize=FIGSIZE_CORE_4PAIR,
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    if len(CORE_4PAIR_PANEL_CASES) == 1:
        axes = [axes]

    for ax, case in zip(axes, CORE_4PAIR_PANEL_CASES):
        sub = panel_rows.get(case, [])
        _, eps = parse_case_meta(case)

        if not sub:
            ax.text(0.5, 0.5, "no selected\nN56 Weyl nodes", ha="center", va="center", transform=ax.transAxes)
        else:
            sub = sorted(sub, key=lambda r: (r.chirality, family_sort_key(r.family_id)))
            for r in sub:
                color = node_chirality_color(r.chirality)
                marker = node_chirality_marker(r.chirality)
                x = r.k1_unwrapped
                y = r.k2_unwrapped
                ax.scatter(x, y, c=color, marker=marker, s=78, edgecolors="k", linewidths=0.25)
                if CORE_4PAIR_ANNOTATE_FAMILY_ID:
                    ax.text(x, y, f" {r.family_id}", fontsize=7, color=color)

        ax.set_title(f"{eps:+g}%")
        ax.set_xlabel("k1 (centered frac)")
        ax.set_ylabel("k2 (centered frac)")
        ax.grid(True)

        if xlim is not None and ylim is not None:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            xs = [r.k1_unwrapped for r in sub]
            ys = [r.k2_unwrapped for r in sub]
            set_k_axis_limits_2d(ax, xs, ys)

        # 让横纵尺度一致，避免视觉上距离被压缩。
        ax.set_aspect("equal", adjustable="box")

    # 统一图例放在第一张图中，减少重复。
    handles = [
        Line2D([0], [0], marker=MARKER_POS, color="w", markerfacecolor=POS_COLOR,
               markeredgecolor="k", markersize=8, label=r"$\chi=+1$"),
        Line2D([0], [0], marker=MARKER_NEG, color="w", markerfacecolor=NEG_COLOR,
               markeredgecolor="k", markersize=8, label=r"$\chi=-1$"),
    ]
    axes[0].legend(handles=handles, loc="best", fontsize=8)

    fig.suptitle(
        "Selected four Weyl-node pairs in the N56 band pair under c-axis strain\n"
        "Only families persistent from -1% to +2% are shown",
        fontsize=12,
    )

    out_png = os.path.join(out_dir, "core_4pair_minus2_to_plus2_k1k2_maps.png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 配对与节点位置图
# ======================================================================

def pair_families_by_chirality(rows: List[TrackRow]) -> List[Tuple[str, str, float, int]]:
    fams = group_tracks_by_family(rows)
    pos = [fid for fid, frs in fams.items() if frs and frs[0].chirality > 0 and len(frs) >= MIN_FAMILY_LENGTH_FOR_PAIR_PLOT]
    neg = [fid for fid, frs in fams.items() if frs and frs[0].chirality < 0 and len(frs) >= MIN_FAMILY_LENGTH_FOR_PAIR_PLOT]

    candidates = []
    for fp in pos:
        rp = fams[fp]
        eps_p = {round(r.eps_percent, 10): r for r in rp}
        for fn in neg:
            rn = fams[fn]
            eps_n = {round(r.eps_percent, 10): r for r in rn}
            common = sorted(set(eps_p.keys()) & set(eps_n.keys()))
            if len(common) < MIN_FAMILY_LENGTH_FOR_PAIR_PLOT:
                continue
            ds = []
            for e in common:
                a = eps_p[e]
                b = eps_n[e]
                d = np.linalg.norm([
                    a.k1_unwrapped - b.k1_unwrapped,
                    a.k2_unwrapped - b.k2_unwrapped,
                    a.k3_unwrapped - b.k3_unwrapped,
                ])
                ds.append(d)
            candidates.append((float(np.mean(ds)), -len(common), fp, fn, len(common)))

    candidates.sort(key=lambda x: (x[0], x[1]))

    used_p = set()
    used_n = set()
    pairs = []
    for score, neg_ncommon, fp, fn, ncommon in candidates:
        if fp in used_p or fn in used_n:
            continue
        used_p.add(fp)
        used_n.add(fn)
        pairs.append((fp, fn, score, ncommon))

    return pairs


def pair_nodes_within_case(nodes: List[NodeRow], case_label: str, max_pairs: int) -> List[Tuple[NodeRow, NodeRow, float]]:
    sub = [nd for nd in nodes if nd.case_label == case_label]
    pos = [nd for nd in sub if nd.chirality > 0]
    neg = [nd for nd in sub if nd.chirality < 0]
    cand = []
    for a in pos:
        for b in neg:
            d = periodic_distance((a.k1_frac, a.k2_frac, a.k3_frac), (b.k1_frac, b.k2_frac, b.k3_frac))
            cand.append((d, a, b))
    cand.sort(key=lambda x: x[0])
    used_pos = set()
    used_neg = set()
    out = []
    for d, a, b in cand:
        ka = (round(a.k1_frac, 8), round(a.k2_frac, 8), round(a.k3_frac, 8), a.chirality)
        kb = (round(b.k1_frac, 8), round(b.k2_frac, 8), round(b.k3_frac, 8), b.chirality)
        if ka in used_pos or kb in used_neg:
            continue
        used_pos.add(ka)
        used_neg.add(kb)
        out.append((a, b, d))
        if len(out) >= max_pairs:
            break
    return out


def plot_chirality_pair_comparison(rows: List[TrackRow], out_png: str, path_name: str):
    fams = group_tracks_by_family(rows)
    pairs = pair_families_by_chirality(rows)
    if not pairs:
        return

    pair_labels = [f"{fp}-{fn}" for fp, fn, _, _ in pairs]
    color_map = make_distinct_color_map(pair_labels)

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_TALL, sharex=True)

    for fp, fn, score, ncommon in pairs:
        rp = {round(r.eps_percent, 10): r for r in fams[fp]}
        rn = {round(r.eps_percent, 10): r for r in fams[fn]}
        common = sorted(set(rp.keys()) & set(rn.keys()))

        xs = []
        dk = []
        de = []
        dt = []
        for e in common:
            a = rp[e]
            b = rn[e]
            xs.append(e)
            dk.append(np.linalg.norm([
                a.k1_unwrapped - b.k1_unwrapped,
                a.k2_unwrapped - b.k2_unwrapped,
                a.k3_unwrapped - b.k3_unwrapped,
            ]))
            de.append(a.energy_rel_ef_eV - b.energy_rel_ef_eV)
            if np.isfinite(a.tilt_ratio) and np.isfinite(b.tilt_ratio):
                dt.append(a.tilt_ratio - b.tilt_ratio)
            else:
                dt.append(np.nan)

        label = f"{fp}-{fn}"
        color = color_map.get(label, UNIFIED_NODE_COLOR)
        axes[0].plot(xs, dk, "o-", linewidth=1.2, color=color, label=label)
        axes[1].plot(xs, de, "o-", linewidth=1.2, color=color, label=label)
        axes[2].plot(xs, dt, "o-", linewidth=1.2, color=color, label=label)

    axes[0].set_ylabel(r"$|\Delta k|$ (centered/unwrapped frac)")
    axes[1].set_ylabel(r"$\Delta(E_W-E_F)$ (eV)")
    axes[2].set_ylabel(r"$\Delta$ tilt_ratio")
    axes[2].set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    for ax in axes:
        ax.grid(True)
        add_family_legend(ax, fontsize=7)
    fig.suptitle(f"path {path_name}: paired family comparison")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def build_case_reciprocal_lattice_cache() -> Dict[str, Optional[np.ndarray]]:
    cache = {}
    for case in TARGET_DIRS:
        case_path = os.path.join(CASE_INPUT_ROOT, case)
        win_path = pick_win_file(case_path)
        real_lat = parse_unit_cell_cart_from_win(win_path)
        recip = reciprocal_lattice_from_real(real_lat) if real_lat is not None else None
        cache[case] = recip
    return cache



def export_origin_pair_distance_tables(pair_rows: List[dict], path_name: str):
    """
    输出 Origin 重绘 Weyl-pair distance vs strain 所需表格。

    现在只导出 centered fractional k 距离，不再导出 1/Å/perA 列。

    long 表：
      每一行是一个 pair 在一个应变点的数据。

    wide 表：
      每一行是一个应变点；不同 pair 的 distance 分列，适合 Origin 直接画多条曲线。
    """
    if not EXPORT_ORIGIN_TABLES or not EXPORT_ORIGIN_PAIR_DISTANCE_TABLES:
        return

    out_dir = ORIGIN_OUT_DIR
    ensure_dir(out_dir)

    if not pair_rows:
        print(f"[WARN] Origin pair-distance 数据为空: path {path_name}")
        write_origin_table(os.path.join(out_dir, "pair_distance_vs_strain_long"), [])
        write_origin_table(os.path.join(out_dir, "pair_distance_vs_strain_wide"), [])
        write_origin_table(os.path.join(out_dir, f"path_{path_name}_pair_distance_vs_strain_long"), [])
        write_origin_table(os.path.join(out_dir, f"path_{path_name}_pair_distance_vs_strain_wide"), [])
        return

    long_rows = []
    for r in pair_rows:
        pair_index = int(r["pair_index"])
        pair_label = f"pair_{pair_index:02d}_{r['family_1']}_{r['family_2']}"
        long_rows.append({
            "path": r["path"],
            "pair_index": pair_index,
            "pair_label": pair_label,
            "family_1": r["family_1"],
            "family_2": r["family_2"],
            "case_label": r["case_label"],
            "eps_percent": r["eps_percent"],
            "distance_centered_frac": r["distance_frac"],
            "E1_minus_EF_eV": r["E1_minus_EF_eV"],
            "E2_minus_EF_eV": r["E2_minus_EF_eV"],
            "energy_separation_E1_minus_E2_eV": r["E1_minus_EF_eV"] - r["E2_minus_EF_eV"],
        })

    long_rows.sort(key=lambda x: (x["pair_index"], x["eps_percent"]))

    if EXPORT_ORIGIN_PAIR_DISTANCE_LONG_TABLE:
        write_origin_table(os.path.join(out_dir, "pair_distance_vs_strain_long"), long_rows)
        write_origin_table(os.path.join(out_dir, f"path_{path_name}_pair_distance_vs_strain_long"), long_rows)

    if EXPORT_ORIGIN_PAIR_DISTANCE_WIDE_TABLE:
        eps_values = sorted(set(r["eps_percent"] for r in long_rows))
        pair_indices = sorted(set(r["pair_index"] for r in long_rows))
        lookup = {(r["eps_percent"], r["pair_index"]): r for r in long_rows}

        wide_rows = []
        for eps in eps_values:
            row = {"eps_percent": eps}
            case_candidates = [r["case_label"] for r in long_rows if r["eps_percent"] == eps]
            row["case_label"] = case_candidates[0] if case_candidates else ""

            for pair_index in pair_indices:
                key = (eps, pair_index)
                r = lookup.get(key)
                prefix = f"pair_{pair_index:02d}"
                if r is None:
                    row[f"{prefix}_distance_centered_frac"] = np.nan
                    row[f"{prefix}_E1_minus_EF_eV"] = np.nan
                    row[f"{prefix}_E2_minus_EF_eV"] = np.nan
                    row[f"{prefix}_energy_separation_E1_minus_E2_eV"] = np.nan
                else:
                    row[f"{prefix}_distance_centered_frac"] = r["distance_centered_frac"]
                    row[f"{prefix}_E1_minus_EF_eV"] = r["E1_minus_EF_eV"]
                    row[f"{prefix}_E2_minus_EF_eV"] = r["E2_minus_EF_eV"]
                    row[f"{prefix}_energy_separation_E1_minus_E2_eV"] = r["energy_separation_E1_minus_E2_eV"]

            wide_rows.append(row)

        write_origin_table(os.path.join(out_dir, "pair_distance_vs_strain_wide"), wide_rows)
        write_origin_table(os.path.join(out_dir, f"path_{path_name}_pair_distance_vs_strain_wide"), wide_rows)

    print(f"[ORIGIN PAIR] path {path_name}: long_rows={len(long_rows)} | wide_eps={len(set(r['eps_percent'] for r in long_rows))}")

def plot_pair_distance_vs_strain(rows: List[TrackRow], out_frac: str, out_cart: str, path_name: str, title_prefix: str):
    """
    绘制并导出 Weyl-pair distance vs strain。

    默认只使用 centered/unwrapped fractional k 距离，不再输出 1/Å/perA 单位。
    out_cart 参数保留只是为了兼容主程序调用；默认不会写出 out_cart。
    """
    fams = group_tracks_by_family(rows)
    pairs = pair_families_by_chirality(rows)
    if not pairs:
        print(f"[WARN] path {path_name}: 没有可配对 family，pair-distance 图和 Origin 表不会生成。")
        export_origin_pair_distance_tables([], path_name)
        return

    pair_labels = [f"pair {i+1}: {fp}-{fn}" for i, (fp, fn, _, _) in enumerate(pairs)]
    color_map = make_distinct_color_map(pair_labels)

    fig1, ax1 = plt.subplots(figsize=FIGSIZE)

    fig2 = None
    ax2 = None
    recip_cache = None
    if PLOT_PAIR_DISTANCE_CART_FIGURE:
        recip_cache = build_case_reciprocal_lattice_cache()
        fig2, ax2 = plt.subplots(figsize=FIGSIZE)

    pair_rows = []

    for ipair, (fp, fn, score, ncommon) in enumerate(pairs, 1):
        rp = {round(r.eps_percent, 10): r for r in fams[fp]}
        rn = {round(r.eps_percent, 10): r for r in fams[fn]}
        common = sorted(set(rp.keys()) & set(rn.keys()))

        xs = []
        dfrac = []
        dcart = []

        for e in common:
            a = rp[e]
            b = rn[e]
            du = np.array([
                a.k1_unwrapped - b.k1_unwrapped,
                a.k2_unwrapped - b.k2_unwrapped,
                a.k3_unwrapped - b.k3_unwrapped,
            ], dtype=float)

            xs.append(e)
            df = float(np.linalg.norm(du))
            dfrac.append(df)

            if PLOT_PAIR_DISTANCE_CART_FIGURE and recip_cache is not None:
                recip = recip_cache.get(a.case_label)
                if recip is not None:
                    dk_cart = du @ recip
                    dc = float(np.linalg.norm(dk_cart))
                else:
                    dc = np.nan
                dcart.append(dc)

            pair_rows.append({
                "path": path_name,
                "pair_index": ipair,
                "family_1": fp,
                "family_2": fn,
                "case_label": a.case_label,
                "eps_percent": e,
                "distance_frac": df,
                "E1_minus_EF_eV": a.energy_rel_ef_eV,
                "E2_minus_EF_eV": b.energy_rel_ef_eV,
            })

        label = f"pair {ipair}: {fp}-{fn}"
        color = color_map.get(label, UNIFIED_NODE_COLOR)
        ax1.plot(xs, dfrac, "o-", linewidth=1.25, color=color, label=label)
        if PLOT_PAIR_DISTANCE_CART_FIGURE and ax2 is not None:
            ax2.plot(xs, dcart, "o-", linewidth=1.25, color=color, label=label)

    write_dict_csv(os.path.join(os.path.dirname(out_frac), f"path_{path_name}_pair_distance_table.csv"), pair_rows)

    # Origin pair-distance 数据输出与本绘图函数同步。
    # 只输出 centered fractional distance，不输出 perA/1Å 列。
    export_origin_pair_distance_tables(pair_rows, path_name)

    ax1.set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
    ax1.set_ylabel(r"$|\Delta k|$ (centered/unwrapped frac)")
    ax1.set_title(f"{title_prefix} path {path_name}: Weyl-pair distance vs strain")
    ax1.grid(True)
    add_family_legend(ax1, fontsize=7)

    fig1.tight_layout()
    fig1.savefig(out_frac, dpi=DPI, bbox_inches="tight")
    plt.close(fig1)

    if PLOT_PAIR_DISTANCE_CART_FIGURE and fig2 is not None and ax2 is not None:
        ax2.set_xlabel(r"c-axis strain $\epsilon_c$ (%)")
        ax2.set_ylabel(r"$|\Delta k|$ (1/$\AA$)")
        ax2.set_title(f"{title_prefix} path {path_name}: Weyl-pair distance vs strain (Cartesian)")
        ax2.grid(True)
        add_family_legend(ax2, fontsize=7)
        fig2.tight_layout()
        fig2.savefig(out_cart, dpi=DPI, bbox_inches="tight")
        plt.close(fig2)

def plot_case_node_maps(nodes: List[NodeRow], out_dir: str):
    ensure_dir(out_dir)

    proj_list = [
        ("k1_frac", "k2_frac", "k1", "k2", "k1k2"),
        ("k1_frac", "k3_frac", "k1", "k3", "k1k3"),
        ("k2_frac", "k3_frac", "k2", "k3", "k2k3"),
    ]

    by_case: Dict[str, List[NodeRow]] = {}
    for nd in nodes:
        by_case.setdefault(nd.case_label, []).append(nd)

    for case in TARGET_DIRS:
        sub = by_case.get(case, [])
        for xattr, yattr, xlabel, ylabel, tag in proj_list:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            xs_all, ys_all = [], []

            if sub:
                for nd in sub:
                    x = node_plot_coord(nd, xattr)
                    y = node_plot_coord(nd, yattr)
                    xs_all.append(x)
                    ys_all.append(y)
                    color = node_chirality_color(nd.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_COLOR
                    marker = node_chirality_marker(nd.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_MARKER
                    ax.scatter(x, y, c=color, marker=marker, s=70, edgecolors="k", linewidths=0.25)
            else:
                ax.text(0.5, 0.5, "no N56 Weyl nodes", ha="center", va="center", transform=ax.transAxes)

            _, eps = parse_case_meta(case)
            suffix = "centered frac" if CENTER_K_COORDINATES else "frac"
            ax.set_xlabel(f"{xlabel} ({suffix})")
            ax.set_ylabel(f"{ylabel} ({suffix})")
            ax.set_title(f"{case} ({eps:+g}%): Weyl-node positions ({tag})")
            ax.grid(True)
            set_k_axis_limits_2d(ax, xs_all, ys_all)
            if sub and USE_CHIRALITY_COLOR_FOR_NODE_MAPS:
                add_chirality_legend(ax)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{sanitize_name(case)}_{tag}.png"), dpi=DPI)
            plt.close(fig)


def plot_selected_case_k1k2_maps(nodes: List[NodeRow], out_dir: str, selected_cases: List[str]):
    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, len(selected_cases), figsize=FIGSIZE_PANEL, constrained_layout=True)
    if len(selected_cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, selected_cases):
        sub = [nd for nd in nodes if nd.case_label == case]
        _, eps = parse_case_meta(case)
        xs_all, ys_all = [], []

        if sub:
            for nd in sub:
                x = node_plot_coord(nd, "k1_frac")
                y = node_plot_coord(nd, "k2_frac")
                xs_all.append(x)
                ys_all.append(y)
                color = node_chirality_color(nd.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_COLOR
                marker = node_chirality_marker(nd.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_MARKER
                ax.scatter(x, y, c=color, marker=marker, s=75, edgecolors="k", linewidths=0.25)
        else:
            ax.text(0.5, 0.5, "no N56\nWeyl nodes", ha="center", va="center", transform=ax.transAxes)

        suffix = "centered frac" if CENTER_K_COORDINATES else "frac"
        ax.set_xlabel(f"k1 ({suffix})")
        ax.set_ylabel(f"k2 ({suffix})")
        set_k_axis_limits_2d(ax, xs_all, ys_all)
        ax.set_title(f"{eps:+g}%")
        ax.grid(True)
        if sub and USE_CHIRALITY_COLOR_FOR_NODE_MAPS:
            add_chirality_legend(ax)

    fig.suptitle("Representative k1-k2 Weyl-node maps under c-axis strain")
    fig.savefig(os.path.join(out_dir, "representative_k1k2_maps.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_selected_case_3d_maps(nodes: List[NodeRow], out_dir: str, selected_cases: List[str]):
    ensure_dir(out_dir)

    for case in selected_cases:
        sub = [nd for nd in nodes if nd.case_label == case]
        _, eps = parse_case_meta(case)

        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")

        xs_all, ys_all, zs_all = [], [], []

        if sub:
            for nd in sub:
                x = node_plot_coord(nd, "k1_frac")
                y = node_plot_coord(nd, "k2_frac")
                z = node_plot_coord(nd, "k3_frac")
                xs_all.append(x)
                ys_all.append(y)
                zs_all.append(z)
                color = node_chirality_color(nd.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_COLOR
                marker = node_chirality_marker(nd.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_MARKER
                ax.scatter(x, y, z, c=color, marker=marker, s=65, edgecolors="k", linewidths=0.25)
        else:
            ax.text2D(0.36, 0.50, "no N56 Weyl nodes", transform=ax.transAxes)

        set_axis_labels_3d(ax)
        set_k_axis_limits_3d(ax, xs_all, ys_all, zs_all)
        ax.set_title(f"{case} ({eps:+g}%): 3D Weyl-node positions")

        if sub and USE_CHIRALITY_COLOR_FOR_NODE_MAPS:
            handles = [
                Line2D([0], [0], marker=MARKER_POS, color="w", markerfacecolor=POS_COLOR,
                       markeredgecolor="k", markersize=8, label=r"$\chi=+1$"),
                Line2D([0], [0], marker=MARKER_NEG, color="w", markerfacecolor=NEG_COLOR,
                       markeredgecolor="k", markersize=8, label=r"$\chi=-1$"),
            ]
            ax.legend(handles=handles, loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{sanitize_name(case)}_3d_kmap.png"), dpi=DPI)
        plt.close(fig)


def plot_overlay_node_maps(rows: List[TrackRow], out_prefix: str, path_name: str, title_prefix: str):
    fams = group_tracks_by_family(rows)
    color_map = make_distinct_color_map(fams.keys())

    proj_list = [
        ("k1_unwrapped", "k2_unwrapped", "k1", "k2", "k1k2"),
        ("k1_unwrapped", "k3_unwrapped", "k1", "k3", "k1k3"),
        ("k2_unwrapped", "k3_unwrapped", "k2", "k3", "k2k3"),
    ]

    for xattr, yattr, xlabel, ylabel, tag in proj_list:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        all_x, all_y = [], []

        for fid, frs in fams.items():
            if len(frs) < 2:
                continue

            color = color_map.get(fid, UNIFIED_NODE_COLOR)
            xs = np.array([getattr(r, xattr) for r in frs], dtype=float)
            ys = np.array([getattr(r, yattr) for r in frs], dtype=float)

            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())

            ax.plot(xs, ys, "-", color=color, linewidth=1.25, alpha=0.90, label=fid)
            ax.scatter(xs, ys, color=color, marker=UNIFIED_NODE_MARKER, s=55,
                       edgecolors="k", linewidths=0.3)

        ax.set_xlabel(xlabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else xlabel + " (unwrapped frac)")
        ax.set_ylabel(ylabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else ylabel + " (unwrapped frac)")
        ax.set_title(f"{title_prefix} path {path_name}: overlay of Weyl-node positions ({tag})")
        set_k_axis_limits_2d(ax, all_x, all_y)
        ax.grid(True)
        add_family_legend(ax, fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{tag}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_schematic_like(rows: List[TrackRow], out_png: str, path_name: str, selected_cases: List[str]):
    fig, axes = plt.subplots(1, len(selected_cases), figsize=(5.0 * len(selected_cases), 4.2), constrained_layout=True)
    if len(selected_cases) == 1:
        axes = [axes]

    fams = group_tracks_by_family(rows)
    xattr, yattr, xlabel, ylabel = SCHEMATIC_PROJECTION

    for ax, case in zip(axes, selected_cases):
        case_rows = [r for r in rows if r.case_label == case]
        _, eps = parse_case_meta(case)

        all_x, all_y = [], []

        if not case_rows:
            ax.text(0.5, 0.5, "no N56\nWeyl nodes", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{eps:+g}%")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            continue

        for fid, frs in fams.items():
            sub = [r for r in frs if r.case_label == case]
            if not sub:
                continue
            r = sub[0]
            x = getattr(r, xattr)
            y = getattr(r, yattr)
            all_x.append(x)
            all_y.append(y)

            color = node_chirality_color(r.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_COLOR
            marker = node_chirality_marker(r.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_MARKER
            ax.scatter(x, y, c=color, marker=marker, s=95, edgecolors="k", linewidths=0.35)

        ref_rows = [r for r in rows if r.case_label == "00_Optimized_Structure"]
        ref_by_family = {r.family_id: r for r in ref_rows}
        cur_by_family = {r.family_id: r for r in case_rows}

        for fid, r0 in ref_by_family.items():
            r1 = cur_by_family.get(fid)
            if r1 is None or case == "00_Optimized_Structure":
                continue
            x0 = getattr(r0, xattr)
            y0 = getattr(r0, yattr)
            x1 = getattr(r1, xattr)
            y1 = getattr(r1, yattr)
            arrow_color = node_chirality_color(r1.chirality) if USE_CHIRALITY_COLOR_FOR_NODE_MAPS else UNIFIED_NODE_COLOR
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=0.9, alpha=0.65, color=arrow_color),
            )

        ax.set_title(f"{eps:+g}%")
        ax.set_xlabel(xlabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else xlabel + " (unwrapped frac)")
        ax.set_ylabel(ylabel + " (centered/unwrapped frac)" if CENTER_K_COORDINATES else ylabel + " (unwrapped frac)")
        set_k_axis_limits_2d(ax, all_x, all_y)
        ax.grid(True)
        if USE_CHIRALITY_COLOR_FOR_NODE_MAPS:
            add_chirality_legend(ax)

    fig.suptitle(f"path {path_name}: schematic-like evolution of N56 Weyl-node positions")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Weyl cone 3D、局部二维切线与点对局部能带图
# ======================================================================

def select_nodes_for_cone(path_rows_all: Dict[str, List[TrackRow]]) -> List[TrackRow]:
    selected: List[TrackRow] = []

    if CONE_NODE_SELECTION_MODE == "all":
        for rows in path_rows_all.values():
            for r in rows:
                if r.case_label in CONE_SELECTED_CASES:
                    selected.append(r)

    elif CONE_NODE_SELECTION_MODE == "manual":
        for path, fam_ids in CONE_MANUAL_SELECTION.items():
            rows = path_rows_all.get(path, [])
            for r in rows:
                if r.case_label in CONE_SELECTED_CASES and r.family_id in fam_ids:
                    selected.append(r)

    elif CONE_NODE_SELECTION_MODE == "one_pos_one_neg":
        for case in CONE_SELECTED_CASES:
            candidates = []
            for rows in path_rows_all.values():
                for r in rows:
                    if r.case_label == case:
                        candidates.append(r)

            unique = {}
            for r in candidates:
                key = (r.case_label, r.chirality, round(r.k1_frac, 8), round(r.k2_frac, 8), round(r.k3_frac, 8))
                unique[key] = r
            candidates = list(unique.values())

            pos = sorted([r for r in candidates if r.chirality > 0], key=lambda x: (abs(x.energy_rel_ef_eV), x.gap_eV))
            neg = sorted([r for r in candidates if r.chirality < 0], key=lambda x: (abs(x.energy_rel_ef_eV), x.gap_eV))
            pick = []
            if pos:
                pick.append(pos[0])
            if neg:
                pick.append(neg[0])
            selected.extend(pick[:CONE_MAX_NODES_PER_CASE])

    else:
        for case in CONE_SELECTED_CASES:
            candidates = []
            for rows in path_rows_all.values():
                for r in rows:
                    if r.case_label == case:
                        candidates.append(r)

            unique = {}
            for r in candidates:
                key = (r.case_label, round(r.k1_frac, 8), round(r.k2_frac, 8), round(r.k3_frac, 8))
                unique[key] = r
            candidates = list(unique.values())
            candidates = sorted(candidates, key=lambda x: (abs(x.energy_rel_ef_eV), x.gap_eV))
            selected.extend(candidates[:CONE_MAX_NODES_PER_CASE])

    out = []
    seen = set()
    for r in selected:
        key = (r.case_label, round(r.k1_frac, 8), round(r.k2_frac, 8), round(r.k3_frac, 8))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


def plot_single_cone_3d(out_png: str, node: TrackRow):
    pack = get_case_hr_pack(node.case_label)
    if pack is None:
        return
    Rvecs, HR_div, recip_lat = pack

    k0 = np.array([node.k1_frac, node.k2_frac, node.k3_frac], dtype=float)
    pair_n = node.pair_n_1based - 1
    pair_m = node.pair_m_1based - 1

    evals0, evecs, _, dH = eigensystem_at_k(k0, Rvecs, HR_div)
    ana = analyze_two_band_linearization(pair_n, pair_m, evecs, dH)
    Vdirs = ana["Vdirs"]
    e_node = 0.5 * (evals0[pair_n] + evals0[pair_m])

    for plane_pair in CONE_PLANE_LIST:
        ia, ib = plane_pair
        va = Vdirs[:, ia].astype(float)
        vb = Vdirs[:, ib].astype(float)
        va = va / np.linalg.norm(va)
        vb = vb / np.linalg.norm(vb)

        if recip_lat is not None:
            ga = frac_k_to_cart(va.reshape(1, 3), recip_lat).reshape(3)
            gb = frac_k_to_cart(vb.reshape(1, 3), recip_lat).reshape(3)
            sa = np.linalg.norm(ga)
            sb = np.linalg.norm(gb)
        else:
            sa = 1.0
            sb = 1.0

        q1 = np.linspace(-CONE_QMAX_FRAC, CONE_QMAX_FRAC, CONE_GRID_N)
        q2 = np.linspace(-CONE_QMAX_FRAC, CONE_QMAX_FRAC, CONE_GRID_N)
        Q1, Q2 = np.meshgrid(q1, q2, indexing="xy")

        X = Q1 * sa
        Y = Q2 * sb
        E1 = np.zeros_like(Q1)
        E2 = np.zeros_like(Q1)

        for i in range(CONE_GRID_N):
            for j in range(CONE_GRID_N):
                dq = Q1[i, j] * va + Q2[i, j] * vb
                kk = np.mod(k0 + dq, 1.0)
                ev, _, _, _ = eigensystem_at_k(kk, Rvecs, HR_div)
                if CONE_SHIFT_TO_NODE_ENERGY:
                    E1[i, j] = ev[pair_n] - e_node
                    E2[i, j] = ev[pair_m] - e_node
                else:
                    E1[i, j] = ev[pair_n]
                    E2[i, j] = ev[pair_m]

        fig = plt.figure(figsize=FIGSIZE_CONE)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, E1, alpha=0.72, linewidth=0.15, edgecolor=(0, 0, 0, 0.18))
        ax.plot_surface(X, Y, E2, alpha=0.72, linewidth=0.15, edgecolor=(0, 0, 0, 0.18))

        ax.set_xlabel(r"$q_%d$ (1/$\AA$)" % (ia + 1))
        ax.set_ylabel(r"$q_%d$ (1/$\AA$)" % (ib + 1))
        ax.set_zlabel(r"$E-E_{\mathrm{node}}$ (eV)" if CONE_SHIFT_TO_NODE_ENERGY else r"$E$ (eV)")
        ax.set_title(
            f"{node.case_label} | path {node.path} {node.family_id}\n"
            f"k=({node.k1_frac:.4f}, {node.k2_frac:.4f}, {node.k3_frac:.4f}), "
            f"E-EF={node.energy_rel_ef_eV:+.4f} eV"
        )
        ax.view_init(elev=CONE_VIEW_ELEV, azim=CONE_VIEW_AZIM)

        out2 = out_png.replace(".png", f"_plane_{ia+1}{ib+1}.png")
        ensure_dir(os.path.dirname(out2))
        fig.tight_layout()
        fig.savefig(out2, dpi=DPI)
        plt.close(fig)


def plot_single_cone_cuts(out_prefix: str, node: TrackRow):
    pack = get_case_hr_pack(node.case_label)
    if pack is None:
        return
    Rvecs, HR_div, recip_lat = pack

    k0 = np.array([node.k1_frac, node.k2_frac, node.k3_frac], dtype=float)
    pair_n = node.pair_n_1based - 1
    pair_m = node.pair_m_1based - 1

    evals0, evecs, _, dH = eigensystem_at_k(k0, Rvecs, HR_div)
    ana = analyze_two_band_linearization(pair_n, pair_m, evecs, dH)
    Vdirs = ana["Vdirs"]
    e_node = 0.5 * (evals0[pair_n] + evals0[pair_m])

    tlist = np.linspace(-CUT_LINE_QMAX_FRAC, CUT_LINE_QMAX_FRAC, CUT_LINE_NPTS)

    for iax in range(3):
        v = Vdirs[:, iax].astype(float)
        v = v / np.linalg.norm(v)

        xplot = []
        y1 = []
        y2 = []

        for t in tlist:
            kk = np.mod(k0 + t * v, 1.0)
            evals, _, _, _ = eigensystem_at_k(kk, Rvecs, HR_div)

            if CONE_SHIFT_TO_NODE_ENERGY:
                e1 = evals[pair_n] - e_node
                e2 = evals[pair_m] - e_node
            else:
                e1 = evals[pair_n]
                e2 = evals[pair_m]

            if recip_lat is not None:
                dq_cart = frac_k_to_cart((t * v).reshape(1, 3), recip_lat).reshape(3)
                x = np.sign(t) * np.linalg.norm(dq_cart)
            else:
                x = t

            xplot.append(x)
            y1.append(e1)
            y2.append(e2)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(xplot, y1, "-", linewidth=1.6, label=f"band {pair_n+1}")
        ax.plot(xplot, y2, "-", linewidth=1.6, label=f"band {pair_m+1}")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.grid(True)
        ax.set_xlabel(r"$\Delta k$ (1/$\AA$)" if recip_lat is not None else r"$\Delta k$ (frac)")
        ax.set_ylabel(r"$E-E_{\mathrm{node}}$ (eV)" if CONE_SHIFT_TO_NODE_ENERGY else r"$E$ (eV)")
        ax.set_title(
            f"{node.case_label} | path {node.path} {node.family_id}\n"
            f"Local band cut along principal dir {iax+1}"
        )
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_dir{iax+1}.png", dpi=DPI)
        plt.close(fig)



def compute_pair_kline_data(node_p: NodeRow, node_n: NodeRow) -> Tuple[List[dict], List[dict]]:
    """
    计算一对 Weyl 点连线方向上的两条能带数据。

    Origin 输出只使用 fractional Δk，不再输出 1/Å/perA 列。
    这样不依赖晶格读取是否成功，只要 HR 文件可读即可生成 k-line 表。

    返回：
      kline_rows      : Origin 画 band 56 / band 57 用的长表数据；
      node_marker_rows: Origin 画 Weyl 点竖线或散点标记用的数据。
    """
    pack = get_case_hr_pack(node_p.case_label)
    if pack is None:
        print(f"[WARN] 无法生成 k-line：未读到 HR | {node_p.case_label}")
        return [], []

    Rvecs, HR_div, _recip_lat = pack

    pair_n = node_p.pair_n_1based - 1
    pair_m = node_p.pair_m_1based - 1

    k_plus = np.array([node_p.k1_frac, node_p.k2_frac, node_p.k3_frac], dtype=float)
    k_minus_raw = np.array([node_n.k1_frac, node_n.k2_frac, node_n.k3_frac], dtype=float)
    dk = minimal_image_delta(k_minus_raw, k_plus)
    dist_frac = float(np.linalg.norm(dk))

    if dist_frac < 1e-12:
        print(f"[WARN] 无法生成 k-line：pair 距离过小 | {node_p.case_label}")
        return [], []

    direction = dk / dist_frac
    center = k_plus + 0.5 * dk

    tmin = -0.5 * dist_frac - PAIR_KLINE_MARGIN_FRAC
    tmax = +0.5 * dist_frac + PAIR_KLINE_MARGIN_FRAC
    tlist = np.linspace(tmin, tmax, PAIR_KLINE_NPTS)

    mode = ORIGIN_KLINE_ENERGY_REFERENCE.lower().strip()
    if mode == "node_mean":
        node_mean_rel = 0.5 * (node_p.energy_rel_ef_eV + node_n.energy_rel_ef_eV)
        if np.isfinite(node_p.fermi_energy_eV):
            e_ref = node_p.fermi_energy_eV + node_mean_rel
        else:
            e_ref = node_mean_rel
        energy_reference_label = "E_minus_node_mean_eV"
    elif mode == "raw":
        e_ref = 0.0
        energy_reference_label = "E_raw_eV"
    else:
        if np.isfinite(node_p.fermi_energy_eV):
            e_ref = node_p.fermi_energy_eV
        else:
            e_ref = 0.0
        energy_reference_label = "E_minus_EF_eV"

    x_plus_frac = -0.5 * dist_frac
    x_minus_frac = +0.5 * dist_frac

    kline_rows = []
    for idx, t in enumerate(tlist):
        kk = np.mod(center + t * direction, 1.0)
        ev, _, _, _ = eigensystem_at_k(kk, Rvecs, HR_div)

        band_n_energy = ev[pair_n] - e_ref
        band_m_energy = ev[pair_m] - e_ref

        kline_rows.append({
            "case_label": node_p.case_label,
            "eps_percent": node_p.eps_percent,
            "pair_n_1based": node_p.pair_n_1based,
            "pair_m_1based": node_p.pair_m_1based,
            "node1_chirality": node_p.chirality,
            "node2_chirality": node_n.chirality,
            "point_index": idx,
            "delta_k_frac": float(t),
            "band_n_index": node_p.pair_n_1based,
            "band_m_index": node_p.pair_m_1based,
            f"band_{node_p.pair_n_1based}_{energy_reference_label}": float(band_n_energy),
            f"band_{node_p.pair_m_1based}_{energy_reference_label}": float(band_m_energy),
            "band_n_energy_eV": float(band_n_energy),
            "band_m_energy_eV": float(band_m_energy),
            "energy_reference": mode,
            "pair_distance_frac": dist_frac,
            "node1_delta_k_frac": x_plus_frac,
            "node2_delta_k_frac": x_minus_frac,
            "node1_E_minus_EF_eV": node_p.energy_rel_ef_eV,
            "node2_E_minus_EF_eV": node_n.energy_rel_ef_eV,
            "fermi_energy_eV": node_p.fermi_energy_eV,
        })

    node_marker_rows = [
        {
            "case_label": node_p.case_label,
            "eps_percent": node_p.eps_percent,
            "node_label": "node_1",
            "chirality": node_p.chirality,
            "delta_k_frac": x_plus_frac,
            "E_minus_EF_eV": node_p.energy_rel_ef_eV,
            "k1_frac": node_p.k1_frac,
            "k2_frac": node_p.k2_frac,
            "k3_frac": node_p.k3_frac,
            "k1_centered": center_frac_value(node_p.k1_frac),
            "k2_centered": center_frac_value(node_p.k2_frac),
            "k3_centered": center_frac_value(node_p.k3_frac),
        },
        {
            "case_label": node_n.case_label,
            "eps_percent": node_n.eps_percent,
            "node_label": "node_2",
            "chirality": node_n.chirality,
            "delta_k_frac": x_minus_frac,
            "E_minus_EF_eV": node_n.energy_rel_ef_eV,
            "k1_frac": node_n.k1_frac,
            "k2_frac": node_n.k2_frac,
            "k3_frac": node_n.k3_frac,
            "k1_centered": center_frac_value(node_n.k1_frac),
            "k2_centered": center_frac_value(node_n.k2_frac),
            "k3_centered": center_frac_value(node_n.k3_frac),
        },
    ]

    return kline_rows, node_marker_rows

def export_origin_kline_tables(nodes: List[NodeRow]):
    """
    输出 Origin 重绘局部 k-line band 图所需表格。

    输出位置：
      <OUT_ROOT>/origin_data/kline/

    输出内容：
      all_kline_data_long.csv / .dat
      all_kline_node_markers.csv / .dat
      每个 case-pair 单独的 kline 表和 node_marker 表。
    """
    if not EXPORT_ORIGIN_TABLES or not EXPORT_ORIGIN_KLINE_TABLES:
        return

    out_dir = os.path.join(ORIGIN_OUT_DIR, "kline")
    ensure_dir(out_dir)

    all_kline_rows = []
    all_marker_rows = []

    if ORIGIN_OUTPUT_SYNC_WITH_PLOTS:
        selected_cases = list(PAIR_PLOT_SELECTED_CASES)
        max_pairs_per_case = PAIR_MAX_PAIRS_PER_CASE
    else:
        selected_cases = list(ORIGIN_KLINE_SELECTED_CASES)
        max_pairs_per_case = ORIGIN_KLINE_MAX_PAIRS_PER_CASE

    for case in selected_cases:
        pairs = pair_nodes_within_case(nodes, case, max_pairs_per_case)
        print(f"[ORIGIN KLINE] {case}: pair 数 = {len(pairs)}")

        for ip, (node_p, node_n, dfrac) in enumerate(pairs, 1):
            kline_rows, marker_rows = compute_pair_kline_data(node_p, node_n)

            if not kline_rows:
                print(f"[WARN] Origin k-line 数据为空: {case} pair {ip:02d}")
                continue

            pair_label = f"{sanitize_name(case)}_pair_{ip:02d}"

            for r in kline_rows:
                r["pair_index"] = ip
                r["pair_label"] = pair_label

            for r in marker_rows:
                r["pair_index"] = ip
                r["pair_label"] = pair_label

            all_kline_rows.extend(kline_rows)
            all_marker_rows.extend(marker_rows)

            if EXPORT_ORIGIN_INDIVIDUAL_KLINE_TABLES:
                base = os.path.join(out_dir, f"{pair_label}_kline")
                write_origin_table(base, kline_rows)

                marker_base = os.path.join(out_dir, f"{pair_label}_node_markers")
                write_origin_table(marker_base, marker_rows)

    all_kline_rows.sort(key=lambda r: (r["eps_percent"], r["case_label"], r["pair_index"], r["point_index"]))
    all_marker_rows.sort(key=lambda r: (r["eps_percent"], r["case_label"], r["pair_index"], r["node_label"]))

    write_origin_table(os.path.join(out_dir, "all_kline_data_long"), all_kline_rows)
    write_origin_table(os.path.join(out_dir, "all_kline_node_markers"), all_marker_rows)


def plot_pair_kline_band(out_png: str, node_p: NodeRow, node_n: NodeRow):
    kline_rows, marker_rows = compute_pair_kline_data(node_p, node_n)
    if not kline_rows:
        return

    pair_n = node_p.pair_n_1based
    pair_m = node_p.pair_m_1based
    dist = kline_rows[0]["pair_distance_frac"]

    xplot = [r["delta_k_frac"] for r in kline_rows]
    e1 = [r["band_n_energy_eV"] for r in kline_rows]
    e2 = [r["band_m_energy_eV"] for r in kline_rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xplot, e1, "-", linewidth=1.6, label=f"band {pair_n}")
    ax.plot(xplot, e2, "-", linewidth=1.6, label=f"band {pair_m}")
    ax.axhline(0.0, color="black", linewidth=0.8, label=r"$E_F$")

    if len(marker_rows) >= 2:
        node1 = marker_rows[0]
        node2 = marker_rows[1]

        color_p = node_chirality_color(node_p.chirality) if USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS else "tab:orange"
        color_n = node_chirality_color(node_n.chirality) if USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS else "tab:orange"

        ax.axvline(node1["delta_k_frac"], color=color_p, linestyle="--", linewidth=1.2,
                   label=fr"node 1, $\chi={node_p.chirality:+d}$")
        ax.axvline(node2["delta_k_frac"], color=color_n, linestyle="--", linewidth=1.2,
                   label=fr"node 2, $\chi={node_n.chirality:+d}$")

    ax.set_xlabel(r"$\Delta k$ along Weyl-pair line (centered fractional coordinate)")
    ax.set_ylabel(r"$E-E_F$ (eV)")
    ax.set_title(
        f"{node_p.case_label}: local k-line band between paired Weyl points\n"
        f"pair distance={dist:.4f} frac"
    )
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def plot_pair_energy_surface_3d(out_png: str, node_p: NodeRow, node_n: NodeRow):
    pack = get_case_hr_pack(node_p.case_label)
    if pack is None:
        return
    Rvecs, HR_div, recip_lat = pack

    pair_n = node_p.pair_n_1based - 1
    pair_m = node_p.pair_m_1based - 1

    k_plus = np.array([node_p.k1_frac, node_p.k2_frac, node_p.k3_frac], dtype=float)
    k_minus_raw = np.array([node_n.k1_frac, node_n.k2_frac, node_n.k3_frac], dtype=float)
    dk = minimal_image_delta(k_minus_raw, k_plus)
    dist = float(np.linalg.norm(dk))
    if dist < 1e-12:
        return

    efermi = node_p.fermi_energy_eV

    v1 = dk / dist
    v2 = np.array([-v1[1], v1[0], 0.0], dtype=float)
    if np.linalg.norm(v2) < 1e-12:
        v2 = np.array([0.0, -v1[2], v1[1]], dtype=float)
    v2 = v2 / np.linalg.norm(v2)

    center = k_plus + 0.5 * dk

    q1_min = -0.5 * dist - PAIR_SURFACE_EXTRA_FRAC
    q1_max = +0.5 * dist + PAIR_SURFACE_EXTRA_FRAC
    q2_min = -PAIR_SURFACE_PERP_HALF_FRAC
    q2_max = +PAIR_SURFACE_PERP_HALF_FRAC

    q1 = np.linspace(q1_min, q1_max, PAIR_SURFACE_GRID_N)
    q2 = np.linspace(q2_min, q2_max, PAIR_SURFACE_GRID_N)
    Q1, Q2 = np.meshgrid(q1, q2, indexing="xy")

    if recip_lat is not None:
        scale1 = np.linalg.norm(frac_k_to_cart(v1.reshape(1, 3), recip_lat).reshape(3))
        scale2 = np.linalg.norm(frac_k_to_cart(v2.reshape(1, 3), recip_lat).reshape(3))
    else:
        scale1 = 1.0
        scale2 = 1.0

    X = Q1 * scale1
    Y = Q2 * scale2
    E1 = np.zeros_like(Q1)
    E2 = np.zeros_like(Q1)

    if PAIR_SURFACE_SHIFT_TO_EF and np.isfinite(efermi):
        e_ref = efermi
        zlabel = r"$E-E_F$ (eV)"
    else:
        evp, _, _, _ = eigensystem_at_k(k_plus, Rvecs, HR_div)
        evn, _, _, _ = eigensystem_at_k(np.mod(k_plus + dk, 1.0), Rvecs, HR_div)
        e_ref = 0.5 * (0.5 * (evp[pair_n] + evp[pair_m]) + 0.5 * (evn[pair_n] + evn[pair_m]))
        zlabel = r"$E-E_{\mathrm{pair}}$ (eV)"

    for i in range(PAIR_SURFACE_GRID_N):
        for j in range(PAIR_SURFACE_GRID_N):
            kk = np.mod(center + Q1[i, j] * v1 + Q2[i, j] * v2, 1.0)
            ev, _, _, _ = eigensystem_at_k(kk, Rvecs, HR_div)
            E1[i, j] = ev[pair_n] - e_ref
            E2[i, j] = ev[pair_m] - e_ref

    fig = plt.figure(figsize=FIGSIZE_CONE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, E1, alpha=0.70, linewidth=0.10, edgecolor=(0, 0, 0, 0.16))
    ax.plot_surface(X, Y, E2, alpha=0.70, linewidth=0.10, edgecolor=(0, 0, 0, 0.16))

    x_plus = -0.5 * dist * scale1
    x_minus = +0.5 * dist * scale1
    e_plus = node_p.energy_rel_ef_eV if PAIR_SURFACE_SHIFT_TO_EF else 0.0
    e_minus = node_n.energy_rel_ef_eV if PAIR_SURFACE_SHIFT_TO_EF else 0.0

    color_p = node_chirality_color(node_p.chirality) if USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS else "tab:orange"
    color_n = node_chirality_color(node_n.chirality) if USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS else "tab:orange"
    marker_p = node_chirality_marker(node_p.chirality) if USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS else "o"
    marker_n = node_chirality_marker(node_n.chirality) if USE_CHIRALITY_COLOR_FOR_PAIR_MARKERS else "o"

    ax.scatter([x_plus], [0.0], [e_plus], color=color_p, marker=marker_p, s=70, edgecolors="k")
    ax.scatter([x_minus], [0.0], [e_minus], color=color_n, marker=marker_n, s=70, edgecolors="k")

    ax.set_xlabel(r"$q_{\parallel}$ (1/$\AA$)" if recip_lat is not None else r"$q_{\parallel}$ (frac)")
    ax.set_ylabel(r"$q_{\perp}$ (1/$\AA$)" if recip_lat is not None else r"$q_{\perp}$ (frac)")
    ax.set_zlabel(zlabel)
    ax.set_title(
        f"{node_p.case_label}: two-band surface across paired Weyl points\n"
        f"k-pair distance={dist:.4f} frac"
    )
    ax.view_init(elev=PAIR_SURFACE_VIEW_ELEV, azim=PAIR_SURFACE_VIEW_AZIM)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# ======================================================================
# 主程序
# ======================================================================

def main():
    ensure_dir(OUT_ROOT)

    with timed_section("find input WT flow root"):
        wt_root = find_latest_wt_flow_root()
        global ACTIVE_WT_ROOT
        ACTIVE_WT_ROOT = wt_root

    print("======================================================")
    print("[INPUT]")
    print(f"WT_FLOW_ROOT = {wt_root}")
    print(f"CASE_INPUT_ROOT = {CASE_INPUT_ROOT}")
    print(f"OUT_ROOT = {OUT_ROOT}")
    print("======================================================")

    with timed_section("load WannierTools nodes and chirality"):
        nodes = load_nodes_from_wt(wt_root)

    with timed_section("enrich nodes with HR tilt quantities"):
        nodes = enrich_nodes_with_hr_quantities(nodes)

    if not nodes:
        raise RuntimeError("未读取到任何满足条件的 Weyl 节点。请检查 WT_FLOW_ROOT、NUM_OCC_TO_USE 和 chirality 输出。")

    with timed_section("write wt_nodes_with_chirality.csv"):
        write_dataclass_csv(os.path.join(OUT_ROOT, "wt_nodes_with_chirality.csv"), nodes)

    print("\n================ 节点读取摘要 ================")
    print(f"读取节点总数: {len(nodes)}")
    for c, eps in case_order_pairs():
        sub = [nd for nd in nodes if nd.case_label == c]
        npos = sum(1 for x in sub if x.chirality > 0)
        nneg = sum(1 for x in sub if x.chirality < 0)
        nzero = sum(1 for x in sub if x.chirality == 0)
        print(f"{c:<30s} | eps={eps:+6.2f}% | N={len(sub):2d} | += {npos:d}, -= {nneg:d}, 0={nzero:d}")

    path_tracks: Dict[str, List[TrackRow]] = {}
    path_tracks_persistent: Dict[str, List[TrackRow]] = {}
    path_tracks_transient: Dict[str, List[TrackRow]] = {}
    persistent_family_ids_by_path: Dict[str, Set[str]] = {}

    all_tracks: List[TrackRow] = []
    all_persistent_tracks: List[TrackRow] = []
    all_transient_tracks: List[TrackRow] = []

    with timed_section("family tracking and persistent/transient split"):
        for path_name, cases in PATH_CASES.items():
            rows = track_one_path(nodes, path_name, cases)
            persistent_rows, transient_rows, persistent_fids = split_persistent_and_transient(rows, PERSISTENT_CORE_CASES)

            path_tracks[path_name] = rows
            path_tracks_persistent[path_name] = persistent_rows
            path_tracks_transient[path_name] = transient_rows
            persistent_family_ids_by_path[path_name] = persistent_fids

            all_tracks.extend(rows)
            all_persistent_tracks.extend(persistent_rows)
            all_transient_tracks.extend(transient_rows)

            write_dataclass_csv(os.path.join(OUT_ROOT, f"family_tracks_{path_name}.csv"), rows)
            write_dataclass_csv(os.path.join(OUT_ROOT, f"family_tracks_{path_name}_persistent_core.csv"), persistent_rows)
            write_dataclass_csv(os.path.join(OUT_ROOT, f"family_tracks_{path_name}_transient.csv"), transient_rows)

            if PRINT_SUMMARY:
                fams = group_tracks_by_family(rows)
                lengths = [len(v) for v in fams.values()]
                print("\n======================================================")
                print(f"[PATH] {path_name}")
                print(f"family 数量: {len(fams)}")
                print(f"track 总点数: {len(rows)}")
                if lengths:
                    print(f"family 长度: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")
                print(f"persistent core family 数: {len(persistent_fids)}")
                print(f"persistent core case 范围: {PERSISTENT_CORE_CASES[0]} -> {PERSISTENT_CORE_CASES[-1]}")
                print(f"transient track 点数: {len(transient_rows)}")
                print("======================================================")

    with timed_section("write family summary tables"):
        family_summary = summarize_families(all_tracks)
        persistent_summary = summarize_families(all_persistent_tracks)
        transient_summary = summarize_families(all_transient_tracks)

        write_dict_csv(os.path.join(OUT_ROOT, "family_summary.csv"), family_summary)
        write_dict_csv(os.path.join(OUT_ROOT, "persistent_family_summary.csv"), persistent_summary)
        write_dict_csv(os.path.join(OUT_ROOT, "transient_family_summary.csv"), transient_summary)

    if PLOT_COUNT_VS_STRAIN:
        with timed_section("plot count_vs_strain"):
            plot_count_vs_strain(nodes, os.path.join(OUT_ROOT, "count_vs_strain.png"))

    if PLOT_CHIRALITY_COUNT_VS_STRAIN:
        with timed_section("plot chirality_count_vs_strain"):
            plot_chirality_count_vs_strain(nodes, os.path.join(OUT_ROOT, "chirality_count_vs_strain.png"))

    if PLOT_ENERGY_SCATTER:
        with timed_section("plot energy_vs_strain_scatter"):
            plot_energy_scatter(nodes, os.path.join(OUT_ROOT, "energy_vs_strain_scatter.png"))

    for path_name, rows in path_tracks_persistent.items():
        if not rows:
            print(f"[WARN] path {path_name} 没有满足 PERSISTENT_CORE_CASES 的 persistent family，主演化图可能不会输出。")
            continue

        title_prefix = "persistent core"

        if PLOT_K_TRAJECTORY_3D_RAW:
            with timed_section(f"plot path_{path_name}_k_trajectory_3d_raw"):
                plot_3d_trajectory(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_k_trajectory_3d_raw.png"),
                    unwrapped=False,
                    title=f"{title_prefix} path {path_name}: centered fractional k-space trajectories",
                )

        if PLOT_K_TRAJECTORY_3D_UNWRAPPED:
            with timed_section(f"plot path_{path_name}_k_trajectory_3d_unwrapped"):
                plot_3d_trajectory(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_k_trajectory_3d_unwrapped.png"),
                    unwrapped=True,
                    title=f"{title_prefix} path {path_name}: centered/unwrapped k-space trajectories",
                )

        if PLOT_K_COMPONENTS_VS_STRAIN:
            with timed_section(f"plot path_{path_name}_k_components_vs_strain"):
                plot_k_components(rows, os.path.join(OUT_ROOT, f"path_{path_name}_k_components_vs_strain.png"), path_name, title_prefix)

        if PLOT_K_PROJECTIONS_2D:
            with timed_section(f"plot path_{path_name}_k_projection_2d"):
                plot_k_projections_2d(rows, os.path.join(OUT_ROOT, f"path_{path_name}_k_projection_2d.png"), path_name, title_prefix)

        if PLOT_K1K2_TRAJECTORY:
            with timed_section(f"plot path_{path_name}_k1k2_trajectory"):
                plot_k1k2_trajectory(rows, os.path.join(OUT_ROOT, f"path_{path_name}_k1k2_trajectory.png"), path_name, title_prefix)

        if PLOT_ENERGY_VS_STRAIN:
            with timed_section(f"plot path_{path_name}_energy_vs_strain"):
                plot_scalar_vs_strain(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_energy_vs_strain.png"),
                    path_name,
                    attr="energy_rel_ef_eV",
                    ylabel=r"$E_W-E_F$ (eV)",
                    title_tail="node energy vs strain",
                    semilogy=False,
                    title_prefix=title_prefix,
                )

        if PLOT_GAP_VS_STRAIN:
            with timed_section(f"plot path_{path_name}_gap_vs_strain"):
                plot_scalar_vs_strain(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_gap_vs_strain.png"),
                    path_name,
                    attr="gap_eV",
                    ylabel="gap (eV)",
                    title_tail="local gap vs strain",
                    semilogy=True,
                    title_prefix=title_prefix,
                )

        if PLOT_TILT_RATIO_VS_STRAIN:
            with timed_section(f"plot path_{path_name}_tilt_ratio_vs_strain"):
                plot_scalar_vs_strain(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_tilt_ratio_vs_strain.png"),
                    path_name,
                    attr="tilt_ratio",
                    ylabel="tilt_ratio",
                    title_tail="tilt ratio vs strain",
                    semilogy=False,
                    title_prefix=title_prefix,
                )

        if PLOT_CHIRALITY_PAIR_COMPARISON:
            with timed_section(f"plot path_{path_name}_chirality_pair_comparison"):
                plot_chirality_pair_comparison(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_chirality_pair_comparison.png"),
                    path_name,
                )

    if PLOT_TRANSIENT_FAMILIES_SEPARATELY:
        transient_dir = os.path.join(OUT_ROOT, "transient_family_plots")
        ensure_dir(transient_dir)

        for path_name, rows in path_tracks_transient.items():
            if not rows:
                continue

            title_prefix = "transient"

            if PLOT_K1K2_TRAJECTORY:
                with timed_section(f"plot path_{path_name}_transient_k1k2_trajectory"):
                    plot_k1k2_trajectory(
                        rows,
                        os.path.join(transient_dir, f"path_{path_name}_transient_k1k2_trajectory.png"),
                        path_name,
                        title_prefix,
                    )

            if PLOT_K_PROJECTIONS_2D:
                with timed_section(f"plot path_{path_name}_transient_k_projection_2d"):
                    plot_k_projections_2d(
                        rows,
                        os.path.join(transient_dir, f"path_{path_name}_transient_k_projection_2d.png"),
                        path_name,
                        title_prefix,
                    )

            if PLOT_ENERGY_VS_STRAIN:
                with timed_section(f"plot path_{path_name}_transient_energy_vs_strain"):
                    plot_scalar_vs_strain(
                        rows,
                        os.path.join(transient_dir, f"path_{path_name}_transient_energy_vs_strain.png"),
                        path_name,
                        attr="energy_rel_ef_eV",
                        ylabel=r"$E_W-E_F$ (eV)",
                        title_tail="node energy vs strain",
                        semilogy=False,
                        title_prefix=title_prefix,
                    )

            if PLOT_TILT_RATIO_VS_STRAIN:
                with timed_section(f"plot path_{path_name}_transient_tilt_ratio_vs_strain"):
                    plot_scalar_vs_strain(
                        rows,
                        os.path.join(transient_dir, f"path_{path_name}_transient_tilt_ratio_vs_strain.png"),
                        path_name,
                        attr="tilt_ratio",
                        ylabel="tilt_ratio",
                        title_tail="tilt ratio vs strain",
                        semilogy=False,
                        title_prefix=title_prefix,
                    )

    if PLOT_CASE_NODE_MAPS:
        with timed_section("plot case_node_maps"):
            plot_case_node_maps(nodes, os.path.join(OUT_ROOT, "case_node_maps"))

    if PLOT_SELECTED_CASE_K1K2_MAPS:
        with timed_section("plot selected_case_k1k2_maps"):
            plot_selected_case_k1k2_maps(nodes, os.path.join(OUT_ROOT, "selected_case_k1k2_maps"), REPRESENTATIVE_CASES)

    if PLOT_SELECTED_CORE_4PAIR_K1K2_MAPS:
        with timed_section("plot selected core 4-pair k1k2 maps"):
            plot_selected_core_4pair_k1k2_maps(
                path_tracks.get("c", []),
                os.path.join(OUT_ROOT, "selected_case_k1k2_maps"),
                path_name="c",
            )

    if PLOT_SELECTED_CASE_3D_MAPS:
        with timed_section("plot selected_case_3d_maps"):
            plot_selected_case_3d_maps(nodes, os.path.join(OUT_ROOT, "selected_case_3d_maps"), REPRESENTATIVE_CASES)

    if PLOT_OVERLAY_NODE_MAPS:
        for path_name, rows in path_tracks_persistent.items():
            if not rows:
                continue
            with timed_section(f"plot path_{path_name}_overlay_node_maps"):
                plot_overlay_node_maps(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_overlay"),
                    path_name,
                    "persistent core",
                )

    if PLOT_PAIR_DISTANCE_VS_STRAIN:
        for path_name, rows in path_tracks_persistent.items():
            if not rows:
                continue
            with timed_section(f"plot path_{path_name}_pair_distance_vs_strain"):
                plot_pair_distance_vs_strain(
                    rows,
                    os.path.join(OUT_ROOT, f"path_{path_name}_pair_distance_frac.png"),
                    os.path.join(OUT_ROOT, f"path_{path_name}_pair_distance_cart.png"),
                    path_name,
                    "persistent core",
                )

    if EXPORT_ORIGIN_KLINE_TABLES and ((not ORIGIN_OUTPUT_SYNC_WITH_PLOTS) or PLOT_PAIR_KLINE_BANDS):
        with timed_section("export Origin k-line tables"):
            export_origin_kline_tables(nodes)

    if PLOT_SCHEMATIC_LIKE:
        if path_tracks_persistent.get("c"):
            with timed_section("plot path_c_schematic_like"):
                plot_schematic_like(
                    path_tracks_persistent["c"],
                    os.path.join(OUT_ROOT, "path_c_schematic_like.png"),
                    "c",
                    SCHEMATIC_CASES_C,
                )

    if PLOT_WEYL_CONE_3D or PLOT_WEYL_CONE_CUTS:
        with timed_section("select nodes for local Weyl-cone plots"):
            selected = select_nodes_for_cone(path_tracks)

        print("\n================ Weyl 局部图节点选择 ================")
        print(f"选中节点数: {len(selected)}")

        for r in selected:
            print(
                f"  - {r.case_label} | path={r.path} | {r.family_id} | "
                f"chi={r.chirality:+d} | k=({r.k1_frac:.4f},{r.k2_frac:.4f},{r.k3_frac:.4f}) | "
                f"E-EF={r.energy_rel_ef_eV:+.4f} eV"
            )

            if PLOT_WEYL_CONE_3D:
                with timed_section(f"plot 3D Weyl cone | {r.case_label} | {r.family_id}"):
                    case_out = os.path.join(OUT_ROOT, "weyl_cones_3d", r.case_label)
                    fname = f"cone_path_{r.path}_{r.family_id}.png"
                    plot_single_cone_3d(os.path.join(case_out, fname), r)

            if PLOT_WEYL_CONE_CUTS:
                with timed_section(f"plot Weyl cone cuts | {r.case_label} | {r.family_id}"):
                    case_out = os.path.join(OUT_ROOT, "weyl_cone_cuts", r.case_label)
                    ensure_dir(case_out)
                    prefix = os.path.join(case_out, f"cut_path_{r.path}_{r.family_id}")
                    plot_single_cone_cuts(prefix, r)

    if PLOT_PAIR_KLINE_BANDS or PLOT_PAIR_ENERGY_SURFACE_3D:
        print("\n================ 成对 Weyl 点局部图 ================")
        for case in PAIR_PLOT_SELECTED_CASES:
            with timed_section(f"select paired nodes | {case}"):
                pairs = pair_nodes_within_case(nodes, case, PAIR_MAX_PAIRS_PER_CASE)

            print(f"{case}: 选中 pair 数 = {len(pairs)}")

            for ip, (node_p, node_n, dfrac) in enumerate(pairs, 1):
                if PLOT_PAIR_KLINE_BANDS:
                    with timed_section(f"plot pair k-line band | {case} | pair {ip:02d}"):
                        out_dir = os.path.join(OUT_ROOT, "pair_kline_bands", case)
                        ensure_dir(out_dir)
                        out_png = os.path.join(out_dir, f"pair_{ip:02d}_kline.png")
                        plot_pair_kline_band(out_png, node_p, node_n)

                if PLOT_PAIR_ENERGY_SURFACE_3D:
                    with timed_section(f"plot pair energy surface 3D | {case} | pair {ip:02d}"):
                        out_dir = os.path.join(OUT_ROOT, "pair_energy_surfaces_3d", case)
                        ensure_dir(out_dir)
                        out_png = os.path.join(out_dir, f"pair_{ip:02d}_surface3d.png")
                        plot_pair_energy_surface_3d(out_png, node_p, node_n)

    with timed_section("write timing summary"):
        write_timing_summary(OUT_ROOT)

    print("\n================ 输出完成 ================")
    print(f"输出目录: {OUT_ROOT}")
    print("核心文件:")
    print("  - wt_nodes_with_chirality.csv")
    print("  - family_tracks_c.csv")
    print("  - family_tracks_c_persistent_core.csv")
    print("  - family_tracks_c_transient.csv")
    print("  - family_summary.csv")
    print("  - persistent_family_summary.csv")
    print("  - transient_family_summary.csv")
    print("  - count_vs_strain.png")
    if PLOT_CHIRALITY_COUNT_VS_STRAIN:
        print("  - chirality_count_vs_strain.png")
    print("  - energy_vs_strain_scatter.png")
    if PLOT_SELECTED_CASE_K1K2_MAPS:
        print("  - selected_case_k1k2_maps/representative_k1k2_maps.png")
    if PLOT_SELECTED_CORE_4PAIR_K1K2_MAPS:
        print("  - selected_case_k1k2_maps/core_4pair_minus2_to_plus2_k1k2_maps.png")
        print("  - selected_case_k1k2_maps/core_4pair_minus2_to_plus2_nodes.csv")
    if PLOT_PAIR_DISTANCE_VS_STRAIN:
        print("  - path_c_pair_distance_frac.png")
        if PLOT_PAIR_DISTANCE_CART_FIGURE:
            print("  - path_c_pair_distance_cart.png")
        print("  - path_c_pair_distance_table.csv")
    if EXPORT_ORIGIN_TABLES and EXPORT_ORIGIN_PAIR_DISTANCE_TABLES and PLOT_PAIR_DISTANCE_VS_STRAIN:
        print("  - origin_data/pair_distance_vs_strain_long.csv")
        print("  - origin_data/pair_distance_vs_strain_wide.csv")
        print("  - origin_data/path_c_pair_distance_vs_strain_long.csv")
        print("  - origin_data/path_c_pair_distance_vs_strain_wide.csv")
    if EXPORT_ORIGIN_TABLES and EXPORT_ORIGIN_KLINE_TABLES and (PLOT_PAIR_KLINE_BANDS or not ORIGIN_OUTPUT_SYNC_WITH_PLOTS):
        print("  - origin_data/kline/all_kline_data_long.csv")
        print("  - origin_data/kline/all_kline_node_markers.csv")
        print("  - origin_data/kline/<case>_pair_XX_kline.csv")
        print("  - origin_data/kline/<case>_pair_XX_node_markers.csv")
    if PLOT_SELECTED_CASE_3D_MAPS:
        print("  - selected_case_3d_maps/*.png")
    if PLOT_TRANSIENT_FAMILIES_SEPARATELY:
        print("  - transient_family_plots/*.png")
    if PLOT_PAIR_KLINE_BANDS:
        print("  - pair_kline_bands/<case>/*.png")
    if PLOT_WEYL_CONE_CUTS:
        print("  - weyl_cone_cuts/<case>/*.png")
    if PLOT_WEYL_CONE_3D:
        print("  - weyl_cones_3d/<case>/*.png")
    if PLOT_PAIR_ENERGY_SURFACE_3D:
        print("  - pair_energy_surfaces_3d/<case>/*.png")
    if WRITE_TIMING_SUMMARY:
        print("  - timing_summary.csv")
        print("  - timing_summary.txt")
    print("==========================================")


if __name__ == "__main__":
    main()