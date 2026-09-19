from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

FILES = [
    ("0pct", "berry_curvature_map_0pct.csv"),
    ("a_0p5pct", "berry_curvature_map_a_0p5pct.csv"),
    ("b_0p5pct", "berry_curvature_map_b_0p5pct.csv"),
]
REGION = (-0.035, 0.035)
GRID_SIZE = 241


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "maps_81x81",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "berry_fixed_region_summary.csv",
    )
    args = parser.parse_args()

    axis = np.linspace(REGION[0], REGION[1], GRID_SIZE)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    rows = []
    for state, name in FILES:
        frame = pd.read_csv(args.data_dir / name)
        local = frame[
            frame["k1_centered"].between(*REGION, inclusive="both")
            & frame["k2_centered"].between(*REGION, inclusive="both")
        ].copy()
        points = local[["k1_centered", "k2_centered"]].to_numpy(float)
        values = local["omega3_reduced"].to_numpy(float)
        zz = griddata(points, values, (xx, yy), method="linear")
        missing = ~np.isfinite(zz)
        if missing.any():
            zz[missing] = griddata(
                points, values, (xx[missing], yy[missing]), method="nearest"
            )
        rows.append(
            {
                "state": state,
                "source_points_in_region": len(local),
                "mean_abs_omega3": float(np.mean(np.abs(zz))),
            }
        )

    reference = rows[0]["mean_abs_omega3"]
    for row in rows:
        row.update(
            {
                "relative_change_percent":
                    (row["mean_abs_omega3"] / reference - 1.0) * 100.0,
                "reference_state": "0pct",
                "value_column": "omega3_reduced",
                "region_k1": "[-0.035,0.035]",
                "region_k2": "[-0.035,0.035]",
                "target_grid": "241x241",
                "interpolation_method": "scipy.griddata(linear)",
                "nan_fill": "nearest_outside_linear_hull",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
