# Berry numerical QA

## Canonical calculation definition

- Source: three canonical 81×81 maps in `03_Berry_curvature/data/maps_81x81/`.
- Value column: `omega3_reduced`.
- Source selection: inclusive `k1_centered` and `k2_centered` bounds from -0.035 to +0.035; 399 source points per state.
- Target grid: `numpy.linspace(-0.035, 0.035, 241)` on both axes and `numpy.meshgrid(..., indexing="xy")`.
- Interpolation: `scipy.interpolate.griddata(method="linear")`.
- Fill: `griddata(method="nearest")` only for target points outside the linear interpolation hull.
- Initial non-finite target count: 4306 for 0%, 4306 for a +0.5%, and 4306 for b +0.5%; zero after nearest fill.
- Statistic: mean of `abs(omega3_reduced)` over all 58,081 target points.
- Reference: common 0% state; `(strained_mean / reference_mean - 1) × 100`.

## Runtime

- Python: 3.13.14
- NumPy: 2.4.6
- SciPy: 1.17.1
- pandas: 3.0.3

## Exact canonical result

| State | mean_abs_omega3 | relative_change_percent | Display |
|---|---:|---:|---:|
| 0% | 0.1656236196139367 | 0 | 0.0% |
| a +0.5% | 0.2375121482556420 | 43.404756404476096 | 43.4% |
| b +0.5% | 0.22612719107050727 | 36.530762700152565 | 36.5% |

The SciPy result above is the sole canonical public result. Historical MATLAB interpolation diagnostics are not part of this repository.
