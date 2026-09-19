# Berry-curvature data

The producer evaluates the occupied-band Berry curvature from complete N56 Wannier Hamiltonians on the fixed k3=0 slice and writes three canonical 81×81 maps. The public statistics script selects `omega3_reduced` in the inclusive reduced-momentum region `[-0.035,0.035] × [-0.035,0.035]`, constructs a 241×241 target grid with `numpy.linspace` and `numpy.meshgrid(indexing="xy")`, and applies `scipy.interpolate.griddata(method="linear")`. Target points outside the linear interpolation hull are filled using `method="nearest"`. The reported quantity is the mean of `abs(omega3_reduced)` over all 58,081 target points and is normalized to the common 0% state.

The canonical SciPy results are stored in `data/berry_fixed_region_summary.csv`: +43.404756404476096% for a +0.5% and +36.530762700152565% for b +0.5%, displayed as +43.4% and +36.5%, respectively. Dense display-only maps and publication-layout code are excluded.
