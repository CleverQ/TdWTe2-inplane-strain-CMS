# Fresnel and PSHE

The complex optical-conductivity tensor is converted to sheet response and then to the full complex Fresnel matrix (`rpp`, `rps`, `rsp`, `rss`). Stored outputs retain amplitude, continuously unwrapped phase, component-substitution increments, nonlinear residuals, and wavelength/incidence-angle dependence. The weak-measurement workflow uses these coefficients with beam and polarization parameters to compute spatial, angular, propagation, and centroid contributions.

The canonical operating configuration is 700 nm, gamma2 = 82 deg, and incidence angles 70.3, 70.4, 70.5, 70.6, and 70.7 deg. Any 632.8 nm branch retained in the scientific screening code is a legacy diagnostic/experiment-compatible comparison and is not the final operating configuration.
