# Strain estimation

The public operating configuration is 700 nm, gamma2 = 82 deg, with five incidence angles of 70.3, 70.4, 70.5, 70.6, and 70.7 deg.

The final assessment uses nested outer leave-one-state-out cross-validation. Parameter screening, feature selection, standardization, and ridge fitting are performed inside each outer training fold. The retained six held-out predictions give RMSE 0.0069280712% strain and predictive R2 0.9983543484.

A separate fixed-parameter five-angle LOOCV result is retained in `data/supporting_fixed_parameters/` only as supporting/non-final validation. Its smaller error must not be interpreted as the final nested generalization result. The 632.8 nm branches in the screening code are legacy diagnostic/experiment-compatible comparisons, not the final operating configuration.
