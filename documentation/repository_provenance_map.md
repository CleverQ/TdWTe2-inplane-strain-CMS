# Repository provenance map

1. State-resolved VASP inputs and relaxed structures → production electronic structures → state-resolved Wannier90 inputs.
2. Wannier90 inputs + complete N56 Hamiltonians (not publicly hosted) → WannierTools node search/chirality → tracked nodes → common-0% energy shifts, displacements, cone slopes, and tilt ratios.
3. Complete N56 Hamiltonians → occupied-band Berry-curvature producer → canonical 81×81 k3=0 maps → fixed-region 241×241 SciPy interpolation → mean `|omega3_reduced|` relative to 0%.
4. State-resolved postw90 Kubo files → `extract_optics_sigma.py` → `S_xx`, `S_xy`, `S_yy`, and `A_xy` → `prepare_sigma_dataset.m` → `sigma_xx`, `sigma_xy`, `sigma_yx`, and `sigma_yy` → canonical complex bulk-conductivity grid.
5. Complex bulk conductivity → S/cm-to-S/m conversion and 10 nm sheet-conductivity construction → full complex Fresnel matrix → amplitude/phase/component-substitution response → weak-measurement centroid response.
6. Weak-response scans → wavelength/gamma2/angle screening → five-angle features → fold-wise standardization and ridge fitting → nested outer LOOCV predictions and metrics.

All six scientific stages have a documented public data/code chain. Complete `wannier90_hr.dat` files remain non-public large intermediates available from the corresponding author upon reasonable request.

**SCIENTIFIC_PROVENANCE_COMPLETE**
