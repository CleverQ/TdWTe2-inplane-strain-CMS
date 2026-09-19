# Anisotropic in-plane strain engineering of Td-WTe2

This repository contains canonical numerical data and scientific analysis code for the workflow

`DFT → Wannier/Weyl → Berry curvature → optical conductivity → Fresnel/PSHE → strain estimation`.

It is organized by scientific stage, not by manuscript artwork. Final PNG/TIFF/JPG/PDF/SVG files and layout-only scripts are intentionally excluded. Each scientific dataset has one canonical copy.

## Structure

- `01_DFT/`: lightweight VASP inputs and relaxed structures for 11 strain states.
- `02_Wannier_Weyl/`: Wannier90/WannierTools text inputs, compact node/chirality outputs, tracked nodes, common-0% energy shifts, cone slopes, tilt ratios, and two-band analysis code.
- `03_Berry_curvature/`: k3=0 canonical 81×81 maps, map producer, and fixed-region SciPy statistics.
- `04_Optical_conductivity/`: raw postw90 Kubo components, extraction/preparation code, 11 independently named processed states, and the assembled conductivity tensor.
- `05_Fresnel_PSHE/`: complex Fresnel and weak-measurement response data and code.
- `06_Strain_estimation/`: feature construction, screening, ridge regression, and nested outer LOOCV.
- `documentation/`: manifests, provenance, consistency checks, exclusions, numerical QA, and remaining release tasks.

## Reference conventions

- Strain is reported in percent.
- Weyl-node energy shifts use the common unstrained 0% state as the reference. At +0.5%, the eight-node means are -5.02884875 meV (a axis) and -2.23229625 meV (b axis).
- Berry statistics use `omega3_reduced` on the k3=0 slice. Canonical maps are 81×81. The fixed region is `[-0.035, 0.035] × [-0.035, 0.035]`; SciPy `griddata(method="linear")` interpolates it to 241×241, with nearest-neighbor fill only outside the linear interpolation hull. The canonical changes are +43.4% (a +0.5%) and +36.5% (b +0.5%) relative to 0%.
- The fixed operating configuration is 700 nm, gamma2 = 82 deg, and incidence angles 70.3, 70.4, 70.5, 70.6, and 70.7 deg.
- The final validation is nested outer LOOCV: RMSE = 0.0069280712% strain and predictive R2 = 0.9983543484. Fixed-parameter five-angle LOOCV is retained only as supporting validation.

## Running analyses

The analysis code mainly uses standard scientific Python packages, including NumPy, pandas, SciPy, Matplotlib, scikit-learn, PyYAML, and h5py. Scripts use repository-relative paths, with command-line or environment overrides where non-public Hamiltonian inputs are required. MATLAB is used for the conductivity-tensor preparation script.

Large intermediate calculation files generated during the first-principles and Wannier calculations, including the complete `wannier90_hr.dat` Hamiltonian files, are not hosted in this repository because of their size. These files are available from the corresponding author upon reasonable request.





## License

Code in this repository is licensed under the MIT License.
Research data and documentation are licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).
