# Repository consistency check

- Scientific provenance status: `SCIENTIFIC_PROVENANCE_COMPLETE`.
- DFT: 11 states, each with POSCAR, CONTCAR, INCAR, and KPOINTS.
- Wannier90: 11 production `wannier90.win` files.
- WannierTools: 11 states with text inputs plus compact node-search and chirality outputs.
- Weyl energy reference: common 0%; a +0.5% = -5.02884875 meV, b +0.5% = -2.23229625 meV.
- Berry: canonical SciPy workflow gives +43.404756404476096% and +36.530762700152565%, displayed as +43.4% and +36.5%. Historical MATLAB values are not public canonical results.
- Optical raw data: 44 Kubo files = 11 states × 4 components. The extractor, component CSV files, tensor preparation, and downstream Fresnel/PSHE chain are present.
- Optical tensor convention: `sigma_xy = S_xy + A_xy` and `sigma_yx = S_xy - A_xy`.
- Bulk optical conductivity remains in S/cm through preparation. The 10 nm thickness enters only downstream in the Fresnel/PSHE stage.
- Canonical operating configuration: 700 nm, gamma2 = 82 deg, angles = 70.3, 70.4, 70.5, 70.6, and 70.7 deg.
- Final model assessment: nested outer LOOCV, RMSE = 0.0069280712% strain, predictive R2 = 0.9983543484.
- Fixed-parameter five-angle LOOCV is stored separately and labelled supporting/non-final. Legacy 632.8 nm screening branches are not the final operating configuration.
- Author-created code is MIT licensed; research data, calculation inputs, and documentation are CC BY 4.0 licensed.
- No manuscript artwork, Figure-centric directories, internal audit reports, prohibited large calculation files, credentials, personal absolute paths, or release-junk files remain.
- `documentation/final_repository_manifest.csv` lists every public file except itself, because a cryptographic self-hash cannot be stable.

**PUBLIC_RELEASE_READY**
