# Public release QA

- QA date: 2026-09-20
- Final regular-file count: 269
- Total repository size: 7088477 bytes
- License files: LICENSE, LICENSE-CODE, and LICENSE-DATA are present and scoped to code versus research data/documentation.
- Prohibited calculation files: POTCAR = 0, WAVECAR = 0, CHGCAR = 0, wannier90_hr.dat = 0.
- Personal absolute paths: 0.
- Sensitive information: 0 private keys, passwords, tokens, API keys, private IP addresses, or non-default NTFS streams detected.
- Figure-centric content: 0 Figure/Supplementary-Figure directories and 0 final PNG/TIFF/JPG/PDF/SVG artwork files.
- Duplicate-data review: no redundant copy of a canonical processed scientific dataset is present. Identical state-local INCAR/KPOINTS inputs are intentionally retained in each calculation-state directory.
- Weyl consistency: common 0% reference confirmed; a +0.5% = -5.02884875 meV and b +0.5% = -2.23229625 meV. No superseded reference-state result remains in the public package.
- Berry consistency: canonical SciPy values confirmed; a +0.5% = 43.404756404476096% (43.4%) and b +0.5% = 36.530762700152565% (36.5%). No historical MATLAB value remains as a public canonical result.
- Optical chain: complete; 44 raw Kubo component files, xtract_optics_sigma.py, component CSV data, prepare_sigma_dataset.m, tensor data, and Fresnel/PSHE consumers are present.
- Final PSHE/LOOCV configuration: 700 nm, gamma2 = 82 deg, and angles = 70.3, 70.4, 70.5, 70.6, 70.7 deg. Final assessment is nested outer LOOCV with RMSE = 0.0069280712% and predictive R2 = 0.9983543484. Fixed-parameter five-angle LOOCV is labelled supporting/non-final; 632.8 nm branches are not described as final.
- Manifest consistency: 268 entries match every public regular file except documentation/final_repository_manifest.csv itself. The manifest intentionally excludes its own row because a stable cryptographic self-hash is impossible.
- Scientific provenance: SCIENTIFIC_PROVENANCE_COMPLETE.

**PUBLIC_RELEASE_READY**
