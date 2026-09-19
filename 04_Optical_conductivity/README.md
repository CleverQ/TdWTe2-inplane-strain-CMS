# Optical conductivity

The public processing chain is:

`raw postw90 Kubo files → extract_optics_sigma.py → S_xx/S_xy/S_yy/A_xy component CSVs → prepare_sigma_dataset.m → full complex conductivity tensor and grids → Fresnel/PSHE`.

`raw_data/` contains four postw90 files for each of 11 states: `wannier90-kubo_S_xx.dat`, `wannier90-kubo_S_xy.dat`, `wannier90-kubo_S_yy.dat`, and `wannier90-kubo_A_xy.dat`. The extractor reads the three raw columns directly as `omega_eV`, `Re`, and `Im`; it applies no sign conversion, unit conversion, or interpolation and writes component values to eight decimal places.

The MATLAB preparation step uses

- `sigma_xx = S_xx`
- `sigma_yy = S_yy`
- `sigma_xy = S_xy + A_xy`
- `sigma_yx = S_xy - A_xy`

The bulk-conductivity unit is S/cm throughout extraction and tensor preparation. The 10 nm thickness is introduced only in the downstream Fresnel/PSHE model, after converting S/cm to S/m by multiplying by 100 and forming sheet conductivity as `sigma_s = d * sigma_bulk`.

Processed states are separated into `unstrained`, `a_0p1` through `a_0p5`, and `b_0p1` through `b_0p5`, preventing repeated source filenames from colliding. `data/processed/grids.mat` is the canonical wavelength-ready conductivity grid used downstream. No final artwork is included.
