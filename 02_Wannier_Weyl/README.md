# Wannier and Weyl analysis

`wannier90/` contains the production `wannier90.win` input for each of the 11 states. `wanniertools/` contains state-resolved node-search and chirality `wt.in` files plus compact coordinate, energy, chirality, and validation outputs. Downstream canonical tables and scientific analysis code remain in `data/` and `scripts/`.

Energy shifts are eight-node signed means relative to the common 0% state. The full-scan and summary tables contain node-resolved and strain-averaged cone/tilt metrics.

The two-band linearization projects analytic Hamiltonian derivatives into the crossing-band subspace. If `v0` is the tilt vector and `G = B^T B` is formed from Pauli-vector slopes, the tilt ratio is `sqrt(v0^T G^-1 v0)`. The mean cone slope is the mean singular value of `B`, in eV per reduced momentum increment. A ratio greater than 1 is classified as type II.

Complete `wannier90_hr.dat` Hamiltonians are not hosted because of their size. They are available from the corresponding author upon reasonable request.
