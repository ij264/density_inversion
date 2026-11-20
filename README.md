# density-inversion

Mantle density from topography and the geoid. This repository holds the forward problem, built step by
step as pygeoinf operators, each step with its own tests and figures.

    h = F_Te q[rho_m; eta]  +  F_Te L[rho_L, dr_i]  +  h_s        (topography)
    N = sum of the potentials of every mass                        (geoid)

| Step | Module | Status |
|---|---|---|
| 1. Mantle density -> load on the base of the plate | `forward/mantle.py` | done |
| 2. Elastic plate: load -> deflection, water at sea and air on land | `forward/plate.py` | done |
| 3. Lithospheric loads: placeholder layers (3a, 3b); WINTERC-G reference lithosphere and its load (3c, 3d) | `forward/lithosphere.py` | done |
| 4. Surface field h_s (additive, prior open) | `forward/surface.py` | done |
| 5. Geoid: mantle part, lithospheric sheets, the deflected plate, the reference lithosphere | `forward/geoid.py` | done |
| 6. Assembled operator: (mantle, lithosphere, h_s) -> (topography, full geoid) | `forward/model.py` | done |

## Layout

    src/density_inversion/
        operators.py            density spaces (layered Sobolev ball) and kernel operators
        forward/mantle.py       step 1: MantleFlow, the kernels from flowkernels and the operator drho_m -> q
        forward/plate.py        step 2: ElasticPlate (water everywhere), OceanLoadedPlate (land and sea)
        forward/lithosphere.py  step 3: layer densities (nodal in depth, GLL) and interfaces -> load
        forward/surface.py      step 4: SurfaceField (h_s), high_pass prior shape
        forward/model.py        step 6: ForwardModel, the block operator with reference offsets
        forward/geoid.py        step 5: Geoid (mantle, lithosphere, deflection and reference parts)
        forward/examples.py     example density fields (a dense blob, a 1/k random field); read no files
        data/reference_model.csv  radial reference model (PREM densities, 257 nodes)
        data/ocean_function.csv   present-day ocean function, 1 degree
        data/reference_lithosphere.npz  WINTERC-G crust, Moho, mantle densities to 110 km, LAB and its jump, 1 degree
    scripts/make_reference_lithosphere.py  builds the .npz from the WINTERC-G v5.4 distribution (run once)
    tests/                      pytest
    figures/                    one script per step; output in figures/output/ (not in git)

## Install

flowkernels is private, so install it first from its own clone, then this package:

    pip install -e ../flowkernels
    pip install -e ".[figures,test]"

## Run

    pytest -q                                  # adjoint, sign, quadrature, examples
    python figures/01_mantle_flow.py           # kernels, two examples (and the fem_1d check, if set)
    python figures/02_plate.py                 # flexural filter, ocean function, flexed and land/sea examples
    python figures/03_lithosphere.py           # layers and nodes, a crustal root, a craton, the WINTERC-G reference
    python figures/04_surface_field.py         # residual topography by degree, candidate h_s priors
    python figures/05_geoid.py                 # geoid kernels, the blob, the reference lithosphere, compensation, spectra
    python figures/06_forward_model.py         # operator blocks; topography and geoid of a synthetic Earth
    python figures/schematic.py                # the slide schematic of the Earth model

Kernels are cached in `~/.cache/density_inversion` (set `DENSITY_INVERSION_CACHE` to change it). The cache
key covers the viscosity, LAB depth, lmax and flowkernels version, so stale kernels are never reused.

## Optional inputs

Nothing outside this repository and flowkernels is needed. Two environment variables switch on extras:

| Variable | Points to | Switches on |
|---|---|---|
| `DENSITY_INVERSION_FEM_KERNELS` | `kernels_jump30.npz`, from fem_1d via `ch2_density/kernels/earth_kernels.py` in the thesis repo | the cross-check against a second solver (test and figure) |
| `CARTOPY_DATA_DIR` | a folder of Natural Earth shapefiles | coastlines offline (otherwise cartopy downloads them) |

## Ocean function

`data/ocean_function.csv` is the fraction of each 1 degree cell below sea level, from `ETOPO2_km_depth_Ice.xyz`
in the WINTERC-G v5.4 distribution (Fullea et al. 2021): surface elevation including grounded ice, so ice sheets
count as land. Area-weighted ocean fraction 0.708.

## Reference lithosphere

`data/reference_lithosphere.npz` is WINTERC-G v5.4 (Fullea et al. 2021) on a 1 degree grid: bedrock elevation,
ice thickness, Moho depth, average crustal density, mantle densities at the Moho and 20-110 km, the LAB and the
density jump across it. Built by `scripts/make_reference_lithosphere.py`. WINTERC-G was itself fitted to
elevation, so its isostatic topography matching observed topography (98 % of the variance) is a consistency
check, not an independent result.

## Reference model

`data/reference_model.csv` holds the radii and densities of Sia's `inputs.sph` (lyness code), extracted on
17 Nov 2026. flowkernels' default constants are used with it.
