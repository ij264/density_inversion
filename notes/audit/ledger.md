# Convention ledger (WP0) — as declared by the code; each row to be verified against source

Status column: "declared" = stated in a docstring or comment; "verified" = confirmed by reading the executing code or by a numerical test (cite it).

| # | Quantity | Convention declared | Where declared | Status |
|---|---|---|---|---|
| C1 | Stress | positive in tension; sigma_rr from flowkernels traces | mantle.py docstring | declared |
| C2 | Gravitational potential | physics sign, nabla^2 Phi = 4 pi G rho (so Phi < 0 near a mass) | geoid.py docstring | declared |
| C3 | Geoid | N = -Phi / g0 at r = R; a mass excess raises N | geoid.py docstring | declared |
| C4 | flowkernels potential trace phi | sign and non-dimensionalisation (PHI = k r phi, k = rho_ref / visc_ref) | flowkernels.py l.63 | declared; sign relative to C2 NOT yet checked |
| C5 | Load q | Pa, positive up; a mass excess gives q < 0 | mantle.py, lithosphere.py | declared |
| C6 | Deflection w, topography h | m, positive up; h = w (bedrock elevation, water-loaded at sea) | plate.py, model.py | declared |
| C7 | Interface perturbation dr_i | positive up; contrast Drho_i = rho_below - rho_above; relative to the plate | lithosphere.py | declared |
| C8 | Depths in reference data | positive down from sea level (bed_elevation positive up) | scripts/make_reference_lithosphere.py | declared; Moho reference (sea level vs surface) NOT yet checked against WINTERC-G README |
| C9 | Reference column | from the column's own surface to T = 105 km below it | lithosphere.py structure(), column_mass() | from code |
| C10 | Real spherical harmonics | orthonormal on the unit sphere (integral of Y^2 = 1) | operators.py | declared |
| C11 | pyshtools usage | norm=4 (orthonormal), csphase=1 (no Condon-Shortley) | examples.py | from code |
| C12 | Component order | degree by degree; orders 0, 1..l, -1..-l | operators.py | declared |
| C13 | Sphere spaces | unit radius; components = orthonormal coefficients; rigidity passed as D / R^4 | plate.py, mantle.py | declared |
| C14 | Degree 0 | geoid and reference load: set to zero | geoid.py, lithosphere.py | from code |
| C15 | Degree 1 | geoid zero (centre-of-mass frame); load and topography at l=1 NOT zeroed | mantle.py, geoid.py | from code; consistency open |
| C16 | RMS of a field | sqrt(sum c^2 / (4 pi)) over the chosen degrees | figures | from code |
| C17 | Gravity | g_lab (flowkernels gravity at r_LAB) for loads and plate; g0 = G M / R^2 for the geoid | mantle.py, plate.py, lithosphere.py, geoid.py | from code |
| C18 | Densities at the flow top | rho_lab = PREM at r_LAB (flow, plate restoring force); WINTERC-G density at 105 km at the base of the lid | mantle.py, geoid.py | from code; consistency open |
| C19 | Water | rho_w = 1030 kg/m^3 (flowkernels constant); ocean function C = 1 at sea, 0 on land | flowkernels, plate.py | from code |
| C20 | Viscosity | 1 above 670 km, 30 below (jump on node nearest 670 km, 666 km) | mantle.py viscosity_jump | from code |

Run environment (container): `cd <repo>; PYTHONPATH=/mnt/user-data/uploads/dev/flowkernels:src python3 -m pytest -q` (53 pass, 2 skip: fem_1d not available).
