# Audit findings

## Summary after round 2 (adversarial cross-check)
Every major round-1 finding was challenged and re-run by a fresh agent. Result: four root causes, all confirmed; one research item refuted.

| # | Root cause | Covers | Verdict | Size | Fix |
|---|---|---|---|---|---|
| 1 | Sea level / degree 0: zero-mean reference load with a land/sea restoring contrast | L1, P12, G8 | **FIXED 2026-11-18** (model.py: K0 fitted on columns holding the whole lithosphere, dK unknown; tests in test_model.py). Thin-lithosphere oceans now match the seafloor to 2 m; continents sit too high in proportion to LAB depth (+0.07 km for LAB < 100 km, +0.19 km for 100-150 km, +0.46 km for 150-200 km, +0.87 km beyond 200 km): the cold root below 105 km, missing from the reference, for the mantle unknown to explain | 705 m spurious land-ocean step; 316 m rms leaking into all degrees | one scalar nuisance unknown K (uniform pressure through P, into h and N); keep l = 0 datum |
| 2 | Ocean water follows the geoid: loaded in K_q but its mass never in the geoid; applied on land; observable is w - N | M1, G1, M2, P4, P5 | **FIXED 2026-11-18** (forward/ocean.py; K_q rock only; model.py: N = (I+T)^-1 r, w = P(q - rho_w g C N), h = w - N; Neumann contraction 0.0014 at L=50). Limit test reproduces flowkernels' K_N_fk and K_h0 to 1e-10 (old: 11 % low at l=2). Real ocean: water sheet 7 % of the mantle geoid at l=2, N/w 5 % at l=2 | mantle geoid 10.9 % low at l=2, 6 % at l=4 | first-order sea-level equation: drop rho_w U from K_q; load -rho_w g C N, sheet rho_w C N at R; solve; data operator h_obs = w - N everywhere |
| 3 | Geoid assembly geometry | E2, G3, L10, G5, G2, G4 | **FIXED 2026-11-18** (geoid.py: exact variable-radius sheets by a series in ln(r/R); sea-level frame for perturbations, column frame for the reference and its own ocean; deflection kept on the fine grid; lid integrated piece by piece between knots, also in column_mass; ice moved with the column). Brute-force Pacific case: -0.241 m vs -0.248 m (was -0.70). In-band mass balance of a compensated root 1e-9 (was 0.5-0.8 %). Net N_ref 7.2 m rms | compensated dMoho/dLAB 50-70 %; N_ref 7-16 %; mantle <= 2.6 % | rewrite: w on fine grid, truncate only observables; sheets at own radii in sea-level frame; one Moho-split quadrature; N_ref by finite-geometry integration |
| 4 | dLAB and fixed 105 km lid base | L2, research 8b | **FIXED 2026-11-18 (code) / OPEN (decision)**: dLAB off by default (lab_interface=False); reference rebuilt with WINTERC-G densities to 400 km so the lid base can move; sensitivity (L=30): continental excess over cratons (LAB > 200 km) +865 m at T=105, +381 at 150, +162 at 200, -9 at 260; rms misfit 364 / 268 / 259 / 290 m. Choice of T is a science decision | drho_lab ~8 kg/m^3, so small | drop dLAB; continuous sub-Moho density with LAB-correlated prior; test T = 105/150/200 km |
| D | Data: geoid must be non-hydrostatic | G6 | stands (data preparation). CORRECTED by referee 3: the hydrostatic part removed is ~10.8 km in the C20 component; -115 +- 1 m is the non-hydrostatic remainder; Nakiboglu's 15 m difference is an error, not an uncertainty | | subtract hydrostatic C20, C40 (Chambat et al. 2010, +-1 m, +-0.2 m); remove C21/S21 (rotational stability) |
| C | Data covariance double counts crustal correction | research 8c | stands | | gravity-model error + modelling-error floor (~10-15 % on compensated lithospheric terms, linearisation) |
| R | "Plate self-gravity neglected in the flow" (25-36 % at l=2) | research 8a | REFUTED as stated: already in the self-gravitating solve; residual <= 0.6 % at l=2, water part fixed by #2 | | document |

Must fix before any inversion, in order: 1, 2, D, 3, C, 4. Document only: research 8a, E3 (planar vs spherical balance, ~14 % floor on compensated geoids), G10.

Cleared (no change needed): flowkernels equations (sympy) and closed-form benchmark (4e-14); kernel algebra and continuation; degree-1 frame in the flow; WINTERC-G depth conventions (t_c = moho + bed correct); ETOPO sign; regridding; fixed-length-column isostasy; Airy with real structure; sheet formula vs Newton (1e-14); assembly linearity, offsets, parts; all Sobolev adjoints; CG solver; flexure operator scaling and manufactured solution.

Minor and wording items are listed per area below.


Severity: blocker / major / minor / wording / cleared. Evidence scripts under scratchpad/audit/.

## E2E — independent brute-force check (round 3)

Case: Moho raised by a 2 km Gaussian cap (sigma 400 km) under the central Pacific (0N, 200E), local isostasy, uniform column (water 4.96 km, crust 2938, lid from WINTERC-G knots, PREM rho_LAB 3370 below). Brute force: Newton's integral by quadrature with elliptic azimuthal integrals, no harmonics (audit/bruteforce/bruteforce_geoid.py; numerical error ~1e-8 m).

| Quantity at centre, l = 2..50 | Brute force | Package (Te = 0) | Spectral, true depths | Spectral, package depths |
|---|---|---|---|---|
| Moho sheet | +15.364 m | +16.64 m* | +15.364 | +15.058 |
| Rigid-lift sheets | -15.612 m | -17.34 m* | -15.621 | -15.722 |
| Total geoid | **-0.248 m** | **-0.70 m** | **-0.256** | **-0.664** |
| w (l = 1..50) | -350.9 m | -373.9 m* | | |

\* the package uses laterally varying WINTERC-G contrasts across the cap (up to 490 kg/m^3 against 414 at the centre), so its individual terms are larger; the last two columns isolate geometry (audit/me/sheet_depths.py).

- **E1 (cleared).** The sheet formula S_l(r) and its sign agree with direct Newton quadrature to 1e-9 m.
- **E2 (major).** The geoid of a compensated perturbation is a small residual of large terms (here -0.25 m from +15.4 and -15.6 m). The package places sheets at mean radii: the surface sheet at sea level instead of the seafloor (4.96 km deep here), and Moho sheets at the global mean Moho (22.9 km) instead of the local one (14.5 km). This alone moves the total from -0.256 to -0.664 m, an error of 160 % for an oceanic Moho perturbation. Code: geoid.py `deflection_sheets` (surface sheet at depth 0; Moho at mean depth), `lithosphere_depths` (mean depths). Fix: place each sheet at its local radius. Since S_l(r) depends on r, use a depth-expanded operator: S_l(r(x)) ≈ S_l(r0) + S_l'(r0)(r(x) - r0) as a pair of multiplication operators, or exact per-degree evaluation on the fine grid by splitting into a few depth bins. Test: this brute-force case to < 1 % of the total.
- **E3 (note).** First-order choices worth stating: the Moho sheet at the unperturbed Moho (moving it to the centroid changes N by ~15 % of the total here), and planar vs spherical mass balance (-0.254 vs -0.219 m). Both are second order in the perturbation, but the cancellation amplifies them.

## P — plate (round 1; scripts in audit/plate/)
| ID | Sev | Finding | Evidence | Proposed fix |
|---|---|---|---|---|
| P1 | cleared | D/R^4 scaling exact; manufactured solution recovered | 7e-16; 1.9e-11 | add manufactured test |
| P2 | minor | pygeoinf bending = flat plate written covariantly, not shell theory (degree-1 stiffness) | F_l diff <= 7.9e-5 at l<=50 | docstring wording |
| P3 | minor | membrane stresses neglected | F_l too large by 0.2-0.4 % | state in chapter; optional term E Te/R^2 (lam-2)/(lam-1+nu) |
| P4 | major (definition) | observable at sea is w - N_s (sea level follows geoid), code uses w | N/h ~1-5 % (5 % at l=2) | data operator [I, -I] on (h,N) at ocean data, or state reference |
| P5 | minor/major | rho_w PHI term in K_q = water-follows-geoid term; applied on land too, uses flowkernels' free-top geoid at r_LAB | 0.6 % rms of h; 5-6 % at l=20-50 for blobs | remove from K_q; solve (P^-1 + C rho_w g G_total) w = q_sigma - C rho_w g G_m x; or document |
| P6 | minor | water restoring term uses g_LAB not g0 | F at sea 0.41 % small | rho_LAB g_LAB - rho_w C g0 |
| P7 | cleared | rho_LAB = PREM consistent; 31.6 kg/m^3 jump to WINTERC-G at 105 km is real in the model | WINTERC-G at 105 km mean 3338.8 | state in chapter |
| P8 | minor | ocean function: grounded ice below sea level (1.2 % area) treated as air; Caspian as ocean; ice_thickness noise 0-1 m in 64 % of cells | area C 0.708 vs bed<0 0.721 | document; optional rho_ice infill; zero ice < 1 m |
| P9 | minor | quad_lmax convergence algebraic: 3L error ~1-1.5 % L2, ~3 % at coasts (docstring says 1 %) | 3L vs 8L table | taper C over ~100 km or exact cell integrals |
| P10 | cleared | CG tolerance/preconditioner fine | cond 1.44 | rtol 1e-6 enough |
| P11 | cleared | P symmetric pos. def. in components; not self-adjoint in Sobolev metric (adjoint correct) | 3.1 % | never use P for P* |
| P12 | major | no degree-0 (volume) constraint: land/sea couples loads into w_00 | mean h_ref -315 m; enforcing w_00=0 changes h_ref at l>=2 by 2 % (grid up to 382 m) | rank-one uniform-pressure correction |
| P13 | note | degree 1 is 71 % of |q| for the power-law test field; keep or drop l=1 must be decided with step 1 | | decide with M |
| P14 | minor | land/sea Galerkin test compares different discretisations; 2 % tolerance hides P9 | | replace with manufactured test |
| P15 | wording | callable T_e sets D=0 so filter() silently returns Te=0 | | raise |
| P16 | cleared | signs and units | | |

## L — lithosphere (round 1; scripts in audit/lithosphere/)
| ID | Sev | Finding | Evidence | Proposed fix |
|---|---|---|---|---|
| L1 | major | zero-mean q_ref sets the wrong sea level: with restoring contrast rho_LAB - rho_w C(x), the compensation constant K leaks into every degree (same root cause as P12) | no plate: mean h_ref -318 m vs bed -2441 m; spurious land-ocean step 705 m; Fullea 2021 calibrate a constant Pi0 | q_ref = -g(M - K); K fitted, or a scalar nuisance unknown |
| L2 | major | dLAB double counted where LAB > lid base (28 % of area): load applied isostatically and sheet at mean LAB 104.8 km although mass is in the flow domain; where LAB in lid, dLAB redundant with lid nodes | lab_frac.py | mask factor to LAB < T - bed, or drop dLAB |
| L3 | minor | 1 deg fields point-sampled on GLQ grid: aliasing, non-monotone convergence | q_ref 3.4 % (8 % upper half) at 3L; 4L worse than 3L at L=40 | band-limit fields by exact cell averages first |
| L4 | minor | PREM rho_LAB vs WINTERC-G lid base: +30 kg/m^3 mean jump; geoid.py uses mantle_density(T) not T - bed | restoring term 0.9-1.3 % | document or rho_bot(x); use T - bed |
| L5 | minor | floating ice shelves: cavity counted as ice; shelves with C=0 restored by rho_LAB | 0.1 % area each; up to 340 m h_ref | flotation test -> water, C=1; zero ice < 1 m |
| L6 | wording | ice density 910 (README) vs 950 (Fullea 2021 s4.3) | | state choice |
| L7 | minor | linearisation: cross terms Moho x density 3-11 % (70-150 m) for 10 km and 30 kg/m^3 | linearisation.py | document; relinearise if posterior shifts large |
| L8 | minor | drho_lab estimator (+-1 km difference) catches part of jump; negative on 15 % | | assert monotonic; one-sided extrapolation |
| L9 | minor | sub-Moho knot averaging | 0.08 % area | negligible |
| L10 | minor | Moho geoid sheet at mean depth (see E2) | Tibet factor 0.74 at l=40 | local depth |
| C1 | cleared | WINTERC-G Moho and knots are below sea level; t_c = moho + bed correct (H2 cleared) | README layers; ocean crust 10.4 km; isostasy rms 407 vs 457 m | |
| C2 | cleared | ETOPO2 sign handling | | |
| C3 | cleared | regridding alignment | | |
| C4 | cleared | fixed-length-column isostasy correct apart from K (H9: doc wording only) | | |
| C5 | cleared | Airy with real structure | 2072.3 vs 2071.0 m ocean | |
| C6 | cleared | factor fields, weights, node depths | | |

## M — mantle flow (round 1; scripts in audit/mantle/)
| ID | Sev | Finding | Evidence | Proposed fix |
|---|---|---|---|---|
| M1 | major | water term booked inconsistently: K_N removes the whole top mass -sigma/g, but the plate re-adds only (rho_LAB - rho_w C) w = -(sigma + rho_w U)/g; the sea-surface water rho_w U/g following the geoid is never added back (same physics as P4/P5) | geoid error vs flowkernels net geoid: 10.9 % l=2, 6.1 % l=4, 2.6 % l=10, 0.7 % l=40 | split K_q into K_sigma and K_U, apply C(x) to K_U in the plate; add surface sheet rho_w C U/g0 (or self-consistent rho_w C N) |
| M2 | minor | rho_w U term applied on land (should be absent with air above) (H1 confirmed) | K_q land vs sea 1.4 % l=2, <0.2 % l>=15 | same split |
| M3 | minor | U at r_top not R; body force of U on the lid omitted | <= 0.46 % | document |
| M4 | wording | flowkernels PHI is geodesy sign (U = -Phi, Corrieu's U), not physics sign; numbers correct (ledger C4 closed) | DPHI jump -4 pi G k r^2 | fix docstring |
| M5 | minor | radial grid convergence first order: kernels 5.6e-4; deep Gaussian at l=40 up to 9 % | refined 4x | fine for l<=10; refine or Richardson for deep high-l |
| M6 | wording | midpoint layer rule puts effective viscosity jump at ~671.7 km vs density split at 666 km | | wording |
| M7 | minor | cache key omits constants and boundary conditions | LEGACY vs CODATA same key | hash constants and BCs |
| M8 | minor | g0 = GM/R^2 = 9.820 vs model surface gravity 9.882 (0.63 %) | | derive from model or document |
| M9 | minor | uniform fluid core 9900 kg/m^3; net l=2 geoid near-cancellation sensitive | rho_core 11085 changes net PHI_top 9.6 % at l=2 | sensitivity / stated uncertainty at l=2-3 |
| M10 | wording | flowkernels cites "Corrieu et al. 1994 GJI 117 831-852"; correct is 1995 GJI 120 516-523 | Ricard bibliography | fix citation in flowkernels |
| M11 | minor | flowkernels KernelSet.predict trapezoid end weights wrong (not used by package) | | fix in flowkernels |
| cleared | | A-matrix equations (sympy), closed-form benchmark to 4e-14, kernel algebra, degree 1 = centre-of-mass frame, signs vs literature (geoid zero crossings 734 km l=2 ... 2100 km l=8), radial modes, density step at jump, (l,m) ordering | | |

## G — geoid and assembly (round 1; scripts in audit/geoid/)
| ID | Sev | Finding | Evidence | Proposed fix |
|---|---|---|---|---|
| G1 | major | = M1: water following the geoid is loaded (K_q) but its mass never added to the geoid | mantle geoid 10.9 % low at l=2 ... 0.9 % at l=30; adding sheet restores to 1e-14 | self-consistent ocean: sheet rho_w C N_total at R and load -g rho_w C N_total, Neumann series (factor <= 0.11); drop PHI term from K_q |
| G2 | major | P truncates w to lmax before G_w multiplies by laterally varying contrasts: compensating mass leaks | single-degree Te=0 root: geoid error 10-23 %; N_ref 7 % | keep w on fine grid (plate.operator(Q, W=H)), truncate only the geoid |
| G3 | major | G_L sheets at summand mean depth (= E2, L10): fails once compensated | dMoho compensated error 57-72 %; dLAB 28-40 % | exact variable-radius sheets: N_lm = 4 pi G R/((2l+1) g0) [sigma (r/R)^(l+2)]_lm per degree |
| G4 | major | reference(): midpoint layers cut through the Moho; column mass mismatch vs column_mass(); net N_ref is 2 % of structure term so error x50 | net N_ref error 11-16 % at n=120 | split-cell or per-column Gauss quadrature |
| G5 | major (N_ref) / minor (mantle) | deflection_sheets mixes column and sea-level frames (surface at 0, Moho at mean sea-level depth, base at T); rho_bot sampled at T not T - bed | net N_ref 6-13 %; mantle net <= 2.6 % | one geometry, own-depth sheets via exact_sheets |
| G6 | major (data) / wording | observed geoid must have hydrostatic C20 (and C40) removed; "full geoid" wording | correction -115.5 m in C20 component vs model l=2 total 59 m; Chambat vs Nakiboglu 15 m | data = GGM - hydrostatic (Chambat 2010); call it "non-hydrostatic geoid" |
| G7 | minor | degree 1: model masses keep an l=1 dipole (plate masses at other radii than flowkernels' top) | N_1/h_1 3.8 % mantle | compute N_1, frame-correct topography, or exclude l=1 data |
| G8 | minor | degree 0: h_ref and h_m have l=0 parts via C(x); bed mean -2.4 km; h_s cannot absorb (cf. L1, P12) | | exclude l=0 or free constant |
| G9 | wording | test_mantle_kernel_plus_the_deflected_top... proves only an identity; cannot detect G1 | | rename; add limit test |
| G10 | note | plate deflection mass at surface vs flowkernels' r_LAB changes mantle geoid 44-220 %; repo placement more faithful but inconsistent with the self-gravitating flow solution | | state; ties to plate self-gravity item |
| cleared | | sheet formula vs Newton 1e-14; potential sign; dipole limit; mid-lid lumping <= 0.3 %; assembly linearity, offsets, parts, Sobolev adjoints (composite 1.3e-11); h_ref vs bed corr 0.992 at l<=12; surface.py | | |


## Smaller fixes done (2026-11-18)
| ID | What changed | Evidence |
|---|---|---|
| L3 / P9 | 1 degree fields and the ocean function projected onto harmonics to degree 89 (tapered 70-89) and evaluated as band-limited fields on the model's grids | reference load error vs 8L: 3L 0.2 % (was 4.4 %), 4L 0.02 % (3.4 %), 6L 6e-8 (3.1 %); Moho operator exact (1e-15) on any grid |
| L5 | floating ice: ice load removed and the cell treated as ocean (flotation test) | 0.79 % of area, 0.30 % with ice > 100 m (Antarctic shelves) |
| L4 / G5 | lid-base density sampled at T - bed; lid integrated piece by piece | in fix 4 |
| M7 | kernel cache key includes constants and boundary conditions | |
| P14 | land/sea plate test is now an exact manufactured solution (1e-9), not a Galerkin comparison at 2 % | |
| P15 | constant-coefficient filter refuses a laterally varying Te instead of returning Te = 0 | |
| G9 | misleading test renamed; the full limit test is in test_model.py | |
| L2 | dLAB off by default | fix 6 |
| documented | P2 (covariant flat plate), P3 (membrane, 0.2-0.4 %), P6 (one g, 0.4 %), M6 (effective jump 672 km), L6 (ice 910 vs 950), G7 (degree-1 topography not data), M8 (g0 0.6 %), L7/E3 (linearisation, planar balance) | in module docstrings |
| not changed here | M10, M11: flowkernels' Corrieu citation (should be 1995, GJI 120, 516-523) and KernelSet.predict end weights: in the flowkernels repository | |
