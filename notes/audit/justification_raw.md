# Research justification — raw agent verdicts (to be merged into justification.tex)

## A. Flow, plate, geoid (agent A)
| Choice | Verdict | Key evidence | Run to settle |
|---|---|---|---|
| Instantaneous compressible self-gravitating flow | justified | Richards & Hager 1984 timescale argument; Richards et al. 2023 EPSL 602 117964 use same kernels (Hager 1985 + Corrieu 1995) | compressible/SG vs incompressible kernels (defensive) |
| x30 viscosity jump | caveat | Hager 1984 needs >= x30; Panasyuk & Hager 2000 ~x10 families; Cadek & Fleitout 2003 >= x100; Lithgow-Bertelloni & Richards 1998 x50 | finite-difference runs x10/x30/x100 (+ low-viscosity asthenosphere) vs adjoint prediction; optional model averaging over published profiles |
| Flow top 105 km sphere | caveat | Le Stunff & Ricard 1995 GJI 122 closest precedent; Davies et al. 2019 need lithospheric structure; cratonic LAB > 260 km | flow top 80/105/150; cratonic roots into lithosphere layer (BoE +-10 % at l=50, <2 % l<=10) |
| Elastic plate Te 40, land/sea | justified (loading), caveat (Te) | BoE: Te 0-80 changes l=50 by up to 42 %, <2 % for l<=10; air/water factor 1.45 | Te 0/20/40/80 + variable Te |
| Full non-hydrostatic geoid l=2-50 | caveat | Chambat et al. 2010 GJI 183 727-732: nonhydrostatic C20 = -5.10e-6 (Nakiboglu gives -4.43e-6, ~4 m difference) | l=2-50 vs 4-50; Chambat vs Nakiboglu |
| Plate self-gravity neglected in the flow | SHOULD CHANGE | BoE feedback 3 drho/((2l+1) rho_bar): 25 % (water) - 36 % (air) at l=2, 6-9 % at l=10 | include plate/sheet masses in the Poisson source of the coupled solve; compare l=2-5 |
| Viscosity known, adjoint kernels | caveat; better: add FD runs + model averaging | kernels nonlinear in viscosity; Panasyuk & Hager 2000 nonuniqueness | as above |
| L = 50 | caveat | Hoggard 2016 power to l~40; Richards 2023 use l<=30 | L=30 vs 50; l>50 power as noise |
Missing items an examiner will raise: total vs residual debate (Molnar et al. 2015; Colli et al. 2016; Yang & Gurnis 2016); lateral viscosity (Cadek & Fleitout 2003; Ghosh et al. 2010); plate-driven flow (Ricard & Vigny 1989); 660 km phase-boundary deflection (Thoraval et al. 1995 Nature 375 777-780); why velocity-density scaling fails (Steinberger 2016/2019 isopycnal patches); reference-state consistency PREM vs WINTERC-G at 105 km.

## B. Lithosphere, priors, formulation (agent B)
What WINTERC-G fitted (from GJI full text): ETOPO2 elevation (low weight, local isostasy); XGM2016 geoid + free-air + GOCE gradients with **degrees 2-3 removed**; crust 3 layers, Moho prior CRUST1.0 with Szwillus et al. 2019 sigma (floors 3 km ocean / 7 km continent); LAB = 1300 C isotherm; sublithospheric density also tuned to gravity down to 400 km. Lebedev is 2nd author.
| Choice | Verdict | Key point | Run |
|---|---|---|---|
| Total topography + lithosphere as unknowns | justified | crustal errors (Moho 3-7 km -> 0.4-1 km topography) exceed the +-500 m dynamic signal; plausible cause of the earlier implausible profiles | residual formulation with crustal covariance propagated vs total formulation |
| Linear-Gaussian, first order | caveat | finite-amplitude correction ~ (l+2) dh/(2r): ~4 % at l=50 for 10 km | exact finite-amplitude forward for posterior mean; chi^2 posterior predictive, mask subduction zones |
| WINTERC-G prior mean (circularity) | caveat; TESTS MANDATORY; should change: use only lithospheric part | double counting understates sigma up to sqrt 2; degrees 2-3 geoid clean | CRUST1.0+CAM2016 reference; hold out residual depths; prior-predictive chi^2; ask Lebedev/Fullea for a seismic+heat-flow-only version |
| Parametrisation (fixed 105 km lid base, LAB + lid density, 3 nodes) | SHOULD CHANGE | cratonic keels 105-260 km fall into the mantle; LAB displacement and lid density describe the same thermal physics twice; independent vs correlated nodes change column-mean prior sigma by 1.4 | tie lid base to LAB or one continuous density field below the Moho with LAB-dependent prior; correlate LAB and lid |
| h_s | caveat | degenerate with shallow (asthenospheric) dynamic topography at l>30 — exactly Hoggard's 10^3 km signal | with/without h_s, cutoffs 20/30/40, synthetic l=30-40 asthenospheric anomaly; consider Airy-compensated geoid for h_s |
| Priors from inter-model spread + evidence | caveat | models share data (Knutti et al. 2013 analogue); SL2013sv-LAB not independent of WINTERC-G; flat evidence for weakly constrained amplitudes | evidence profiles/Hessian; fixed vs evidence amplitudes; marginalise amplitudes |
| Data and error budgets | SHOULD CHANGE | geoid error "dominated by crustal correction" double counts once crust is an unknown; residual depths share corrections with topography | data covariance = gravity-model + modelling error only; seismic crustal thicknesses as point Moho data; residual depths held out |
