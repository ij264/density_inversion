"""Task 3: compensated columns vs the dipole formula.

For sheets sigma_i at depths d_i with sum sigma_i = 0 (local compensation), the exact sheet sum
N_l = sum S_l(R - d_i) sigma_i expands to
    N_l = -(4 pi G / g0) (l+2)/(2l+1) * [sum sigma_i d_i]_l  + O((l d / R)^2),
whose flat-Earth limit (l >> 1) is Haxby & Turcotte's N = -(2 pi G / g) int drho(z) z dz (z depth).

(1) kernel level: an Airy root, constant contrasts, the repo's S_l and the repo's sheet depths;
(2) operator level: dMoho root through G_L + G_w P A_L with Te = 0 (OceanLoadedPlate, real coastlines);
(3) the reference lithosphere with Te = 0: reference() + G_w P q_ref vs the dipole field of the columns."""
import numpy as np
import t45_sheets as T45

Q, mf, geo, ref, H, LA, LO, s, deg = T45.Q, T45.mf, T45.geo, T45.ref, T45.H, T45.LA, T45.LO, T45.s, T45.deg
apply, comps = T45.apply, T45.comps
R, g0, G, T = geo.R, geo.g0, geo.G, ref.T
LMAX = Q.lmax
L = np.arange(2, LMAX + 1)

# (1) Airy root: crust 2900, mantle 3370 (rho_lab), water on top; 10 km root at the 30 km Moho
rc, rm, rw = 2900.0, geo.rho_lab, geo.rho_water
root, dM = 10e3, 30e3
sig_root = -(rm - rc) * root
w = (rm - rc) * root / (rm - rw)                    # Te = 0 uplift
sheets = [(dM, sig_root), (0.0, (rc - rw) * w), (dM, (rm - rc) * w)]   # lifted column: surface & Moho (lid uniform, base contrast 0)
S = lambda d: 4 * np.pi * G * (R - d) * ((R - d) / R) ** (L + 1) / ((2 * L + 1) * g0)
N_exact = sum(S(d) * sg for d, sg in sheets)
D = sum(sg * d for d, sg in sheets)
N_sph = -(4 * np.pi * G / g0) * (L + 2) / (2 * L + 1) * D
N_flat = -(2 * np.pi * G / g0) * D
print(f"(1) Airy root 10 km at 30 km, uplift {w:.0f} m; flat dipole N = {N_flat:.3f} m")
for l in (2, 4, 8, 16, 30, 50):
    i = l - 2
    print(f"   l={l:2d}: sheets {N_exact[i]: .4f}   spherical dipole {N_sph[i]: .4f} (rel {abs(N_exact[i]-N_sph[i])/abs(N_sph[i]):.1e})   flat {N_flat: .4f}")

# (2) operator level, dMoho root at single degrees, Te = 0, real ocean function
from density_inversion.forward.plate import OceanLoadedPlate
A = ref.operator(Q); GL = geo.lithosphere_operator(Q); Gw = geo.deflection_operator(Q)
P0 = OceanLoadedPlate.from_mantle(mf, Te=0.0).operator(Q)
i = ref.index("dMoho")
depths = geo.lithosphere_depths()
print("(2) operator level: dMoho = -1 km at (l,0); N_model vs spherical-dipole of the model's own sheets")
for l in (2, 6, 12, 24, 40):
    x = np.zeros(A.domain.dim)
    u = np.where((deg == l) & (np.asarray(Q.orders) == 0), -1e3, 0.0)
    x[i * Q.dim:(i + 1) * Q.dim] = u
    xs = A.domain.from_components(x)
    wq = apply(P0, Q.to_components(A(xs)))
    N = Q.to_components(GL(xs)) + apply(Gw, wq)
    full = np.zeros(H.dim); full[:Q.dim] = u; ug = np.asarray(H.grid_values(H.from_components(full)))
    full[:Q.dim] = wq; wg = np.asarray(H.grid_values(H.from_components(full)))
    sig = [(depths[0], s["drho_moho"] * ug)] + [(d, f * wg) for d, f in geo.deflection_sheets(LA, LO)]
    Dg = sum(sg * d for d, sg in sig)
    resid_mass = sum(sg for _, sg in sig)
    Dc = comps(Dg); Mc = comps(resid_mass)
    fac = np.where(deg >= 2, -(4 * np.pi * G / g0) * (deg + 2) / (2 * deg + 1.0), 0)
    Nd = fac * Dc
    Nmass = geo.sheet_factor(Q, R) * Mc
    m = deg >= 2
    print(f"   l={l:2d}: |N|={np.linalg.norm(N[m]):.3e}  |N-N_dip|/|N| = {np.linalg.norm(N[m]-Nd[m])/np.linalg.norm(N[m]):.3f}"
          f"   |N-N_dip-N(uncompensated mass at R)|/|N| = {np.linalg.norm(N[m]-Nd[m]-Nmass[m])/np.linalg.norm(N[m]):.3f}"
          f"   rms residual mass {np.sqrt(np.mean(resid_mass**2)):.1f} kg/m^2 vs root mass {np.sqrt(np.mean((s['drho_moho']*ug)**2)):.1f}")

# (3) reference lithosphere, Te = 0
q_ref = ref.reference_load(Q)
h0 = apply(P0, q_ref)
for n in (120, 960):
    Nref = geo.reference(Q, n) + apply(Gw, h0)
    # dipole field of the columns (column frame) + lifted sheets
    dz = T / n
    Dg = np.zeros(LA.shape); Mg = np.zeros(LA.shape)
    wl = np.cos(np.radians(LA))
    for z in (np.arange(n) + 0.5) * dz:
        rho = np.where(z < s["t_c"], s["rho_c"], ref.mantle_density(z - s["bed"], LA, LO))
        sg = dz * (rho - np.sum(wl * rho) / np.sum(wl))
        Dg += sg * z; Mg += sg
    ice = ref.RHO_ICE * ref.sample("ice_thickness", LA, LO); Mg += ice
    full = np.zeros(H.dim); full[:Q.dim] = h0; hg = np.asarray(H.grid_values(H.from_components(full)))
    for d, f in geo.deflection_sheets(LA, LO):
        Dg += f * hg * d; Mg += f * hg
    fac = np.where(deg >= 2, -(4 * np.pi * G / g0) * (deg + 2) / (2 * deg + 1.0), 0)
    Nd = fac * comps(Dg) + geo.sheet_factor(Q, R) * comps(Mg)
    print(f"(3) reference, Te=0, n={n}: rms N_ref {np.sqrt(np.sum(Nref[deg>=2]**2)/(4*np.pi)):.2f} m, "
          f"|N_ref - (dipole + residual-mass)|/|N_ref| = {T45.nrm(Nref-Nd)/T45.nrm(Nref):.3f}; "
          f"dipole-only {T45.nrm(Nref-fac*comps(Dg))/T45.nrm(Nref):.3f}; residual mass rms {np.sqrt(np.mean(comps(Mg)[deg>=2]**2)):.1f}")
    print("    by band (dipole+mass):", T45.band(Nd, Nref))
