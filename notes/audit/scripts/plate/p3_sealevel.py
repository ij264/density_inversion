"""Audit step 2: sea level follows the geoid.
Correct (first order) plate equation, water at sea filling to the geoid N_s:
    D nabla^4 w + (rho_LAB - rho_w C) g w = q_sigma - C rho_w g N_s
The code's load is q = q_sigma - rho_w g N_fk (flowkernels' geoid at r_LAB, top deflected freely, water everywhere).
This script measures: |q_phi|/|q|, the deflection error P[(C N_s - N_fk) rho_w g], and N_s / h (observable h - N).
"""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel
from density_inversion.forward import examples as ex
from density_inversion import operators as po
from density_inversion.forward.plate import sample_ocean_function

L = 50
mf = MantleFlow(lmax=L)
k = mf.kernels
R, rw, g = float(k["R"]), float(k["rho_water"]), float(k["g_lab"])
fm = ForwardModel(mf, include_reference=False)
X, Q = fm.X, fm.Q
deg, _ = ex.sphere_ordering(L)
apply = fm._apply
A_Nfk = po.density_operator(X, k, "N_fk", Q)

r_lab = float(k["r_lab"])
fields = {
    "power_law k^-1, whole mantle": ex.power_law(R, float(k["r_cmb"]) / R, r_lab / R, lmax=L),
    "dense blob 300 km": ex.dense_blob(R, lmax=L),
    "dense blob 1500 km, 1000 km wide": ex.dense_blob(R, depth=1500e3, width=1000e3, lmax=L),
}

def rms_by_deg(c):
    return np.array([np.sqrt(np.sum(c[deg == l] ** 2)) for l in range(L + 1)])

for name, prof in fields.items():
    x = ex.project(X, prof)
    p = fm.parts(x_m=x)
    h = p["topography"]["dynamic"]
    Ns = p["geoid"]["total"]
    q = apply(fm.A_m, x)
    Nfk = apply(A_Nfk, x)
    q_phi = -rw * g * Nfk
    # correct water/geoid term, C N_s on a grid
    lat, lon, Ng = ex.to_grid(Ns, L)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    C = sample_ocean_function(LA, LO)
    import pyshtools as pysh
    cilm = pysh.expand.SHExpandDH(C * Ng, norm=4, sampling=2, csphase=1, lmax_calc=L)
    CN = ex.cilm_to_components(cilm, L)
    e = -rw * g * CN - q_phi  # what the load is missing
    dw = apply(fm.P, e)
    n = lambda v: np.linalg.norm(v[deg >= 2])
    print(f"== {name}")
    print(f"   |q_phi|/|q| (l>=2) = {n(q_phi)/n(q):.3f}; |q_phi|/|q| by l=2,5,10,20,50:",
          np.round((rms_by_deg(q_phi) / rms_by_deg(q))[[2, 5, 10, 20, 50]], 3))
    print(f"   |N_s|/|h| (l>=2) = {n(Ns)/n(h):.4f}; by l=2,5,10,20,50:",
          np.round((rms_by_deg(Ns) / rms_by_deg(h))[[2, 5, 10, 20, 50]], 4))
    print(f"   |N_fk(LAB)|/|N_s| = {n(Nfk)/n(Ns):.3f}")
    print(f"   deflection error |P e|/|h| = {n(dw)/n(h):.4f}; by l=2,5,10,20,50:",
          np.round((rms_by_deg(dw) / rms_by_deg(h))[[2, 5, 10, 20, 50]], 4))
    # grid view: on land, the code applies -rho_w g N_fk where no water is
    _, _, hg = ex.to_grid(h, L)
    _, _, dwg = ex.to_grid(dw, L)
    print(f"   max |h| = {abs(hg).max():.1f} m; max |P e| = {abs(dwg).max():.2f} m; max |N_s| = {abs(Ng).max():.2f} m")
