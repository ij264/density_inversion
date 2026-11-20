"""Truncation leak for the mantle term: G_m x + G_w P A_m x, w truncated to lmax (repo) vs kept on the fine grid."""
import numpy as np
import t45_sheets as T45
from density_inversion.forward.plate import OceanLoadedPlate
Q, mf, geo, H, LA, LO, deg = T45.Q, T45.mf, T45.geo, T45.H, T45.LA, T45.LO, T45.deg
apply = T45.apply
X = mf.density_space(); x = np.random.default_rng(7).standard_normal(X.dim)
q = apply(mf.operator(X, Q), x)
Nm = apply(geo.mantle_operator(X, Q), x)
down = H.degree_transfer_operator(Q)
GwH = None
for d, f in geo.deflection_sheets(LA, LO):
    op = Q.spectral_operator(geo.sheet_factor(Q, geo.R - d)) @ down @ H.multiplication_operator(np.asarray(f, float))
    GwH = op if GwH is None else GwH + op
for Te in (0.0, 40e3):
    pl = OceanLoadedPlate.from_mantle(mf, Te=Te)
    Na = Nm + apply(geo.deflection_operator(Q), apply(pl.operator(Q), q))
    Nb = Nm + Q.to_components(GwH(pl.operator(Q, W=H)(Q.from_components(q))))
    print(f"Te={Te/1e3:.0f} km: net mantle geoid, repo vs fine-w: {T45.band(Na, Nb)}")
