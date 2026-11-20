"""Compensation leak from truncating w to lmax before G_w multiplies it by laterally varying contrasts.

Te = 0, dMoho root at single degree l: the in-band (l <= lmax) mass of root + lifted column should be zero.
Compare (a) the repo path, G_w (P A_L x) with w truncated to lmax, with (b) w kept on the fine grid
(P.operator(Q, W=H)) and G_w applied there, truncating only the geoid."""
import numpy as np
import t45_sheets as T45
from density_inversion.forward.plate import OceanLoadedPlate

Q, mf, geo, ref, H, LA, LO, s, deg = T45.Q, T45.mf, T45.geo, T45.ref, T45.H, T45.LA, T45.LO, T45.s, T45.deg
apply, comps, nrm = T45.apply, T45.comps, T45.nrm
A = ref.operator(Q); GL = geo.lithosphere_operator(Q); Gw = geo.deflection_operator(Q)
plate = OceanLoadedPlate.from_mantle(mf, Te=0.0)
P0 = plate.operator(Q)
P0H = plate.operator(Q, W=H)
sheets = geo.deflection_sheets(LA, LO)
down = H.degree_transfer_operator(Q)
GwH = None
for d, f in sheets:
    op = Q.spectral_operator(geo.sheet_factor(Q, geo.R - d)) @ down @ H.multiplication_operator(np.asarray(f, float))
    GwH = op if GwH is None else GwH + op
i = ref.index("dMoho")
m = deg >= 2
for l in (2, 6, 12, 24, 40):
    x = np.zeros(A.domain.dim)
    x[i * Q.dim:(i + 1) * Q.dim] = np.where((deg == l) & (np.asarray(Q.orders) == 0), -1e3, 0.0)
    xs = A.domain.from_components(x)
    q = Q.to_components(A(xs))
    w = apply(P0, q)
    NL = Q.to_components(GL(xs))
    Na = NL + apply(Gw, w)
    wH = P0H(Q.from_components(q))
    Nb = NL + Q.to_components(GwH(wH))
    # in-band residual mass: root mass (q/-g) + lifted mass, at l<=lmax
    tot = np.zeros(LA.shape)
    full = np.zeros(H.dim); full[:Q.dim] = w; wg = np.asarray(H.grid_values(H.from_components(full)))
    rootm = -q / ref.g
    lift_a = comps(sum(f * wg for _, f in sheets))
    lift_b = comps(sum(f * np.asarray(H.grid_values(wH)) for _, f in sheets))
    print(f"l={l:2d}: in-band residual mass / root mass: repo {np.linalg.norm((rootm+lift_a)[m])/np.linalg.norm(rootm[m]):.3f}, "
          f"fine-w {np.linalg.norm((rootm+lift_b)[m])/np.linalg.norm(rootm[m]):.1e};  |N_repo - N_fine|/|N_fine| = {nrm(Na-Nb)/nrm(Nb):.3f}")

# same for the reference geoid with Te = 40 km (the model's own setting)
P40 = OceanLoadedPlate.from_mantle(mf, Te=40e3)
q_ref = ref.reference_load(Q)
Nr = geo.reference(Q, 480)
Na = Nr + apply(Gw, apply(P40.operator(Q), q_ref))
Nb = Nr + Q.to_components(GwH(P40.operator(Q, W=H)(Q.from_components(q_ref))))
print(f"reference geoid (Te=40, n=480): |N_repo - N_fine-w|/|N_fine-w| = {nrm(Na-Nb)/nrm(Nb):.3f}; by band {T45.band(Na, Nb)}")
