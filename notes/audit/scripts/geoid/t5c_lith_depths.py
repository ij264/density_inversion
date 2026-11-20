"""Task 5: G_L puts each lithospheric summand's sheet at the summand's MEAN depth. Compare with the sheet at
its own laterally varying depth (sea-level frame, where the reference structure actually is), per summand,
for a random red-spectrum perturbation (l <= 50)."""
import numpy as np
import t45_sheets as T45

Q, geo, ref, H, LA, LO, s, deg = T45.Q, T45.geo, T45.ref, T45.H, T45.LA, T45.LO, T45.s, T45.deg
GL = geo.lithosphere_operator(Q)
fields = ref.factor_fields(LA, LO)
own = {"dMoho": s["moho"], "dLAB": s["lab"], "drho crust": s["moho"] - s["t_c"] / 2}
for n in range(ref.lid_nodes):
    own[f"drho lid node {n}"] = s["node_depths"][n]
depths = geo.lithosphere_depths()
rng = np.random.default_rng(5)
u = rng.standard_normal(Q.dim) * np.where(deg >= 1, (deg + 1.0) ** -1.0, 0)
full = np.zeros(H.dim); full[:Q.dim] = u; ug = np.asarray(H.grid_values(H.from_components(full)))
for j, (label, _, _) in enumerate(ref.summands()):
    x = np.zeros(GL.domain.dim); x[j * Q.dim:(j + 1) * Q.dim] = u
    Nrepo = Q.to_components(GL(GL.domain.from_components(x)))
    Nex = T45.exact_sheets([(own[label], fields[j] * ug)])
    d = own[label]
    print(f"{label:16s} mean depth {depths[j]/1e3:6.1f} km (range {d.min()/1e3:5.1f}-{d.max()/1e3:5.1f}):  {T45.band(Nrepo, Nex)}")

# compensated (Te = 40 km): G_L x + G_w P A_L x, repo vs own-depth sheet for the perturbation
A = ref.operator(Q); P = T45.P; Gw = T45.Gw
for j, (label, _, _) in enumerate(ref.summands()):
    if label not in ("dMoho", "dLAB", "drho lid node 1"):
        continue
    x = np.zeros(GL.domain.dim); x[j * Q.dim:(j + 1) * Q.dim] = u
    xs = GL.domain.from_components(x)
    Nw = T45.apply(Gw, T45.apply(P, Q.to_components(A(xs))))
    Nrepo = Q.to_components(GL(xs)) + Nw
    Nex = T45.exact_sheets([(own[label], fields[j] * ug)]) + Nw
    print(f"compensated {label:16s}: {T45.band(Nrepo, Nex)}")

# fully consistent: perturbation at own depth AND the deflection sheets at their own depths (sea-level frame)
for j, (label, _, _) in enumerate(ref.summands()):
    if label not in ("dMoho", "dLAB", "drho lid node 1"):
        continue
    x = np.zeros(GL.domain.dim); x[j * Q.dim:(j + 1) * Q.dim] = u
    xs = GL.domain.from_components(x)
    w = T45.apply(P, Q.to_components(A(xs)))
    Nrepo = Q.to_components(GL(xs)) + T45.apply(Gw, w)
    full = np.zeros(H.dim); full[:Q.dim] = w; wg = np.asarray(H.grid_values(H.from_components(full)))
    Nex = T45.exact_sheets([(own[label], fields[j] * ug)] + T45.lifted_column(wg, "sealevel"))
    print(f"fully own-depth {label:16s}: {T45.band(Nrepo, Nex)}   |N| rms {np.sqrt(np.sum(Nex[deg>=2]**2)/(4*np.pi)):.3f} m per unit-rms u")
