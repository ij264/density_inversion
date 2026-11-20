"""Tasks 4 and 5: placement of the deflection and reference sheets, l <= 50.

exact_sheets(): the geoid (l>=2) of thin sheets sigma_k(x) at laterally varying radii r_k(x), computed
exactly, degree by degree:  N_lm = 4 pi G R / ((2l+1) g0) [ sum_k sigma_k (r_k/R)^(l+2) ]_lm.
"""
import sys
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.geoid import Geoid
from density_inversion.forward.plate import OceanLoadedPlate

LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 50
mf = MantleFlow(lmax=LMAX)
Q = mf.load_space()
ref = ReferenceLithosphere.from_mantle(mf)
geo = Geoid(mf, ref)
R, g0, G, T = geo.R, geo.g0, geo.G, ref.T
H = ref._fine(Q)
LA, LO = ref._grid(H)
s = ref.structure(LA, LO)
C = np.asarray(geo.ocean(LA, LO), float)
deg = np.asarray(Q.degrees)


def comps(v):
    return H.to_components(H.from_grid_values(v))[: Q.dim]




def exact_sheets(sheets):
    """sheets: list of (depth field (m below R), sigma field (kg/m^2)) -> N components (l>=2)."""
    out = np.zeros(Q.dim)
    for L in range(2, LMAX + 1):
        acc = np.zeros(LA.shape)
        for d, sig in sheets:
            acc += sig * ((R - d) / R) ** (L + 2)
        c = comps(acc)
        m = deg == L
        out[m] = 4 * np.pi * G * R / ((2 * L + 1) * g0) * c[m]
    return out


def lifted_column(w, frame, n=200):
    """Linear (in w) sheets of the reference column lifted by w, all contrasts at their own depth,
    lid gradient resolved. frame='column': surface at R; 'sealevel': surface at R + bed."""
    off = np.zeros(LA.shape) if frame == "column" else -s["bed"]  # depth of the column top below R
    rho_top = ref.sample("rho_mantle", LA, LO)[0]
    sh = [(off, (s["rho_c"] - geo.rho_water * C) * w), (off + s["t_c"], (rho_top - s["rho_c"]) * w)]
    zs = s["t_c"][None] + s["t_l"][None] * (np.arange(n + 1)[:, None, None] / n)  # below own surface
    rho = np.array([ref.mantle_density(z - s["bed"], LA, LO) for z in zs])
    for k in range(n):
        sh.append((off + 0.5 * (zs[k] + zs[k + 1]), (rho[k + 1] - rho[k]) * w))
    rho_bot = ref.mantle_density(ref.T, LA, LO)
    sh.append((off + T, (geo.rho_lab - rho[-1]) * w))
    return sh


def nrm(x, L=None):
    m = deg >= 2 if L is None else deg == L
    return np.linalg.norm(x[m])


def band(x, y, bands=((2, 4), (5, 10), (11, 20), (21, 35), (36, 50))):
    out = []
    for a, b in bands:
        if a > LMAX:
            break
        m = (deg >= a) & (deg <= min(b, LMAX))
        out.append(f"l{a}-{b}: {np.linalg.norm(x[m]-y[m])/np.linalg.norm(y[m]):.3f}")
    return "  ".join(out)


apply = lambda op, c: op.codomain.to_components(op(op.domain.from_components(c)))
Gw = geo.deflection_operator(Q)
P = OceanLoadedPlate.from_mantle(mf, Te=40e3).operator(Q)
q_ref = ref.reference_load(Q)
h_ref = apply(P, q_ref)

# a "dynamic" deflection: random, red spectrum, ~1 km rms
rng = np.random.default_rng(1)
wdyn = rng.standard_normal(Q.dim) * np.where(deg >= 1, (deg + 1.0) ** -1.5, 0)
wdyn *= 1e3 / np.sqrt(np.sum(wdyn ** 2) / (4 * np.pi))
fields = {}
for name, wc in (("h_ref", h_ref), ("w_dyn", wdyn)):
    full = np.zeros(H.dim); full[: Q.dim] = wc
    wg = np.asarray(H.grid_values(H.from_components(full)))
    fields[name] = (wc, wg)

for name, (wc, wg) in fields.items():
    N_repo = apply(Gw, wc)
    # repo's own sheets via exact_sheets with constant depths (checks the sheet operator path)
    rep = [(np.full(LA.shape, d), f * wg) for d, f in geo.deflection_sheets(LA, LO)]
    N_rep_exact = exact_sheets(rep)
    print(f"\n[{name}] repo G_w vs exact-sheet evaluation of the same sheets: {nrm(N_repo-N_rep_exact)/nrm(N_repo):.2e}")
    for frame in ("column", "sealevel"):
        N_ex = exact_sheets(lifted_column(wg, frame))
        print(f"  repo vs exact lifted column ({frame:8s} frame): rel diff by band: {band(N_repo, N_ex)}")
    # isolate the mid-lid lumping: column frame, lid lumped at mid-lid but Moho at t_c(x)
    sh = lifted_column(wg, "column")
    lid = sh[2:-1]
    tot = sum(sg for _, sg in lid)
    mid = 0.5 * (s["t_c"] + T)
    lumped = sh[:2] + [(mid, tot)] + sh[-1:]
    print(f"  mid-lid lumping alone (column frame):                 {band(exact_sheets(lumped), exact_sheets(sh))}")
    # isolate mean-radius Moho: column frame exact vs Moho at mean(t_c)
    wl = np.cos(np.radians(LA)); mean = lambda f: float(np.sum(wl * f) / np.sum(wl))
    mm = sh[:1] + [(np.full(LA.shape, mean(s["t_c"])), sh[1][1])] + sh[2:]
    print(f"  Moho at mean radius alone (column frame):             {band(exact_sheets(mm), exact_sheets(sh))}")
    print(f"  mean(moho below sea level) = {mean(s['moho'])/1e3:.2f} km, mean(t_c) = {mean(s['t_c'])/1e3:.2f} km, mean(bed) = {mean(s['bed'])/1e3:.2f} km")
