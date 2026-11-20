"""Task 2b / 5: how much does WHERE the deflection mass sits matter, relative to flowkernels' r_LAB?

Per degree (kernel level, mean contrasts): mantle geoid with the deflection mass
  (a) lumped at r_LAB (flowkernels' geometry),
  (b) at the repo's four radii with global-mean contrasts (surface, mean Moho, mid-lid, T),
and then (c) the repo's laterally varying contrasts (operator level, single-degree input)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.geoid import Geoid
from density_inversion.forward.plate import ElasticPlate

LMAX = 30
mf = MantleFlow(lmax=LMAX)
k = mf.kernels
R, g0, rt, gt = float(k["R"]), float(k["g0"]), float(k["r_lab"]), float(k["g_lab"])
rl, rw = float(k["rho_lab"]), float(k["rho_water"])
G = float(mf.shell.constants.G)
l = np.asarray(k["degrees"], float)[:, None]
S = lambda r: 4 * np.pi * G * r * (r / R) ** (l + 1) / ((2 * l + 1) * g0)
ref = ReferenceLithosphere.from_mantle(mf)
geo = Geoid(mf, ref, ocean=lambda la, lo: np.ones_like(la))
LA, LO = np.meshgrid(ref.data["lat"], ref.data["lon"], indexing="ij")
wl = np.cos(np.radians(LA))
mean = lambda f: float(np.sum(wl * f) / np.sum(wl))
sheets = geo.deflection_sheets(LA, LO)
print("repo deflection sheets (depth km, mean contrast):", [(round(d / 1e3, 2), round(mean(f), 1)) for d, f in sheets])
w = k["K_q"] / ((rl - rw) * gt)
a = k["K_N"] + S(rt) * (rl - rw) * w
b = k["K_N"] + sum(S(R - d) * mean(f) for d, f in sheets) * w
fk = k["K_N_fk"] * (gt / g0) * (rt / R) ** (l + 1)
wts = k["weights"]
rel = lambda x, y: np.sqrt(np.sum(wts * (x - y) ** 2, axis=1) / np.sum(wts * y ** 2, axis=1))
r_ab = rel(b, a)
print("deg  |(b)-(a)|/|(a)|   |(a)-fk|/|fk|   top-mass geoid / net geoid (cancellation)")
for L in (2, 3, 4, 6, 8, 12, 16, 20, 30):
    j = L - 1
    canc = np.sqrt(np.sum(wts * (S(rt) * (rl - rw) * w)[j] ** 2) / np.sum(wts * a[j] ** 2))
    print(f"{L:3d}  {r_ab[j]:.4f}          {rel(a[j:j+1], fk[j:j+1])[0]:.4f}          {canc:.2f}")
