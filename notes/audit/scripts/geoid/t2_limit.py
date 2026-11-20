"""Task 2: Te = 0, water everywhere, no lithospheric perturbations, deflection contrast lumped at the flow
top as rho_LAB - rho_w.  Is K_N + S_l(r_top) (rho_LAB - rho_w) P K_q == flowkernels' geoid (continued)?

Kernel-level (per degree and source node) and operator-level (through ElasticPlate and a Geoid whose
deflection_sheets is replaced by one lumped sheet)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.geoid import Geoid
from density_inversion.forward.plate import ElasticPlate

LMAX = 30
mf = MantleFlow(lmax=LMAX)
k = mf.kernels
c = mf.shell.constants
R, g0, rt, gt = float(k["R"]), float(k["g0"]), float(k["r_lab"]), float(k["g_lab"])
rl, rw = float(k["rho_lab"]), float(k["rho_water"])
l = np.asarray(k["degrees"], float)[:, None]
G = c.G
S_top = 4 * np.pi * G * rt * (rt / R) ** (l + 1) / ((2 * l + 1) * g0)
fk_surf = k["K_N_fk"] * (gt / g0) * (rt / R) ** (l + 1)       # flowkernels' geoid at R
w = k["K_q"] / ((rl - rw) * gt)                                  # Te = 0, C = 1: w = K_h0
readded = (rl - rw) * w                                         # = K_q / g_lab
removed = k["K_m_top"]                                          # -sigma_rr/g
model = k["K_N"] + S_top * readded

wts = k["weights"]
def rel(a, b):
    return np.sqrt(np.sum(wts * (a - b) ** 2, axis=1) / np.sum(wts * b ** 2, axis=1))

print("deg  |readded-removed|/|removed|   |model-fk|/|fk|   predicted (rho_w * S_l(r_top)) and (rho_w*K_N_fk vs diff)")
diff_mass = readded - removed
pred_mass = -rw * k["K_N_fk"]          # -rho_w U/g at r_top
for j in range(LMAX):
    if l[j, 0] < 2:
        continue
    if l[j, 0] in (2, 3, 4, 6, 8, 12, 20, 30):
        print(f"{int(l[j,0]):3d}  {rel(readded[j:j+1], removed[j:j+1])[0]:.4f}"
              f"                      {rel(model[j:j+1], fk_surf[j:j+1])[0]:.4f}"
              f"            mass diff == -rho_w*N_fk(r_top): max rel err {np.abs(diff_mass[j]-pred_mass[j]).max()/np.abs(pred_mass[j]).max():.1e}")

# self-consistent factor: model = fk - S_top rho_w N_fk(r_top); N_fk(R) = N_fk(r_top)(g_t/g0)(r_t/R)^(l+1)
ratio = rw * S_top[:, 0] * gt / g0 / ((gt / g0) * (rt / R) ** (l[:, 0] + 1))
print("predicted fractional deficit rho_w S_l(r_top) / continuation:  l=2", ratio[1], " l=4", ratio[3], " l=12", ratio[11], " l=30", ratio[29])
print("i.e. 3 rho_w/((2l+1) rho_mean):", [3 * rw / ((2 * L + 1) * (c.earth_mass / (4 / 3 * np.pi * R ** 3))) for L in (2, 4, 12, 30)])

# integrated check: lower-mantle (1500 km) unit anomaly
i = np.argmin(np.abs(k["r"] - (R - 1500e3)))
print("\nsource at ~1500 km depth, per-node kernel values at l=2..6:")
for j in range(1, 6):
    print(f" l={int(l[j,0])}: fk {fk_surf[j,i]: .4e}  model {model[j,i]: .4e}  K_N(no top) {k['K_N'][j,i]: .4e}  ratio model/fk {model[j,i]/fk_surf[j,i]:.4f}")

# operator-level: the repo's operators with a lumped sheet
ref = ReferenceLithosphere.from_mantle(mf)
geo = Geoid(mf, ref, ocean=lambda la, lo: np.ones_like(la))
geo.deflection_sheets = lambda LA, LO: [(R - rt, np.full(LA.shape, rl - rw))]
Q = mf.load_space(); X = mf.density_space()
P = ElasticPlate.from_mantle(mf, Te=0.0).operator(Q)
op = geo.mantle_operator(X, Q) + geo.deflection_operator(Q) @ P @ mf.operator(X, Q)
import density_inversion.operators as po
fkop = po.density_operator(X, dict(k, K_fks=fk_surf), "fks", Q)
rng = np.random.default_rng(0)
x = X.from_components(rng.standard_normal(X.dim))
a = Q.to_components(op(x)); b = Q.to_components(fkop(x))
deg = np.asarray(Q.degrees)
for L in (2, 3, 4, 8, 16, 30):
    m = deg == L
    print(f"operator-level l={L}: |model-fk|/|fk| = {np.linalg.norm(a[m]-b[m])/np.linalg.norm(b[m]):.4f}")
# with the rho_w N_fk term added back as a sheet at r_top the identity should be exact
fix = po.density_operator(X, dict(k, K_fix=S_top * rw * k["K_N_fk"]), "fix", Q)
a2 = a + Q.to_components(fix(x))
m = deg >= 2
print("with + S(r_top) rho_w N_fk(r_top) added:", np.linalg.norm(a2[m] - b[m]) / np.linalg.norm(b[m]))
# lateral-structure radii: actual repo sheets with water everywhere
geo2 = Geoid(mf, ref, ocean=lambda la, lo: np.ones_like(la))
op2 = geo2.mantle_operator(X, Q) + geo2.deflection_operator(Q) @ P @ mf.operator(X, Q)
a3 = Q.to_components(op2(x))
for L in (2, 4, 8, 16, 30):
    mm = deg == L
    print(f"repo sheets (radii spread, lateral contrasts) l={L}: |model-fk|/|fk| = {np.linalg.norm(a3[mm]-b[mm])/np.linalg.norm(b[mm]):.4f}, vs lumped {np.linalg.norm(a3[mm]-a[mm])/np.linalg.norm(a[mm]):.4f}")
