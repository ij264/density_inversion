"""Audit step 2 (plate): pygeoinf flexure operator.
(a) radius scaling D/R^4 on unit sphere vs operator on radius R (variable coefficients);
(b) symbol vs thin-shell theory (bending with sphere curvature change; membrane);
(c) manufactured solution, variable buoyancy, constant D, through OceanLoadedPlate;
(d) variable-D energy check of pygeoinf's operator against an independent finite-difference Hessian.
"""
import numpy as np, pyshtools as pysh
from pygeoinf.symmetric_space import sphere
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.plate import ElasticPlate, OceanLoadedPlate, rigidity
from density_inversion.forward import examples as ex

rng = np.random.default_rng(0)
mf = MantleFlow(lmax=50)
k = mf.kernels
R, rho, rw, g = float(k["R"]), float(k["rho_lab"]), float(k["rho_water"]), float(k["g_lab"])
nu = 0.25
D = rigidity(40e3)
print(f"R={R:.0f} rho_lab={rho:.2f} rho_w={rw} g_lab={g:.4f} D={D:.4e}  D/R^4={D/R**4:.4e} Pa/m")

# ---------------- (a) radius scaling with variable coefficients
L = 16
S1 = sphere.Lebesgue(L, grid="GLQ")
SR = sphere.Lebesgue(L, radius=R, grid="GLQ")
lat, lon = pysh.expand.GLQGridCoord(L)
LA, LO = np.meshgrid(lat, lon, indexing="ij")
Dfield = D * (1 + 0.5 * np.cos(np.radians(LA)) ** 2 * np.sin(np.radians(LO)))
bfield = g * (rho - rw) * (1 + 0.3 * np.sin(np.radians(LA)))
op1 = S1.flexural_operator(Dfield / R**4, nu, bfield)
opR = SR.flexural_operator(Dfield, nu, bfield)
c = rng.standard_normal(S1.dim)
# components on radius R are R * coefficient: feed the same coefficients
q1 = S1.to_components(op1(S1.from_components(c)))
qR = SR.to_components(opR(SR.from_components(R * c))) / R
print("(a) radius scaling, variable D and buoyancy: rel diff =", np.linalg.norm(q1 - qR) / np.linalg.norm(q1))

# ---------------- (b) symbols
l = np.arange(0, 51)
lam = l * (l + 1.0)
b_sea, b_land = (rho - rw) * g, rho * g
pyg = D / R**4 * (lam**2 - (1 - nu) * lam)
bend = D / R**4 * (lam - 2) * (lam - 1 - nu)  # bending energy with kappa = Hess w + w/R^2 g
ETe = 90e9 * 40e3
with np.errstate(divide="ignore", invalid="ignore"):
    # Turcotte et al. 1981 / Beuthe 2008 thin-shell (bending + membrane), restoring kept separate:
    turc = (D / R**4 * lam * (lam - 2) ** 2 + ETe / R**2 * (lam - 2)) / (lam - 1 + nu)
turc[0] = ETe / R**2 * (-2) / (-1 + nu)  # l=0: membrane only, 2 E Te / ((1-nu) R^2)
for name, b in (("sea", b_sea), ("land", b_land)):
    F_pyg = 1 / (pyg + b)
    F_bend = 1 / (bend + b)
    F_turc = 1 / (turc + b)
    print(f"(b) {name}: b={b:.1f} Pa/m. elastic/b at l=10,30,50: {pyg[[10,30,50]]/b}")
    print(f"    F_pyg/F_bend - 1 max over l<=50: {np.max(np.abs(F_pyg/F_bend-1)):.2e}")
    print(f"    F_pyg/F_turc - 1 at l=0,1,2,5,10,30,50: {(F_pyg/F_turc-1)[[0,1,2,5,10,30,50]]}")
print(f"    membrane stiffness E Te / R^2 = {ETe/R**2:.1f} Pa/m")

# ---------------- (c) manufactured solution, variable buoyancy, constant D
L = 40
Lw, Lb = 30, 8
Q = mf.load_space()
Q = sphere.Sobolev(L, Q.order, Q.length_scale)
deg, _ = ex.sphere_ordering(L)
w_true = rng.standard_normal(Q.dim) * (1 + deg) ** -1.0 * 100.0
w_true[deg > Lw] = 0.0
# smooth buoyancy from the ocean function, band-limited to Lb
latf, lonf, Cf = (None, None, None)
from density_inversion.forward.plate import ocean_function
_, _, C1 = ocean_function()
# 1-degree cell-centred grid -> DH 180x360 is cell-centred? use pyshtools DH with sampling 2: 180 x 360 rows at 90..-89
grid = pysh.SHGrid.from_array(C1[:, :], grid="DH")
cl = grid.expand(lmax_calc=Lb, normalization="ortho", csphase=1).coeffs
Csm = lambda la, lo: pysh.expand.MakeGridPoint(cl, lat=la.ravel(), lon=lo.ravel(), norm=4, csphase=1).reshape(la.shape)
Cs_min, Cs_max = None, None
plate = OceanLoadedPlate(40e3, rho - rw, g, R, L, rho_water=rw, ocean=lambda la, lo: Csm(la, lo))
# independent q: spectral bending (pygeoinf symbol, verified in (a)/(b) separately) + exact product b*w on a fine GLQ grid
Nq = 2 * (Lw + Lb) + 4
z, wq = pysh.expand.SHGLQ(Nq)
la, lo = pysh.expand.GLQGridCoord(Nq)
LA, LO = np.meshgrid(la, lo, indexing="ij")
Cgrid = Csm(LA, LO)
print("(c) smoothed C range:", Cgrid.min(), Cgrid.max())
bgrid = g * (rho - rw + rw * (1 - Cgrid))
cw = np.zeros((2, Nq + 1, Nq + 1)); cw[:, : L + 1, : L + 1] = ex.components_to_cilm(w_true, L)
wg = pysh.expand.MakeGridGLQ(cw, z, norm=4, csphase=1)
bw = pysh.expand.SHExpandGLQ(bgrid * wg, wq, z, norm=4, csphase=1)
q_buoy_full = bw
print("    power of b*w above L:", np.sum(bw[:, L + 1 :, :] ** 2) / np.sum(bw ** 2))
q = ex.cilm_to_components(bw, L) + D / R**4 * (lam_ := deg * (deg + 1.0)) ** 2 * w_true - D / R**4 * (1 - nu) * lam_ * w_true
op = plate.operator(Q)
w_rec = Q.to_components(op(Q.from_components(q)))
print("    manufactured recovery rel err:", np.linalg.norm(w_rec - w_true) / np.linalg.norm(w_true))

# ---------------- (d) variable D: energy of pygeoinf's operator vs FD Hessian
L = 24
S = sphere.Lebesgue(L, grid="GLQ")
Nf = 50
z, wq = pysh.expand.SHGLQ(Nf)
la, lo = pysh.expand.GLQGridCoord(Nf)
LAf, LOf = np.meshgrid(la, lo, indexing="ij")
th, ph = np.radians(90 - LAf), np.radians(LOf)
Dfun = lambda t, p: 1.0 + 0.6 * np.cos(t) * np.sin(p) + 0.3 * np.cos(t) ** 2
lat0, lon0 = pysh.expand.GLQGridCoord(L)
L0, O0 = np.meshgrid(lat0, lon0, indexing="ij")
Dgrid_op = Dfun(np.radians(90 - L0), np.radians(O0))
opD = S.flexural_operator(Dgrid_op, nu, 0.0)
cw = rng.standard_normal(S.dim)
Sdeg, _ = ex.sphere_ordering(L)
cw *= (1 + Sdeg) ** -1.5
cw[Sdeg > 12] = 0.0
energy_op = cw @ S.to_components(opD(S.from_components(cw)))
cilm = ex.components_to_cilm(cw, L)
h = 1e-4
def f(t, p):
    return pysh.expand.MakeGridPoint(cilm, lat=90 - np.degrees(t).ravel(), lon=np.degrees(p).ravel(), norm=4, csphase=1).reshape(t.shape)
w0 = f(th, ph)
wt = (f(th + h, ph) - f(th - h, ph)) / (2 * h)
wtt = (f(th + h, ph) - 2 * w0 + f(th - h, ph)) / h**2
wp = (f(th, ph + h) - f(th, ph - h)) / (2 * h)
wpp = (f(th, ph + h) - 2 * w0 + f(th, ph - h)) / h**2
wtp = (f(th + h, ph + h) - f(th + h, ph - h) - f(th - h, ph + h) + f(th - h, ph - h)) / (4 * h * h)
s, cth = np.sin(th), np.cos(th)
# orthonormal-frame Hessian on unit sphere
H11 = wtt
H12 = (wtp - cth / s * wp) / s
H22 = wpp / s**2 + cth / s * wt
lapl = H11 + H22
Hn2 = H11**2 + 2 * H12**2 + H22**2
Dv = Dfun(th, ph)
area = lambda F: np.sum(F * wq[:, None]) * 2 * np.pi / F.shape[1]
E_pyg = area(nu * Dv * lapl**2 + (1 - nu) * Dv * Hn2)
k11, k22 = H11 + w0, H22 + w0
E_shell = area(nu * Dv * (k11 + k22) ** 2 + (1 - nu) * Dv * (k11**2 + 2 * H12**2 + k22**2))
print(f"(d) variable D: <w,Op w> = {energy_op:.6e}; FD energy (flat-covariant) = {E_pyg:.6e}; rel diff {abs(energy_op-E_pyg)/E_pyg:.2e}")
print(f"    shell energy (kappa = Hess w + w g) = {E_shell:.6e}; pygeoinf/shell - 1 = {energy_op/E_shell-1:.3e}")
