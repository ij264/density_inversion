"""Task 8: degree-1 consistency. The geoid is zeroed at l=1 (centre-of-mass frame), the topography is not.
If the model's own masses (with the plate where the repo puts them) have a net l=1 mass dipole, the CM of the
model Earth is displaced by delta = N_1 (the l=1 'geoid' it would have), and the topography in the CM frame
is h_1 - N_1. Quantify N_1 / h_1 for the mantle term and for the reference term."""
import numpy as np
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel
from density_inversion.forward.geoid import Geoid
import density_inversion.operators as po

LMAX = 12
mf = MantleFlow(lmax=LMAX)
fm = ForwardModel(mf)
Q, deg = fm.Q, np.asarray(fm.Q.degrees)
k = mf.kernels
c = mf.shell.constants
s = mf.shell
R, g0 = float(k["R"]), float(k["g0"])
# l=1 'internal' kernel (anomaly + CMB, top removed), not zeroed
u, sig, phi, sb, pb = fk.solve_degree_traces(s, 1).T
rt, gt = s.r[-1], s.gravity[-1]
geo_pref = c.visc_ref / (c.rho_ref * rt * gt)
c_top = 4 * np.pi * c.G / 3 * rt / gt * c.rho_ref
internal = geo_pref * (phi + c_top * sig) * (gt / g0) * (rt / R) ** 2
print("l=1: max|phi_top| (should be ~0, CM frame):", np.abs(phi).max(), " vs max|c_top sig|:", np.abs(c_top * sig).max())


class Geo1(Geoid):
    def sheet_factor(self, Q, radius):
        l = np.asarray(Q.degrees, float); r = float(radius)
        S = 4 * np.pi * self.G * r * (r / self.R) ** (l + 1) / ((2 * l + 1) * self.g0)
        return np.where(l >= 1, S, 0.0)


g1 = Geo1(mf, fm.ref)
Gw1 = g1.deflection_operator(Q)
KN1 = k["K_N"].copy()
idx = np.concatenate([np.arange(0, np.flatnonzero(np.diff(k['eta']) != 0)[0] + 1)])  # not needed; rebuild below
# rebuild K_N row for l=1 on the duplicated-node quadrature
n = s.n
eta = s.viscosity(1); i_jump = int(np.flatnonzero(np.diff(eta) != 0)[0]) + 1
idx = np.concatenate([np.arange(0, i_jump + 1), np.arange(i_jump, n)])
KN1[0] = internal[idx]
Gm1 = po.density_operator(fm.X, dict(k, K_N1=KN1), "N1", Q)

rng = np.random.default_rng(3)
xm = rng.standard_normal(fm.X.dim)
a = fm._apply
hm = a(fm.P, a(fm.A_m, xm))
N1m = a(Gm1, xm) + a(Gw1, hm)
m1 = deg == 1
print(f"mantle term: |N_1| / |h_1| = {np.linalg.norm(N1m[m1])/np.linalg.norm(hm[m1]):.4f}  (|h_1| = {np.linalg.norm(hm[m1]):.1f} m)")
h_ref, _ = fm.offset()
N1r = g1.reference(Q, 480) + a(Gw1, h_ref)
print(f"reference term: |N_1| / |h_ref,1| = {np.linalg.norm(N1r[m1])/np.linalg.norm(h_ref[m1]):.4f}  (|h_ref,1| = {np.linalg.norm(h_ref[m1]):.1f} m, |N_1| = {np.linalg.norm(N1r[m1]):.2f} m)")
