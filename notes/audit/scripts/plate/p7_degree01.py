"""Audit step 2: degree 0 / 1 of the land/sea deflection, and a volume-conserving variant.
The mantle top's mean radial displacement must vanish (incompressible flow; flowkernels has no l=0),
but drho(x) couples a degree>=2 load into w_00. Fix: add a uniform pressure p0 so that w_00 = 0:
    w = P q + p0 P 1,  p0 = -(P q)_00 / (P 1)_00.
"""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel
from density_inversion.forward import examples as ex

L = 50
mf = MantleFlow(lmax=L)
k = mf.kernels
R = float(k["R"])
fm = ForwardModel(mf, include_reference=False)
X, Q = fm.X, fm.Q
deg, _ = ex.sphere_ordering(L)
x = ex.project(X, ex.power_law(R, float(k["r_cmb"]) / R, float(k["r_lab"]) / R, lmax=L))
q = fm._apply(fm.A_m, x)
w = fm._apply(fm.P, q)
print("load: |q_00|/|q| =", abs(q[0]) / np.linalg.norm(q), " |q_l=1|/|q| =", np.linalg.norm(q[deg == 1]) / np.linalg.norm(q))
print("deflection: w_00 / sqrt(4pi) (mean, m) =", w[0] / np.sqrt(4 * np.pi), "; |w_00|/|w| =", abs(w[0]) / np.linalg.norm(w),
      "; |w_l=1|/|w| =", np.linalg.norm(w[deg == 1]) / np.linalg.norm(w))
one = np.zeros(Q.dim); one[0] = np.sqrt(4 * np.pi)
P1 = fm._apply(fm.P, one)
p0 = -w[0] / P1[0]
wc = w + p0 * P1
print("volume-conserving correction: p0 =", p0, "Pa; |wc - w|/|w| (l>=2) =",
      np.linalg.norm((wc - w)[deg >= 2]) / np.linalg.norm(w[deg >= 2]))
# the reference lithosphere load (degree 0 removed) through P
fr = ForwardModel(mf, include_reference=True)
h_ref, _ = fr.offset()
print("reference topography: mean h_ref =", h_ref[0] / np.sqrt(4 * np.pi), "m; rms(l>=1) =",
      np.linalg.norm(h_ref[deg >= 1]) / np.sqrt(4 * np.pi), "m")
# negative uniform-at-sea style check: sign of w for a negative local load (Pacific)
from density_inversion.forward.plate import OceanLoadedPlate
lat, lon, _ = ex.to_grid(w, L)
p0r = -h_ref[0] / P1[0]
hc = h_ref + p0r * P1
print("reference, volume-conserving: p0 =", p0r, "Pa; |hc-h_ref|/|h_ref| (l>=2) =",
      np.linalg.norm((hc - h_ref)[deg >= 2]) / np.linalg.norm(h_ref[deg >= 2]),
      "; max grid change (m) =", abs(ex.to_grid(hc - h_ref, L)[2]).max())
