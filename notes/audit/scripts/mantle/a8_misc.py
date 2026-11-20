"""Task 7: miscellaneous checks: cache vs fresh kernels, cache key coverage, surface g vs g0, core density
sensitivity, thickness of the last layer, K_h0 vs flowkernels surftopo, location of the effective viscosity jump."""
import dataclasses
import numpy as np
import flowkernels as fk
from density_inversion.forward import mantle as M
from density_inversion.forward.mantle import MantleFlow, viscosity_jump, reference_model, truncated

a = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40, cache=True)
b = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40, cache=False)
print("cache file:", M.CACHE / f"mantle_{a._key()}.npz")
for k in ("K_q", "K_N", "K_N_fk"):
    print(f"cached vs fresh {k}: max rel diff {np.max(abs(a.kernels[k] - b.kernels[k])) / np.max(abs(b.kernels[k])):.1e}")

# cache key ignores constants
s2 = dataclasses.replace(b.shell, constants=fk.CODATA2018)
b2 = MantleFlow.__new__(MantleFlow); b2.__dict__.update(b.__dict__); b2.shell = s2
print("cache key identical for LEGACY and CODATA2018 constants:", b2._key() == b._key())

base = reference_model()
c = base.constants
print("flowkernels g at surface", base.gravity[-1], " g0 = GM/R^2", c.G * c.earth_mass / base.r[-1]**2,
      " rel", base.gravity[-1] / (c.G * c.earth_mass / base.r[-1]**2) - 1)
print("flowkernels' integrated Earth mass", base.gravity[-1] * base.r[-1]**2 / c.G, "vs earth_mass", c.earth_mass)
r = b.shell.r
print("last layers thickness km", np.diff(r)[-3:] / 1e3, " first", np.diff(r)[:2] / 1e3)

# K_h0 equals flowkernels' surftopo of the truncated shell
ks = fk.kernels(b.shell, [2, 10])
k = b.kernels
idx = np.r_[0:k["r"].size]
Kh = np.interp(b.shell.r, k["r"], k["K_h0"][1])
print("K_h0 vs fk surftopo (l=2) max rel", np.max(abs(Kh - ks.surftopo[0])) / np.max(abs(ks.surftopo[0])))

# effective viscosity jump: midpoint quadrature averages the nodes
eta = b.shell.viscosity(1)
i = np.flatnonzero(np.diff(eta))[0]
print(f"eta nodes around jump: depth {(b.R - r[i]) / 1e3:.2f} km eta {eta[i]}, depth {(b.R - r[i + 1]) / 1e3:.2f} km "
      f"eta {eta[i + 1]}; layer between gets midpoint eta {0.5 * (eta[i] + eta[i + 1])}; density split at "
      f"{(b.R - k['r_jump']) / 1e3:.2f} km")

# core density: rho_core 9900 vs gm_core-implied 11085 -> effect on K_N, K_q
for rc in (11085.0,):
    cst = dataclasses.replace(c, rho_core=rc)
    sh = dataclasses.replace(b.shell, constants=cst)
    for l in (2, 4, 10):
        t0 = fk.solve_degree_traces(b.shell, l); t1 = fk.solve_degree_traces(sh, l)
        print(f"rho_core {rc:.0f}: l={l} rel change sig_top {np.max(abs(t1[:,1]-t0[:,1]))/np.max(abs(t0[:,1])):.1e}"
              f" phi_top {np.max(abs(t1[:,2]-t0[:,2]))/np.max(abs(t0[:,2])):.1e}")
