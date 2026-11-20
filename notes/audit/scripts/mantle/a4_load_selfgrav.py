"""Task 2: the load on a thin plate with self-gravity, water vs air above; and the geoid bookkeeping of the
water that follows the geoid.

Vertical force balance on a plate column (LAB at r_t to surface R, density rho_L), lifted rigidly by w, with
Eulerian incremental mantle stress sigma (tension +) and geodesy-sign potential U = -Phi:
   0 = -sigma(r_t) - rho_m g w      (Lagrangian traction from the mantle)
       - rho_a U(R) + rho_a g w     (Lagrangian pressure of a fluid overburden of density rho_a in equilibrium)
       + int_{r_t}^{R} rho_L dU/dr dr  (body force of the potential perturbation on the plate)
       - D nabla^4 w
=> q_true = -(sigma + rho_a U(R)) + int rho_L U' dr,  restoring (rho_m - rho_a) g w.
Code: q_code = -(sigma + rho_w U(r_t)) everywhere.
"""
import numpy as np
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow, viscosity_jump

mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40, cache=False)
s, c = mf.shell, mf.shell.constants
r, R = s.r, mf.R
rt, gt = r[-1], s.gravity[-1]
g0 = c.G * c.earth_mass / R**2
rho_L = 3300.0
w = np.gradient(r)
print(" l | code-sea vs true-sea | code-sea vs true-land | geoid: missing sea-surface mass rho_w U/g (Te=0, C=1)")
print("   |  L2 rel   max rel     |  L2 rel   max rel      |  L2 rel of K_N_fk   max rel")
for l in [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40]:
    tr = fk.solve_degree_traces(s, l)
    sig = tr[:, 1] * c.visc_ref / rt                       # Pa per kg/m^2
    U = tr[:, 2] * c.visc_ref / (c.rho_ref * rt)           # m^2/s^2 per kg/m^2 at r_t
    UR = U * (rt / R)**(l + 1)                             # everything is below r_t
    q_code = -(sig + c.rho_water * U)
    q_sea = -(sig + c.rho_water * UR) + rho_L * (UR - U)
    q_land = -sig + rho_L * (UR - U)
    sl = slice(1, -1)
    def rel(a, b):
        return (np.sqrt(np.sum(w[sl] * (a - b)[sl]**2) / np.sum(w[sl] * b[sl]**2)),
                np.max(np.abs(a - b)[sl]) / np.max(np.abs(b)[sl]))
    a1 = rel(q_code, q_sea); a2 = rel(q_code, q_land)
    # geoid: flowkernels' geoid at R (continued) vs K_N + rock mass sheet at r_t, mass -(sig + rho_w U)/g
    if l > 1:
        S = 4 * np.pi * c.G * rt * (rt / R)**(l + 1) / ((2 * l + 1) * g0)
        N_fk = mf.kernels["K_N_fk"][l - 1] * (gt / g0) * (rt / R)**(l + 1)
        idx = np.argsort(mf.kernels["r"], kind="stable")
        # kernels are on the duplicated node list; map back to shell nodes
        kr = mf.kernels["r"]
        Nfk = np.interp(r, kr, N_fk)
        KN = np.interp(r, kr, mf.kernels["K_N"][l - 1])
        N_plate = KN + S * (-(sig + c.rho_water * U) / gt)
        a3 = rel(N_plate, Nfk)
    else:
        a3 = (np.nan, np.nan)
    print(f"{l:2d} | {a1[0]:7.4f} {a1[1]:8.4f}     | {a2[0]:7.4f} {a2[1]:8.4f}      | {a3[0]:7.4f} {a3[1]:8.4f}")
