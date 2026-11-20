"""Item 8a: bound the effect of the top boundary's self-gravity on the flow (load q and the net geoid).

flowkernels' potential-matching rows contain c_top * sigma_rr(top): the potential of the deflected top's mass
(-sigma_rr/g). Scale those entries by `s` (1 = flowkernels, 0 = top mass exerts no gravity on the flow) and
compare the load kernel K_q = -(eta/r)(sigma + rho_w/rho_ref PHI) and the mass-only part sigma.
Also compare with the back-of-envelope 3 drho / ((2l+1) rho_bar).
"""
import numpy as np
import scipy.sparse as sp
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow, truncated

mf = MantleFlow(lmax=12)
shell = mf.shell
orig = fk._assemble


def make_assemble(s):
    def asm(model, l, P):
        M, b0 = orig(model, l, P)
        M = M.tolil()
        n = model.n
        top = 6 * (n - 1)
        # entries (b0+4, top+SIG_RR) and (b0+5, top+SIG_RR) carry c_top
        M[b0 + 4, top + fk.SIG_RR] = M[b0 + 4, top + fk.SIG_RR] * s
        M[b0 + 5, top + fk.SIG_RR] = M[b0 + 5, top + fk.SIG_RR] * s
        return M.tocsc(), b0
    return asm


c = shell.constants
r = shell.r
wts = np.gradient(r)
rho_bar = c.earth_mass / (4 / 3 * np.pi * mf.R ** 3)
res = {}
for s in (1.0, 0.0):
    fk._assemble = make_assemble(s)
    out = {}
    for l in (2, 3, 4, 6, 8, 12):
        u, sig, phi, sb, pb = fk.solve_degree_traces(shell, l).T
        q = -c.visc_ref / r[-1] * (sig + c.rho_water / c.rho_ref * phi)
        qs = -c.visc_ref / r[-1] * sig
        out[l] = (q, qs, phi)
    res[s] = out
fk._assemble = orig

print("l   |dK_q|/|K_q| (rms over depth)  |dsigma|/|sigma|   BoE 3(rho_L-rho_w)/((2l+1)rho_bar)  BoE air")
for l in (2, 3, 4, 6, 8, 12):
    a, b = res[1.0][l], res[0.0][l]
    rel = lambda x, y: np.sqrt(np.sum(wts * (x - y) ** 2) / np.sum(wts * y ** 2))
    boe_w = 3 * (shell.rho[-1] - c.rho_water) / ((2 * l + 1) * rho_bar)
    boe_a = 3 * shell.rho[-1] / ((2 * l + 1) * rho_bar)
    print(f"{l:2d}  {rel(b[0], a[0]):.3f}                          {rel(b[1], a[1]):.3f}            {boe_w:.3f}                                {boe_a:.3f}")
# depth profile at l=2 of ratio
l = 2
a, b = res[1.0][l], res[0.0][l]
for d in (300e3, 700e3, 1500e3, 2500e3):
    i = np.argmin(np.abs(r - (mf.R - d)))
    print(f"l=2 source {d/1e3:.0f} km: K_q(no top grav)/K_q(fk) = {b[0][i]/a[0][i]:.3f}")
