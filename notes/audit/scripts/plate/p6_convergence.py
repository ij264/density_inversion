"""Audit step 2: convergence of OceanLoadedPlate in quad_lmax and CG tolerance; symmetry of the truncated operator."""
import time
import numpy as np
from pygeoinf.symmetric_space import sphere
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.plate import OceanLoadedPlate
from density_inversion.forward import examples as ex

L = 50
mf = MantleFlow(lmax=L)
Q = mf.load_space()
deg, _ = ex.sphere_ordering(L)
rng = np.random.default_rng(11)
q = rng.standard_normal(Q.dim) * 1e6 / (1.0 + deg)  # red load spectrum, Pa
q[deg < 2] = 0.0

def solve(Te, fac, rtol=1e-10):
    p = OceanLoadedPlate.from_mantle(mf, Te=Te, quad_lmax=int(fac * L), rtol=rtol)
    op = p.operator(Q)
    t = time.time()
    w = Q.to_components(op(Q.from_components(q)))
    return w, time.time() - t

for Te in (40e3, 0.0):
    ref, tr = solve(Te, 8)
    print(f"Te={Te/1e3:.0f} km, reference quad_lmax=8L ({tr:.1f}s)")
    lat, lon, rg = ex.to_grid(ref, L)
    for fac in (1, 2, 3, 4, 6):
        w, t = solve(Te, fac)
        _, _, wg = ex.to_grid(w, L)
        err = np.linalg.norm(w - ref) / np.linalg.norm(ref)
        print(f"   quad_lmax={fac}L: rel L2 err vs 8L = {err:.2e}; max grid err / max|w| = {abs(wg-rg).max()/abs(rg).max():.2e} ({t:.1f}s)")

# CG tolerance vs achieved accuracy at 3L
ref, _ = solve(40e3, 3, rtol=1e-13)
for rt in (1e-4, 1e-6, 1e-8, 1e-10):
    w, t = solve(40e3, 3, rtol=rt)
    print(f"rtol={rt:g}: rel err vs rtol 1e-13 = {np.linalg.norm(w-ref)/np.linalg.norm(ref):.2e} ({t:.1f}s)")

# symmetry of the truncated operator in components (L2) and in the Sobolev metric, L=16
Ls = 16
mfs = MantleFlow(lmax=Ls)
Qs = mfs.load_space()
p = OceanLoadedPlate.from_mantle(mfs, Te=40e3)
M = p.assembled(Qs)
Mc = np.array([Qs.to_components(M(Qs.from_components(e))) for e in np.eye(Qs.dim)]).T
print("component-matrix asymmetry ||M-M^T||/||M|| =", np.linalg.norm(Mc - Mc.T) / np.linalg.norm(Mc))
# Sobolev Gram (diagonal)
G = np.array([Qs.inner_product(Qs.from_components(e), Qs.from_components(e)) for e in np.eye(Qs.dim)])
S = G[:, None] * Mc
print("Sobolev-metric asymmetry ||G M - (G M)^T||/||G M|| =", np.linalg.norm(S - S.T) / np.linalg.norm(S))
ev = np.linalg.eigvalsh(0.5 * (Mc + Mc.T))
print("eigenvalues of component matrix: min", ev.min(), "max", ev.max())
