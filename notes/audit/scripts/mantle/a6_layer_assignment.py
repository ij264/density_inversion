"""Task 4c follow-up: why does a field constant in one layer miss the direct quadrature by ~1e-3,
independent of radial_modes? Check _points_by_layer assignment and the fit of a constant."""
import sys
import numpy as np
from density_inversion.forward.mantle import MantleFlow, viscosity_jump
from density_inversion import operators as po
sys.path.insert(0, "tests")
from test_mantle import profile_components

mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40)
k = mf.kernels
R = float(k["R"])
X = mf.density_space()
A = mf.operator(X)
parts = po._points_by_layer(X, k["r"], R)
print("interfaces", X.interfaces, "layer bounds", X.layer_bounds)
for i, (idx, x) in enumerate(parts):
    print(f"layer {i}: {idx.size} points, first {idx[:2]}, last {idx[-2:]}, x range {x.min():.9f}-{x.max():.9f}")
print("total points", sum(p[0].size for p in parts), "of", k["r"].size)

for l in (2, 20):
    for i, ball in enumerate(X.layers):
        a, b = X.layer_bounds[i]
        x = np.linspace(a, b, 400)
        Phi = ball.basis[l].evaluate(x)
        coef, *_ = np.linalg.lstsq(Phi, np.ones_like(x), rcond=None)
        fit = Phi @ coef
        idx, xx = parts[i]
        fit_nodes = ball.basis[l].evaluate(xx) @ coef
        direct = np.sum(k["weights"][idx] * k["K_q"][l - 1][idx])
        viaop = np.sum(k["weights"][idx] * k["K_q"][l - 1][idx] * fit_nodes)
        print(f"l={l} layer {i}: fit of 1 on 400 pts: max err {abs(fit - 1).max():.2e}, at nodes {abs(fit_nodes - 1).max():.2e}"
              f"  (ends {fit_nodes[0]:.5f},{fit_nodes[-1]:.5f});  direct {direct:.6e}  via fit {viaop:.6e}  rel {viaop / direct - 1:.2e}")
