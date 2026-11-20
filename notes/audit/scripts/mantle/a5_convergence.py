"""Task 4: numerical convergence.
(a) radial grid: kernels + trapezoid on 257 nodes vs the same piecewise-linear PREM on 2x and 4x refined nodes
    (original nodes kept, so the viscosity jump stays on the same radius);
(b) radial_modes (16,10) vs (32,20): response to test fields, per-degree singular values;
(c) a density step exactly at the viscosity jump.
"""
import sys
import numpy as np
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow, viscosity_jump, reference_model

LMAX = 40
base = reference_model()
R = base.r[-1]


def refine(r, k):
    out = [r[:1]]
    for a, b in zip(r[:-1], r[1:]):
        out.append(np.linspace(a, b, k + 1)[1:])
    return np.concatenate(out)


mf0 = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=LMAX, cache=False)
r_jump = float(mf0.kernels["r_jump"])
print("jump node depth km", (R - r_jump) / 1e3, " LAB node r_top km depth", (R - mf0.kernels["r_lab"]) / 1e3)
d_jump = R - r_jump

models = {"257": mf0}
for k in (2, 4):
    rr = refine(base.r, k)
    m = fk.ShellModel(r=rr, rho=np.interp(rr, base.r, base.rho), eta=np.ones_like(rr))
    models[f"x{k}"] = MantleFlow(viscosity=viscosity_jump(30.0, d_jump), lab_depth=105e3, lmax=LMAX,
                                 model=m, cache=False)
    assert abs(models[f"x{k}"].kernels["r_jump"] - r_jump) < 1e-3

def gauss(z0, wkm):
    return lambda r: np.exp(-0.5 * ((r - (R - z0)) / wkm) ** 2)

tests = {
    "gauss 300km": gauss(300e3, 100e3),
    "gauss 1000km": gauss(1000e3, 150e3),
    "gauss 2000km": gauss(2000e3, 200e3),
    "step: 1 below jump": lambda r: (r < r_jump + 1).astype(float),
    "uniform upper": lambda r: (r >= r_jump).astype(float),
}


def q(mf, f, l, name="K_q"):
    k = mf.kernels
    rr = k["r"].copy()
    vals = f(rr)
    if "step" in name or True:
        # respect the duplicated jump node: lower copy belongs below, upper copy above
        n_low = np.searchsorted(rr, r_jump, side="right")  # index after the lower copy
    return np.sum(k["weights"] * k[name][l - 1] * vals)

# for the step functions, evaluate on duplicated nodes properly: lower copy = r_jump exactly -> "below"
tests["step: 1 below jump"] = lambda r: (r <= r_jump).astype(float)
tests["uniform upper"] = lambda r: (r > r_jump).astype(float)

print("\n(a) relative change of q_lm (and N_lm) on refining the radial grid, vs x4")
for name, f in tests.items():
    line = f"{name:22s}"
    for l in (2, 8, 20, 40):
        ref = q(models["x4"], f, l)
        line += f" l={l}: {abs(q(models['257'], f, l) / ref - 1):.1e}/{abs(q(models['x2'], f, l) / ref - 1):.1e}"
    print(line)
print("geoid K_N:")
for name, f in tests.items():
    line = f"{name:22s}"
    for l in (2, 8, 20, 40):
        ref = q(models["x4"], f, l, "K_N")
        line += f" l={l}: {abs(q(models['257'], f, l, 'K_N') / ref - 1):.1e}/{abs(q(models['x2'], f, l, 'K_N') / ref - 1):.1e}"
    print(line)

# (c) the step at the jump: check the duplicated-node bookkeeping against a sub-sampled exact integral of the
# interpolated kernel (kernel is continuous; density steps at r_jump)
k = mf0.kernels
rr, w = k["r"], k["weights"]
i_low = np.flatnonzero(rr == r_jump)
print("\n(c) jump nodes in kernel list:", i_low, rr[i_low[0]:i_low[0] + 2] - r_jump, "weights", w[i_low[0]:i_low[0] + 2])
print("    sum weights below/above:", w[rr <= r_jump].sum() - (r_jump - k["r_cmb"]), w[rr > r_jump].sum() - (k["r_lab"] - r_jump))

# (b) radial modes
from density_inversion import operators as po
sys.path.insert(0, "tests")
from test_mantle import profile_components

print("\n(b) radial_modes")
spaces = {}
for rm in [(16, 10), (32, 20), (64, 40)]:
    X = mf0.density_space(radial_modes=rm)
    A = mf0.operator(X)
    spaces[rm] = (X, A)
    print(rm, "dim", X.dim)


def response(X, A, f, l):
    c = profile_components(X, l, 0, lambda x: f(x * R))
    out = A.codomain.to_components(A(X.from_components(c)))
    return out[l * l]


def projection_error(X, f, l):
    # how well the space represents the test profile (on 400 pts per layer)
    err = []
    for i, ball in enumerate(X.layers):
        a, b = X.layer_bounds[i]
        x = np.linspace(a, b, 400)
        Phi = ball.basis[l].evaluate(x)
        coef, *_ = np.linalg.lstsq(Phi, f(x * R), rcond=None)
        err.append(np.linalg.norm(Phi @ coef - f(x * R)) / max(np.linalg.norm(f(x * R)), 1e-300))
    return err

for name, f in tests.items():
    line = f"{name:22s}"
    for l in (2, 8, 20, 40):
        exact = q(mf0, f, l)
        line += f" l={l}:" + "/".join(f"{abs(response(*spaces[rm], f, l) / exact - 1):.1e}" for rm in spaces)
    print(line)
print("projection error of test fields (lower, upper) at l=20 for (16,10):",
      {n: np.round(projection_error(spaces[(16, 10)][0], f, 20), 4).tolist() for n, f in tests.items()})


def sing(X, A, l):
    Q = A.codomain
    e = np.zeros(Q.dim); e[l * l] = 1.0
    ev = Q.from_components(e)
    y = Q.to_components(A(A.adjoint(ev)))
    return np.sqrt(y[l * l] / e[l * l])

print("\nper-degree singular value of drho -> q, relative to (64,40):")
for l in (1, 2, 5, 10, 20, 30, 40):
    s = {rm: sing(*spaces[rm], l) for rm in spaces}
    print(f"l={l:2d} sv(64,40)={s[(64, 40)]:.4e}  rel (16,10): {s[(16, 10)] / s[(64, 40)] - 1:+.2e}  "
          f"(32,20): {s[(32, 20)] / s[(64, 40)] - 1:+.2e}")
