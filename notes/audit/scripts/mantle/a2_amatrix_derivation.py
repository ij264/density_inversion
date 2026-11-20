"""Task 1: check flowkernels' A matrix rows from first principles (sympy).
(i) flow rows, compressible (anelastic continuity div(rho0 u) = 0), with and without the -2/3 eta div(u)
    term in the constitutive law;
(ii) the self-gravity coupling: a static state u = 0 with sigma = -rho0 U (U = -Phi, geodesy sign) and uniform
    rho0 must satisfy the system (it is the exact hydrostatic response of a uniform fluid to a potential U).
"""
import numpy as np
import sympy as sy
import flowkernels as fk

r, th = sy.symbols("r theta", positive=True)
l = 3
ll = l * (l + 1)
eta = sy.Symbol("eta", positive=True)
Uf, Vf, Sf, Tf, rho = (sy.Function(n)(r) for n in ("U", "V", "S", "T", "rho"))
kap = r * sy.diff(rho, r) / rho
P = sy.legendre(l, sy.cos(th))
dP = sy.diff(P, th)

# A-matrix relations (eta0 = 1): S = sigma_rr coeff * r, T = sigma_rtheta coeff * r
dU = ((-2 - kap) * Uf + ll * Vf) / r
dV = (-Uf + Vf + Tf / eta) / r
dS_A = (4 * (3 + kap) * eta * Uf - 6 * ll * eta * Vf + Sf + ll * Tf) / r
dT_A = (-2 * (3 + kap) * eta * Uf + 2 * (2 * ll - 1) * eta * Vf - Sf - 2 * Tf) / r

for bulk in (True, False):
    ur, ut = Uf * P, Vf * dP
    div = sy.diff(r**2 * ur, r) / r**2 + sy.diff(sy.sin(th) * ut, th) / (r * sy.sin(th))
    err = sy.diff(ur, r)
    ett = sy.diff(ut, th) / r + ur / r
    epp = ur / r + ut * sy.cos(th) / (r * sy.sin(th))
    ert = sy.Rational(1, 2) * (r * sy.diff(ut / r, r) + sy.diff(ur, th) / r)
    iso = sy.Rational(1, 3) * div if bulk else 0
    # pressure from sigma_rr = -p + 2 eta (e_rr - iso) and sigma_rr = (S/r) P
    p = -(Sf / r) * P + 2 * eta * (err - iso)
    srr = -p + 2 * eta * (err - iso)
    stt = -p + 2 * eta * (ett - iso)
    spp = -p + 2 * eta * (epp - iso)
    srt = 2 * eta * ert
    fr = sy.diff(srr, r) + sy.diff(srt, th) / r + (2 * srr - stt - spp + srt * sy.cos(th) / sy.sin(th)) / r
    ft = sy.diff(srt, th) / r + sy.diff(srt, r) * 0  # placeholder replaced below
    ft = sy.diff(srt, r) + sy.diff(stt, th) / r + ((stt - spp) * sy.cos(th) / sy.sin(th) + 3 * srt) / r
    subs = {sy.Derivative(Uf, (r, 2)): sy.diff(dU, r), }
    # replace derivatives using the A relations
    def red(e):
        e = e.subs(sy.Derivative(Uf, (r, 2)), sy.diff(dU, r))
        e = e.subs(sy.Derivative(Vf, (r, 2)), sy.diff(dV, r))
        e = e.subs({sy.Derivative(Uf, r): dU, sy.Derivative(Vf, r): dV,
                    sy.Derivative(Sf, r): dS_A, sy.Derivative(Tf, r): dT_A})
        e = e.subs({sy.Derivative(Uf, r): dU, sy.Derivative(Vf, r): dV,
                    sy.Derivative(Sf, r): dS_A, sy.Derivative(Tf, r): dT_A})
        return e
    # check: srt from kinematics equals (T/r) dP
    chk_T = sy.simplify(red(srt - (Tf / r) * dP))
    res_r = red(fr)
    res_t = red(ft)
    # evaluate numerically at random values
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(3):
        sub = {Uf: rng.normal(), Vf: rng.normal(), Sf: rng.normal(), Tf: rng.normal(),
               eta: abs(rng.normal()) + .5, r: 1.3, th: 0.7}
        rr = {rho: 2.0, sy.Derivative(rho, r): rng.normal(), sy.Derivative(rho, (r, 2)): rng.normal()}
        f = lambda e: complex(sy.N(e.subs(sy.Derivative(rho, (r, 2)), rr[sy.Derivative(rho, (r, 2))])
                                    .subs(sy.Derivative(rho, r), rr[sy.Derivative(rho, r)])
                                    .subs(rho, rr[rho]).subs(sub)))
        vals.append((abs(f(chk_T)), abs(f(res_r)), abs(f(res_t))))
    print(f"bulk term -2/3 eta div u {'included' if bulk else 'omitted '}: "
          f"max |sigma_rtheta mismatch| {max(v[0] for v in vals):.2e}, "
          f"max |r-momentum residual| {max(v[1] for v in vals):.2e}, max |theta residual| {max(v[2] for v in vals):.2e}")

# (ii) hydrostatic check with the actual flowkernels A
A = fk._amatrix(5, np.array([1.3]), np.array([4500.0]), np.array([0.0]), 4000.0)[0]
PHI, DPHI = 0.7, -1.9
y = np.array([0, 0, -(4500 / 4000) * PHI, 0, PHI, DPHI])
dy = A @ y
# expected: d/dlnr of (-rho/rho_ref PHI) = -rho/rho_ref (PHI + DPHI) for uniform rho; others 0 except potential rows
print("hydrostatic test: dSIG_RR", dy[2], "expected", -(4500 / 4000) * (PHI + DPHI), " dSIG_RTHETA", dy[3],
      "(expected 0), du", dy[:2])
