"""Task 3: independent closed-form benchmark. Incompressible, uniform viscosity, NON-self-gravitating sphere,
a density sheet (unit weight g*sigma = 1 Pa) at r', free slip at r0 and rt. Fundamental power-law solutions
derived here with sympy from the Stokes equations in spherical coordinates (axisymmetric, P_l(cos th)),
solved in 50-digit mpmath. Compared with flowkernels run with compressible=False, uniform viscosity and its
self-gravity coupling switched off (A[SIG_RR, DPHI] = A[SIG_RTHETA, PHI] = 0, by monkeypatching _amatrix).
Also checks flowkernels' potential bookkeeping (geoid of anomaly + deflected top + deflected CMB with the
fluid core's response) against a direct calculation from the analytic stresses."""
import sys
import numpy as np
import sympy as sy
import mpmath as mp
import flowkernels as fk
from density_inversion.forward.mantle import reference_model

mp.mp.dps = 50
r, th, n = sy.symbols("r theta n", positive=True)
a, b, cc = sy.symbols("a b c")


def fundamental(l):
    """Four solutions (n, a, b, c): u_r = a r^n P, u_th = b r^n dP/dth, p = c r^(n-1) P, eta = 1."""
    x = sy.cos(th)
    P = sy.legendre(l, x)
    dP = sy.diff(P, th)
    sols = []
    for nn in [l + 1, l - 1, -l, -l - 2]:
        ur = a * r**nn * P
        ut = b * r**nn * dP
        p = cc * r**(nn - 1) * P
        lap = lambda f: sy.diff(r**2 * sy.diff(f, r), r) / r**2 + sy.diff(sy.sin(th) * sy.diff(f, th), th) / (r**2 * sy.sin(th))
        div = sy.diff(r**2 * ur, r) / r**2 + sy.diff(sy.sin(th) * ut, th) / (r * sy.sin(th))
        Lr = lap(ur) - 2 * ur / r**2 - 2 / (r**2 * sy.sin(th)) * sy.diff(ut * sy.sin(th), th)
        Lt = lap(ut) + 2 / r**2 * sy.diff(ur, th) - ut / (r**2 * sy.sin(th)**2)
        er = -sy.diff(p, r) + Lr
        et = -sy.diff(p, th) / r + Lt
        # evaluate at a few (r, th) points to get linear equations in a, b, c
        eqs = []
        for (rv, tv) in [(sy.Rational(13,10), sy.Rational(2,5)), (sy.Rational(7,10), sy.Rational(11,10)), (sy.Rational(19,10), sy.Rational(11,5)), (sy.Rational(11,10), sy.Rational(9,10))]:
            for e in (div, er, et):
                eqs.append(sy.expand(e.subs({r: rv, th: tv})))
        A = sy.Matrix([[sy.N(sy.diff(e, v), 60) for v in (a, b, cc)] for e in eqs])
        ns = A.nullspace(simplify=False) if False else None
        # numerical null space with mpmath
        M = mp.matrix([[mp.mpf(str(A[i, j])) for j in range(3)] for i in range(A.rows)])
        U, S, V = mp.svd_r(M)
        vec = [V[2, j] for j in range(3)]
        assert S[2] < mp.mpf(10)**-25 * S[0], (l, nn, S)
        k = vec[0]
        sols.append((nn, mp.mpf(1), vec[1] / k, vec[2] / k))
    return sols


def state(sols, rv, l):
    """4x4 matrix: rows (U, V, Srr, Srt), columns the fundamental solutions, at radius rv (eta = 1).
    Coefficients of P (U, Srr) and of dP/dth (V, Srt)."""
    rv = mp.mpf(rv)
    F = mp.matrix(4, 4)
    for j, (nn, aa, bb, ccc) in enumerate(sols):
        U = aa * rv**nn
        V = bb * rv**nn
        p = ccc * rv**(nn - 1)
        dU = aa * nn * rv**(nn - 1)
        dV = bb * nn * rv**(nn - 1)
        Srr = -p + 2 * dU
        Srt = dV - V / rv + U / rv          # (1/r) d/dth (U P) = U/r dP/dth
        F[0, j], F[1, j], F[2, j], F[3, j] = U, V, Srr, Srt
    return F


def analytic(l, r0, rt, rs):
    sols = fundamental(l)
    F0, Ft, Fs = state(sols, r0, l), state(sols, rt, l), state(sols, rs, l)
    M = mp.matrix(8, 8)
    rhs = mp.matrix(8, 1)
    for j in range(4):
        M[0, j] = F0[0, j]; M[1, j] = F0[3, j]          # below: alpha
        M[2, 4 + j] = Ft[0, j]; M[3, 4 + j] = Ft[3, j]  # above: beta
        for k in range(4):                               # continuity at rs, jump in Srr
            M[4 + k, j] = -Fs[k, j]
            M[4 + k, 4 + j] = Fs[k, j]
    rhs[6] = 1  # sigma_rr(above) - sigma_rr(below) = +g sigma = 1
    x = mp.lu_solve(M, rhs)
    st = sum(Ft[2, j] * x[4 + j] for j in range(4))
    sb = sum(F0[2, j] * x[j] for j in range(4))
    return float(st), float(sb)


def fk_nonselfgrav(model, l):
    orig = fk._amatrix
    def patched(*args, **kw):
        A = orig(*args, **kw)
        A[:, fk.SIG_RR, fk.DPHI] = 0.0
        A[:, fk.SIG_RTHETA, fk.PHI] = 0.0
        return A
    fk._amatrix = patched
    try:
        return fk.solve_degree_traces(model, l)
    finally:
        fk._amatrix = orig


def geoid_direct(l, model, rs_idx, st, sb, c):
    """U at the top from: unit sheet at rs, top mass -st/gt, CMB mass (sb + rho_c U_b)/g_b, fluid core."""
    rr, g = model.r, model.gravity
    r0, rt, rs = rr[0], rr[-1], rr[rs_idx]
    f = 4 * np.pi * c.G / (2 * l + 1)
    mt = -st / g[-1]
    # U_b = f [ rs (r0/rs)^l *1 + rt (r0/rt)^l mt + r0 (sb + rho_c U_b)/g_b ]
    Ub = f * (rs * (r0 / rs)**l + rt * (r0 / rt)**l * mt + r0 * sb / g[0]) / (1 - f * r0 * c.rho_core / g[0])
    mb = (sb + c.rho_core * Ub) / g[0]
    Ut = f * (rs * (rs / rt)**(l + 1) + rt * mt + r0 * (r0 / rt)**(l + 1) * mb)
    return Ut


if __name__ == "__main__":
    base = reference_model()
    for label, rr in [("257 nodes", base.r),
                      ("1025 nodes", np.linspace(base.r[0], base.r[-1], 1025))]:
        rho = np.interp(rr, base.r, base.rho)
        model = fk.ShellModel(r=rr, rho=rho, eta=np.ones_like(rr), compressible=False)
        c = model.constants
        r0, rt = rr[0], rr[-1]
        print(f"\n=== {label}: incompressible, uniform eta, no self-gravity in the flow")
        print(" l  depth_km  sig_top(analytic) sig_top(fk)  rel     sig_bot(an)  sig_bot(fk)  rel     geoid rel")
        for l in [1, 2, 5, 10, 20]:
            tr = fk_nonselfgrav(model, l)
            for d in [200e3, 670e3, 1500e3, 2500e3]:
                s = int(np.argmin(abs(rr - (rt - d))))
                st_a, sb_a = analytic(l, r0 / rt, 1.0, rr[s] / rt)
                g_s = model.gravity[s]
                st_f = tr[s, 1] * c.visc_ref / rt / g_s     # Pa per (g sigma) = per Pa
                sb_f = tr[s, 3] * c.visc_ref / r0 / g_s
                # potential: flowkernels PHI_top -> U = PHI visc_ref/(rho_ref rt); per unit sheet mass
                if l > 1:
                    U_f = tr[s, 2] * c.visc_ref / (c.rho_ref * rt)
                    U_a = geoid_direct(l, model, s, st_a * g_s, sb_a * g_s, c)
                    grel = abs(U_f - U_a) / abs(U_a)
                else:
                    grel = float("nan")
                print(f"{l:2d} {d/1e3:8.0f}  {st_a: .6e}  {st_f: .6e}  {abs(st_f-st_a)/abs(st_a):.1e}"
                      f"  {sb_a: .6e} {sb_f: .6e} {abs(sb_f-sb_a)/abs(sb_a):.1e}  {grel:.1e}")
