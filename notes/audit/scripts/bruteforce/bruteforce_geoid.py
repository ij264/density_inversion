"""Independent brute-force geoid of an isostatically compensated Gaussian Moho uplift.

Plain numpy/scipy. Newton kernel integrated in space (azimuth done analytically
via complete elliptic integral K, colatitude by adaptive quadrature).
All quantities first order in A.
"""
import sys
import numpy as np
from scipy.special import ellipk, eval_legendre
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

R = 6370000.0
G = 6.67e-11
g0 = 9.820015821809946
rho_w, rho_c, rho_LAB = 1030.0, 2938.42, 3370.361
d_s, d_m = 4960.615, 14461.275
d_b = 4960.615 + 105000.0
kn_d = np.array([14461.275, 20000, 36000, 56000, 80000, 110000.0])
kn_r = np.array([3352.344, 3346.235, 3355.871, 3347.637, 3312.168, 3335.498])
A = 2000.0
s = 400e3 / R


def rhoL(d):
    return np.interp(d, kn_d, kn_r)  # d_b < 110000 so no extrapolation needed


f = lambda th: np.exp(-th**2 / (2 * s**2))  # angular shape (w and dr share it)

# ---------------- isostasy ----------------
drho_M = rhoL(d_m) - rho_c                    # mass/area added per metre of Moho uplift
w0 = -drho_M * A / (rho_LAB - rho_w)          # planar (per-unit-area) balance

# lift sheets (per unit w): sigma = (rho_below - rho_above) * w at each jump
jump_s = rho_c - rho_w
jump_m = rhoL(d_m) - rho_c
jump_b = rho_LAB - rhoL(d_b)
# lid gradient: delta rho(d) = w * d rho_L/dd, piecewise constant gradient
segs = [d_m] + [x for x in kn_d if d_m < x < d_b] + [d_b]


def kernel(rp, th, eps=1e-12):
    """K(rp,th) = int f(th') sin th' dth' int dphi / D, D distance from (R,th) to (rp,th',phi)."""
    def integrand(tp):
        a = R**2 + rp**2 - 2 * R * rp * np.cos(th) * np.cos(tp)
        b = 2 * R * rp * np.sin(th) * np.sin(tp)
        m = 2 * b / (a + b)
        return f(tp) * np.sin(tp) * 4 * ellipk(m) / np.sqrt(a + b)
    # source support: f < 1e-40 beyond 14 s (s=0.0628 rad), so integrate th' in [0, 14 s]
    upper = min(np.pi, 14 * s)
    pts = [p for p in (th - 0.01, th, th + 0.01) if 0 < p < upper]
    lo = 0.0
    val, err = quad(integrand, lo, upper, points=pts or None, limit=2000,
                    epsabs=0, epsrel=eps)
    return val, err


def geoid_parts(th, nz=12):
    """Geoid N (m) at colatitude th from each mass group, per the spec."""
    fac = G / g0  # N = -Phi/g0 = G/g0 * sum sigma r^2 K
    out = {}
    r_s, r_m, r_b = R - d_s, R - d_m, R - d_b
    Km, em = kernel(r_m, th)
    Ks, es = kernel(r_s, th)
    Kb, eb = kernel(r_b, th)
    out['moho_pert'] = fac * drho_M * A * r_m**2 * Km
    out['lift_seafloor'] = fac * jump_s * w0 * r_s**2 * Ks
    out['lift_moho'] = fac * jump_m * w0 * r_m**2 * Km
    out['lift_base'] = fac * jump_b * w0 * r_b**2 * Kb
    xg, wg = leggauss(nz)
    lid = 0.0
    for d1, d2 in zip(segs[:-1], segs[1:]):
        grad = (rhoL(d2) - rhoL(d1)) / (d2 - d1)
        dd = 0.5 * (d2 - d1) * xg + 0.5 * (d1 + d2)
        for di, wi in zip(dd, 0.5 * (d2 - d1) * wg):
            rp = R - di
            lid += wi * grad * w0 * rp**2 * kernel(rp, th)[0]
    out['lift_lid'] = fac * lid
    out['lift_total'] = out['lift_seafloor'] + out['lift_moho'] + out['lift_base'] + out['lift_lid']
    out['total'] = out['moho_pert'] + out['lift_total']
    return out


def legendre_coeffs(vals, x, wx, L):
    return np.array([(2 * l + 1) / 2 * np.sum(wx * vals * eval_legendre(l, x)) for l in range(L + 1)])


if __name__ == '__main__':
    print(f'rhoL(Moho)={rhoL(d_m):.4f} rhoL(base)={rhoL(d_b):.4f}')
    print(f'drho_M={drho_M:.4f}  w0={w0:.6f} m')
    # mass check: lift total mass/area per w = rho_LAB - rho_w
    lidmass = rhoL(d_b) - rhoL(d_m)
    print('mass check (should be 0):', drho_M * A + w0 * (jump_s + jump_m + lidmass + jump_b))

    # ---- centre value, convergence in nz and quad tolerance ----
    for nz in (4, 8, 12, 20):
        p = geoid_parts(0.0, nz)
        print(f'nz={nz:2d} centre:', {k: round(v, 9) for k, v in p.items()})
    # tolerance check
    print('K tol check (Moho, th=0, th=0.05):',
          [kernel(R - d_m, t, 1e-8)[0] / kernel(R - d_m, t, 1e-13)[0] - 1 for t in (0.0, 0.05)])
    # 2nd-order sensitivity: Moho sheet placed at d_m - A/2 (centroid of replaced crust)
    Km2 = kernel(R - d_m + A / 2, 0.0)[0]
    Km1 = kernel(R - d_m, 0.0)[0]
    print('moho_pert sheet at centroid / at Moho - 1 =',
          (R - d_m + A / 2)**2 * Km2 / ((R - d_m)**2 * Km1) - 1)

    # ---- Legendre decomposition by Gauss-Legendre quadrature in cos(theta) ----
    L = 50
    results = {}
    for n in (int(a) for a in (sys.argv[1:] or ['160', '240'])):
        x, wx = leggauss(n)
        th = np.arccos(x)
        parts = [geoid_parts(t, 8) for t in th]
        keys = parts[0].keys()
        res = {}
        for k in keys:
            v = np.array([p[k] for p in parts])
            c = legendre_coeffs(v, x, wx, 200)
            res[k] = (c[2:L + 1].sum(), c.sum(), c[0], c[1])
        cw = legendre_coeffs(w0 * f(th), x, wx, 200)
        res['w'] = (cw[1:L + 1].sum(), cw[:L + 1].sum(), cw[0], cw.sum())
        results[n] = res
        print(f'\n n={n}')
        for k, v in res.items():
            print(f'  {k:14s} ' + '  '.join(f'{z: .9f}' for z in v))
        # spectral cross-check for total (exterior-potential formula per degree)
        np.save(f'/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/audit/bruteforce/N_total_n{n}.npy',
                np.array([[t, p['total']] for t, p in zip(th, parts)]))
    # independent spectral cross-check: N_l = 4 pi G r' (r'/R)^(l+1) sigma_l /((2l+1) g0)
    x, wx = leggauss(400)
    fl = legendre_coeffs(f(np.arccos(x)), x, wx, 400)
    l = np.arange(401)

    def spec(rp, sig):
        return 4 * np.pi * G * rp * (rp / R)**(l + 1) * sig * fl / ((2 * l + 1) * g0)
    Nl = spec(R - d_m, drho_M * A) + spec(R - d_s, jump_s * w0) + spec(R - d_m, jump_m * w0) + spec(R - d_b, jump_b * w0)
    xg, wg = leggauss(20)
    for d1, d2 in zip(segs[:-1], segs[1:]):
        grad = (rhoL(d2) - rhoL(d1)) / (d2 - d1)
        for di, wi in zip(0.5 * (d2 - d1) * xg + 0.5 * (d1 + d2), 0.5 * (d2 - d1) * wg):
            Nl = Nl + spec(R - di, wi * grad * w0)
    print('\nspectral check total: all l', Nl.sum(), ' l=2..50', Nl[2:51].sum(), ' N0', Nl[0], ' N1', Nl[1])
    Nm = spec(R - d_m, drho_M * A)
    print('spectral check moho_pert: all', Nm.sum(), ' 2..50', Nm[2:51].sum())
    print('w spectral: l>=1..50', w0 * fl[1:51].sum(), ' 0..50', w0 * fl[:51].sum(), ' all', w0 * fl.sum())
