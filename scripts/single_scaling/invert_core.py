"""Single 1D scaling with the core forward model: Holdt oceanic residual depths, the geoid, or both.

Written by Claude (AI assistant), 19 Nov 2026.

Topography block (Holdt 2022 oceanic points): d_h = G_h R + b0 + c(x) + e,  e ~ N(0, diag s^2), c a correlated
  field (degrees 1..40, pointwise sd sigma_c, length ell_c), as in invert.py.
Geoid block (products/geoid_for_inversion.npz, degrees 2-25, C21/S21 removed): d_N = G_N R + e_N,
  e_N ~ N(0, C_N + tau^2 diag(P_l)): C_N the stated errors (diagonal, see geoid_data); tau^2 P_l a modelling error,
  a fraction tau^2 of the observed power at each degree, for geoid structure a 1D scaling cannot produce.
R ~ N(0, sigma_R^2 SE(ell_R) + SIGMA_MU^2 11^T) on 28 nodes (a common offset marginalised).
Holdt points are averaged in 1-degree cells (see thin). Hyperparameters by maximum evidence within physical bounds (Joint.BOUNDS); null model R = 0.

Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/invert_core.py core_<...>.npz {topo,geoid,joint}
           [--datum {lab,cut105}] [--starts N]
  --datum lab    (default) the product's datum: WINTERC-G's lithosphere removed down to its LAB; variance LAB + crust
                 + satellite + hydrostatic.
  --datum cut105 the lithosphere removed down to 105 km, the top of the core model's flow domain, so that no material
                 is both removed from the data and represented by the mantle unknown (added 23 Nov 2026). The LAB
                 variance does not apply; the crustal variance is the product's (computed for the cut at the LAB).
  --starts N     N Nelder-Mead starts for the hyperparameters (the first at the default point, the rest perturbed by
                 one unit in each log-parameter); the best is kept and the spread of logZ is reported.
Outputs are suffixed with the datum when it is not 'lab'.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from invert import B_EARTH, DATA, LC, N_STEPHENSON, OUT, SIGMA_B0, harmonics_at  # noqa: E402

SIGMA_MU = 1.0
PRODUCT = Path(__file__).resolve().parents[3] / "products" / "geoid_for_inversion.npz"


def thin(lat, lon, d, s, rows):
    """Average points and rows (operator rows, harmonics) within 1-degree cells. The error of a cell is the rms of
    its points' errors, not reduced by sqrt(n): residuals of nearby points are correlated (0.6 within 25 km, ~0
    beyond 100 km), so averaging does not reduce the error by sqrt(n)."""
    key = np.floor(lat).astype(int) * 1000 + np.floor(np.mod(lon, 360.0)).astype(int)
    u, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    avg = lambda x: (np.bincount(inv, weights=x) / cnt) if x.ndim == 1 else \
        np.stack([np.bincount(inv, weights=x[:, j]) for j in range(x.shape[1])], axis=1) / cnt[:, None]
    return avg(lat), avg(lon), avg(d), np.sqrt(avg(s ** 2)), [avg(r) for r in rows]


CUT105 = Path(__file__).resolve().parents[3] / "ch2_density" / "geoid" / "cache" / "N_lith_cut105.npy"


def geoid_data(datum="lab"):
    """Geoid datum and covariance, degrees 2-25 without C21/S21.

    C_N is diagonal: the product's var_total (LAB + crust + satellite, floored at 0.2 x the within-degree median),
    with the hydrostatic term replaced by Chambat et al. (2010)'s uncertainty (1 m on C20, 0.2 m on C40). The
    low-rank LAB and crust covariances are not used: each is the mean of only 2-4 difference vectors, so they
    assert exact correlations and leave combinations of coefficients with satellite variance alone (~1e-13 m^2).
    """
    g = np.load(PRODUCT)
    l, m = g["l"], g["m"]
    keep = ~((l == 2) & (np.abs(m) == 1))
    hyd = np.zeros(l.size)
    hyd[(l == 2) & (m == 0)] = 1.0 ** 2
    hyd[(l == 4) & (m == 0)] = 0.2 ** 2
    var_lab = g["var_lab"] if datum == "lab" else 0.0 * g["var_lab"]
    var_sum = var_lab + g["var_crust"] + g["var_satellite"] + hyd
    var = var_sum.copy()
    for L in np.unique(l):
        s = l == L
        var[s] = np.maximum(var_sum[s], 0.20 * np.median(var_sum[s]))
    d = g["d"] if datum == "lab" else g["N_obs"] - np.load(CUT105)
    dk, lk = d[keep], l[keep]
    power = np.array([np.mean(dk[lk == L] ** 2) for L in lk])  # observed power per coefficient at each degree
    return dk, var[keep], power, keep, lk


class Joint:
    def __init__(self, z, topo=None, geoid=None, mantle=True):
        self.z, self.mantle = z, mantle
        self.topo, self.geoid = topo, geoid
        self.nr = z.size if mantle else 0
        if topo is not None:
            Gh, d, s, Yc, self.deg = topo
            self.nc = Yc.shape[1]
            A = np.hstack([Yc] + ([Gh] if mantle else []) + [np.ones((d.size, 1))])
            W = 1 / s ** 2
            self.Mt = A.T @ (A * W[:, None])
            self.bt = A.T @ (d * W)
            self.dDd_t = float(d @ (d * W))
            self.logdet_t = float(np.sum(np.log(s ** 2)))
            self.n_t = d.size
        else:
            self.nc = 0
        if geoid is not None:
            self.GN, self.dN, self.vN, self.PN = geoid
            self.n_N = self.dN.size
        self.dim = self.nc + self.nr + (1 if topo is not None else 0)

    BOUNDS = dict(sigma_c=(10.0, 2000.0), ell_c=(50e3, 5000e3), tau=(1e-3, 3.0), sigma_R=(1e-3, 10.0), ell_R=(50.0, 5000.0))

    def unpack(self, x):
        """log-parameters, clipped to physical bounds (BOUNDS)."""
        b = self.BOUNDS
        e = lambda v, k: float(np.exp(np.clip(v, np.log(b[k][0]), np.log(b[k][1]))))
        th, i = {}, 0
        if self.topo is not None:
            th["sigma_c"], th["ell_c"] = e(x[0], "sigma_c"), e(x[1], "ell_c")
            i = 2
        if self.geoid is not None:
            th["tau"] = e(x[i], "tau")
            i += 1
        if self.mantle:
            th["sigma_R"], th["ell_R"] = e(x[i], "sigma_R"), e(x[i + 1], "ell_R")
        return th

    def solve(self, th, full=True):
        n = self.dim
        M = np.zeros((n, n))
        b = np.zeros(n)
        m = np.zeros(n)
        Pinv = np.zeros((n, n))
        dDd = logdet = 0.0
        ndat = 0
        iR = slice(self.nc, self.nc + self.nr)
        if self.topo is not None:
            M += self.Mt
            b += self.bt
            dDd += self.dDd_t
            logdet += self.logdet_t
            ndat += self.n_t
            l = self.deg.astype(float)
            a = (1.0 + (th["ell_c"] / B_EARTH) ** 2 * l * (l + 1)) ** -2
            lmb = th["sigma_c"] ** 2 * 4 * np.pi * a / np.sum(a)
            Pinv[np.arange(self.nc), np.arange(self.nc)] = 1 / lmb
            logdet += np.sum(np.log(lmb))
            Pinv[-1, -1] = 1 / SIGMA_B0 ** 2
            logdet += np.log(SIGMA_B0 ** 2)
        if self.geoid is not None:
            v = self.vN + th["tau"] ** 2 * self.PN  # stated errors + modelling error, a fraction tau^2 of the power
            if self.mantle:
                M[iR, iR] += self.GN.T @ (self.GN / v[:, None])
                b[iR] += self.GN.T @ (self.dN / v)
            dDd += float(self.dN @ (self.dN / v))
            logdet += float(np.sum(np.log(v)))
            ndat += self.n_N
        if self.mantle:
            dz = self.z[:, None] - self.z[None, :]
            # a common offset of R with sd SIGMA_MU is marginalised: prior mean 0, covariance + SIGMA_MU^2 11^T
            CR = th["sigma_R"] ** 2 * np.exp(-0.5 * (dz / th["ell_R"]) ** 2) + SIGMA_MU ** 2 + 1e-10 * np.eye(self.nr)
            Cf = cho_factor(CR)
            Pinv[iR, iR] = cho_solve(Cf, np.eye(self.nr))
            logdet += 2 * np.sum(np.log(np.diag(Cf[0])))
        if n == 0:
            return -0.5 * (dDd + logdet + ndat * np.log(2 * np.pi))
        H = Pinv + M
        Hf = cho_factor(H)
        g = b - M @ m
        Hg = cho_solve(Hf, g)
        quad = dDd - 2 * m @ b + m @ M @ m - g @ Hg
        logZ = -0.5 * (quad + logdet + 2 * np.sum(np.log(np.diag(Hf[0]))) + ndat * np.log(2 * np.pi))
        if not full:
            return logZ
        Hinv = cho_solve(Hf, np.eye(n))
        return logZ, m + Hg, Hinv, Pinv

    def fit(self, starts=1, seed=0):
        x0 = ([np.log(400.0), np.log(1000e3)] if self.topo is not None else []) + \
             ([np.log(0.3)] if self.geoid is not None else []) + ([np.log(0.1), np.log(400.0)] if self.mantle else [])
        if not x0:
            return {}, self.solve({}, full=False)
        rng = np.random.default_rng(seed)
        best, self.start_logZ = None, []
        for k in range(starts):
            xs = np.array(x0) + (rng.normal(0.0, 1.0, len(x0)) if k else 0.0)
            r = minimize(lambda x: -self.solve(self.unpack(x), full=False), xs, method="Nelder-Mead",
                         options=dict(maxiter=6000, xatol=1e-4, fatol=1e-3))
            self.start_logZ.append(float(-r.fun))
            if best is None or r.fun < best.fun:
                best = r
        return self.unpack(best.x), -best.fun


def main():
    gname, mode = sys.argv[1], sys.argv[2]
    opt = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    datum, starts = opt.get("--datum", "lab"), int(opt.get("--starts", 1))
    tag = f"{Path(gname).stem}_{mode}" + ("" if datum == "lab" else f"_{datum}")
    c = np.load(OUT / gname)
    z = c["nodes_km"]
    topo = geoid = None
    if mode in ("topo", "joint"):
        raw = np.loadtxt(DATA)[N_STEPHENSON:]
        Y = harmonics_at(*np.loadtxt(DATA)[:, [1, 0]].T, LC, OUT / f"Y_points_L{LC}.npy")[N_STEPHENSON:]
        deg = np.concatenate([[l] * (2 * l + 1) for l in range(LC + 1)])
        Gh = c["G_points"][N_STEPHENSON:]
        lat, lon, d, s, (Gh, Yt) = thin(raw[:, 1], raw[:, 0], raw[:, 2] * 1e3, raw[:, 3] * 1e3, [Gh, Y[:, deg >= 1]])
        print(f"{raw.shape[0]} Holdt points -> {d.size} one-degree cells", flush=True)
        topo = (Gh, d, s, Yt, deg[deg >= 1])
    if mode in ("geoid", "joint"):
        dN, vN, PN, keep, gl = geoid_data(datum)
        geoid = (c["G_geoid"][keep], dN, vN, PN)
    res = {}
    for name, mantle in (("null", False), ("scaling", True)):
        p = Joint(z, topo, geoid, mantle)
        th, logZ = p.fit(starts)
        out = dict(logZ=float(logZ), hyper={k: float(v) for k, v in th.items()}, start_logZ=getattr(p, "start_logZ", []))
        if mantle:
            _, pm, pc, Pinv = p.solve(th)
            iR = slice(p.nc, p.nc + p.nr)
            R, Cp = pm[iR], pc[iR, iR]
            Cprior = np.linalg.inv(Pinv[iR, iR])
            out.update(R_mean=R.tolist(), R_sd=np.sqrt(np.diag(Cp)).tolist(),
                       trace_resolution=float(np.trace(np.eye(p.nr) - Cp @ Pinv[iR, iR])))
            if topo is not None:
                Gh, d, s, Yc, _ = topo
                b0 = pm[-1]
                r0 = d - b0
                pr = Gh @ R
                W = 1 / s ** 2
                out["topo"] = dict(VR=float(1 - np.sum(W * (r0 - pr) ** 2) / np.sum(W * r0 ** 2)),
                                   corr=float(np.corrcoef(pr, r0)[0, 1]), rms_pred=float(np.std(pr)),
                                   rms_corr_field=float(np.std(Yc @ pm[: p.nc])), rms_data=float(np.std(r0)), b0=float(b0))
            if geoid is not None:
                GN, dN, vN, PN = geoid
                pr = GN @ R
                v = vN + th["tau"] ** 2 * PN
                out["geoid"] = dict(VR=float(1 - np.sum((dN - pr) ** 2 / vN) / np.sum(dN ** 2 / vN)),
                                    chi2_n_stated=float(np.mean((dN - pr) ** 2 / vN)),
                                    chi2_n_total=float(np.mean((dN - pr) ** 2 / v)),
                                    corr=float(np.corrcoef(pr, dN)[0, 1]))
                out["geoid_corr_by_degree"] = {int(L): float(np.corrcoef(pr[gl == L], dN[gl == L])[0, 1]) for L in range(2, 9)}
            np.savez(OUT / f"posterior_{tag}.npz", R_mean=R, R_cov=Cp, R_prior_cov=Cprior, nodes_km=z)
        res[name] = out
    res["log_bayes_factor"] = res["scaling"]["logZ"] - res["null"]["logZ"]
    s = res["scaling"]
    print(mode, "logBF", round(res["log_bayes_factor"], 1), "hyper", {k: round(v, 3) for k, v in s["hyper"].items()},
          "trRes", round(s["trace_resolution"], 1), "topo", s.get("topo"), "geoid", s.get("geoid"), flush=True)
    print("R(z):", ", ".join(f"{zz:.0f}:{r:+.3f}±{sd:.3f}" for zz, r, sd in zip(z, s["R_mean"], s["R_sd"])))
    res["datum"], res["starts"] = datum, starts
    print("logZ over starts (scaling):", [round(v, 1) for v in s["start_logZ"]])
    (OUT / f"result_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
