"""Single 1D scaling R(z) fitted to residual depths at the data points, exactly (dense linear algebra).

Written by Claude (AI assistant), 19 Nov 2026.

Data: d_i = residual depth at point i (Stephenson 2024 continental bins + Holdt 2022 oceanic), with its own
independent error s_i. Model:
    d = G R + b0 + c(x_i) + e,   e ~ N(0, diag s^2)
  G R : the mantle's dynamic topography at the points for a scaling R on 28 depth nodes (build_operator.py)
  b0  : a constant (residual depths have no fixed zero; dynamic topography has no degree 0), prior sd 1 km
  c   : a spatially correlated field the 1D scaling does not explain (lithospheric and crustal error, structure
        finer than the tomography, laterally varying scaling): degrees 1..LC, pointwise sd sigma_c, spectrum
        (1 + (ell_c / b)^2 l(l+1))^-2
  R   : Gaussian, mean mu_R at every node, sd sigma_R, squared-exponential correlation in depth, length ell_R
Hyperparameters (sigma_c, ell_c, mu_R, sigma_R, ell_R) maximise the log evidence. The null model (R = 0) uses
the same noise model, so the Bayes factor measures what the 1D scaling explains beyond a correlated field.

Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/invert.py G_<...>.npz [--no-correlated]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyshtools as pysh
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from density_inversion.forward.examples import cilm_to_components

HERE = Path(__file__).resolve()
OUT = HERE.parents[2] / "outputs" / "single_scaling"
DATA = HERE.parents[3] / "data" / "hoggard" / "global.xyz"
B_EARTH = 6371e3
LC = 40
SIGMA_B0 = 1000.0
N_STEPHENSON = 4119


def harmonics_at(lat, lon, lmax, cache):
    """(n_points, (lmax+1)^2) orthonormal real harmonics, csphase +1, pygeoinf component order."""
    if cache.exists():
        return np.load(cache)
    Y = np.empty((lat.size, (lmax + 1) ** 2))
    for i, (la, lo) in enumerate(zip(lat, lon)):
        y = pysh.expand.spharm(lmax, 90.0 - la, lo, normalization="ortho", csphase=1, kind="real", degrees=True)
        Y[i] = cilm_to_components(y, lmax)
    np.save(cache, Y)
    return Y


class Problem:
    def __init__(self, G, d, s, Y, degrees, z_km, correlated=True, mantle=True):
        self.correlated, self.mantle = correlated, mantle
        cols = []
        if correlated:
            keep = degrees >= 1
            cols.append(Y[:, keep])
            self.deg = degrees[keep]
        self.nc = cols[0].shape[1] if correlated else 0
        self.nr = G.shape[1] if mantle else 0
        if mantle:
            cols.append(G)
        cols.append(np.ones((d.size, 1)))
        A = np.hstack(cols)
        Wd = 1.0 / s ** 2
        self.M = A.T @ (A * Wd[:, None])
        self.bvec = A.T @ (d * Wd)
        self.dDd = float(d @ (d * Wd))
        self.logdetD = float(np.sum(np.log(s ** 2)))
        self.n = d.size
        self.z = z_km

    def prior(self, th):
        """(mean, covariance) of [c, R, b0] for hyperparameters th (a dict)."""
        blocks, mean = [], []
        if self.correlated:
            l = self.deg.astype(float)
            a = (1.0 + (th["ell_c"] / B_EARTH) ** 2 * l * (l + 1)) ** -2
            lmb = th["sigma_c"] ** 2 * 4 * np.pi * a / np.sum(a)  # sum over components = sum (2l+1) a_l
            blocks.append(np.diag(lmb))
            mean.append(np.zeros(self.nc))
        if self.mantle:
            dz = self.z[:, None] - self.z[None, :]
            C = th["sigma_R"] ** 2 * np.exp(-0.5 * (dz / th["ell_R"]) ** 2) + 1e-10 * np.eye(self.z.size)
            blocks.append(C)
            mean.append(np.full(self.z.size, th["mu_R"]))
        blocks.append(np.array([[SIGMA_B0 ** 2]]))
        mean.append(np.zeros(1))
        n = sum(b.shape[0] for b in blocks)
        P = np.zeros((n, n))
        k = 0
        for b in blocks:
            P[k:k + b.shape[0], k:k + b.shape[0]] = b
            k += b.shape[0]
        return np.concatenate(mean), P

    def solve(self, th, full=True):
        m, P = self.prior(th)
        # P is block diagonal: diagonal for the correlated field and b0, dense 28 x 28 for R
        Pinv = np.zeros_like(P)
        logdetP = 0.0
        k = 0
        if self.correlated:
            dg = np.diag(P)[: self.nc]
            Pinv[np.arange(self.nc), np.arange(self.nc)] = 1.0 / dg
            logdetP += np.sum(np.log(dg))
            k = self.nc
        if self.mantle:
            Cf = cho_factor(P[k:k + self.nr, k:k + self.nr])
            Pinv[k:k + self.nr, k:k + self.nr] = cho_solve(Cf, np.eye(self.nr))
            logdetP += 2 * np.sum(np.log(np.diag(Cf[0])))
            k += self.nr
        Pinv[-1, -1] = 1.0 / P[-1, -1]
        logdetP += np.log(P[-1, -1])
        H = Pinv + self.M
        Hf = cho_factor(H)
        g = self.bvec - self.M @ m
        rDr = self.dDd - 2 * m @ self.bvec + m @ self.M @ m
        Hg = cho_solve(Hf, g)
        quad = rDr - g @ Hg
        logdet = self.logdetD + logdetP + 2 * np.sum(np.log(np.diag(Hf[0])))
        logZ = -0.5 * (quad + logdet + self.n * np.log(2 * np.pi))
        if not full:
            return logZ
        post_mean = m + Hg
        post_cov = cho_solve(Hf, np.eye(H.shape[0]))
        return logZ, post_mean, post_cov, m, P


def unpack(x, problem):
    th = {}
    i = 0
    if problem.correlated:
        th["sigma_c"], th["ell_c"] = np.exp(x[0]), np.exp(x[1])
        i = 2
    if problem.mantle:
        th["mu_R"], th["sigma_R"], th["ell_R"] = x[i], np.exp(x[i + 1]), np.exp(x[i + 2])
    return th


def fit(problem, x0):
    f = lambda x: -problem.solve(unpack(x, problem), full=False)
    best = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-3))
    return unpack(best.x, problem), -best.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("G")
    ap.add_argument("--no-correlated", action="store_true")
    a = ap.parse_args()
    g = np.load(OUT / a.G)
    G, z = g["G_points"], g["nodes_km"]
    raw = np.loadtxt(DATA)
    d, s = raw[:, 2] * 1e3, raw[:, 3] * 1e3
    lat, lon = raw[:, 1], raw[:, 0]
    Y = harmonics_at(lat, lon, LC, OUT / f"Y_points_L{LC}.npy")
    degrees = np.concatenate([[l] * (2 * l + 1) for l in range(LC + 1)])
    corr = not a.no_correlated

    res = {}
    x0_c = [np.log(400.0), np.log(1000e3)] if corr else []
    for name, mantle in (("null", False), ("scaling", True)):
        cache = OUT / f"null{'' if corr else '_nocorr'}.json"
        if not mantle and cache.exists():  # the null model does not depend on G
            res[name] = json.loads(cache.read_text())
            print(name, "(cached)", res[name]["logZ"], flush=True)
            continue
        pr = Problem(G, d, s, Y, degrees, z, correlated=corr, mantle=mantle)
        x0 = x0_c + ([0.05, np.log(0.1), np.log(400.0)] if mantle else [])
        if not x0:
            th, logZ = {}, pr.solve({}, full=False)
        else:
            th, logZ = fit(pr, np.array(x0))
        logZ, pm, pc, m, P = pr.solve(th)
        # split the posterior mean into its parts and measure each against the data
        c = pm[: pr.nc]
        R = pm[pr.nc: pr.nc + pr.nr]
        b0 = pm[-1]
        pred_R = G @ R if mantle else np.zeros_like(d)
        pred_c = Y[:, degrees >= 1] @ c if corr else np.zeros_like(d)
        out = dict(logZ=float(logZ), hyper={k: float(v) for k, v in th.items()}, b0=float(b0))
        for lab, sl in (("all", slice(None)), ("stephenson", slice(0, N_STEPHENSON)), ("holdt", slice(N_STEPHENSON, None))):
            r0 = d[sl] - b0
            w = 1 / s[sl] ** 2
            out[lab] = dict(
                chi2_n_full=float(np.mean(((r0 - pred_R[sl] - pred_c[sl]) / s[sl]) ** 2)),
                chi2_n_mantle_only=float(np.mean(((r0 - pred_R[sl]) / s[sl]) ** 2)),
                VR_mantle=float(1 - np.sum(w * (r0 - pred_R[sl]) ** 2) / np.sum(w * r0 ** 2)),
                corr_mantle=float(np.corrcoef(pred_R[sl], r0)[0, 1]) if mantle else None,
                rms_mantle=float(np.std(pred_R[sl])), rms_correlated=float(np.std(pred_c[sl])), rms_data=float(np.std(r0)),
            )
        if mantle:
            iR = slice(pr.nc, pr.nc + pr.nr)
            Cpost = pc[iR, iR]
            Cprior = P[iR, iR]
            res_matrix = np.eye(pr.nr) - Cpost @ np.linalg.inv(Cprior)
            out.update(R_mean=R.tolist(), R_sd=np.sqrt(np.diag(Cpost)).tolist(), nodes_km=z.tolist(),
                       resolution_diag=np.diag(res_matrix).tolist(), trace_resolution=float(np.trace(res_matrix)))
            np.savez(OUT / f"posterior_{Path(a.G).stem}{'_nocorr' if not corr else ''}.npz",
                     R_mean=R, R_cov=Cpost, R_prior_cov=Cprior, resolution=res_matrix, nodes_km=z,
                     pred_mantle=pred_R, pred_correlated=pred_c, b0=b0, d=d, s=s, lat=lat, lon=lon)
        res[name] = out
        if not mantle:
            cache.write_text(json.dumps(out))
        print(name, json.dumps({k: v for k, v in out.items() if k not in ("R_mean", "R_sd", "nodes_km", "resolution_diag")}, indent=1), flush=True)
    res["log_bayes_factor_scaling_vs_null"] = res["scaling"]["logZ"] - res["null"]["logZ"]
    print("log Bayes factor (scaling vs null):", round(res["log_bayes_factor_scaling_vs_null"], 1))
    print("R(z):", ", ".join(f"{zz:.0f}:{r:+.3f}±{sd:.3f}" for zz, r, sd in zip(z, res["scaling"]["R_mean"], res["scaling"]["R_sd"])))
    (OUT / f"result_{Path(a.G).stem}{'_nocorr' if not corr else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
