"""Synthetic calibration of invert.py (written by Claude, 19 Nov 2026).

(a) recovery: d = G R_true + b0 + c + e with R_true smooth and unlike the prior mean, c drawn from the correlated
    error model at the hyperparameters fitted to the real data, e from the stated point errors. Same inversion
    (hyperparameters re-fitted). Reports coverage of R_true by the 2-sigma posterior and the Bayes factor.
(b) false positive: R_true = 0. Reports the Bayes factor, which should be near or below zero.
Same points, errors and G as the real run.

Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/synthetic.py G_<...>.npz [recovery|false_positive] [seed]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from invert import B_EARTH, DATA, LC, OUT, Problem, fit, harmonics_at  # noqa: E402


def main():
    gname = sys.argv[1]
    only_case = sys.argv[2] if len(sys.argv) > 2 else None
    seeds = [int(sys.argv[3])] if len(sys.argv) > 3 else range(4)
    g = np.load(OUT / gname)
    G, z = g["G_points"], g["nodes_km"]
    real = json.loads((OUT / f"result_{Path(gname).stem}.json").read_text())["scaling"]["hyper"]
    raw = np.loadtxt(DATA)
    s = raw[:, 3] * 1e3
    Y = harmonics_at(raw[:, 1], raw[:, 0], LC, OUT / f"Y_points_L{LC}.npy")
    degrees = np.concatenate([[l] * (2 * l + 1) for l in range(LC + 1)])
    l = degrees[degrees >= 1].astype(float)
    a = (1.0 + (real["ell_c"] / B_EARTH) ** 2 * l * (l + 1)) ** -2
    lmb = real["sigma_c"] ** 2 * 4 * np.pi * a / np.sum(a)
    R_true = 0.10 + 0.08 * np.tanh((z - 1200.0) / 400.0)
    out = []
    for case, Rt in (("recovery", R_true), ("false_positive", np.zeros_like(z))):
        if only_case and case != only_case:
            continue
        for seed in seeds:
            rng = np.random.default_rng(seed)
            c = Y[:, degrees >= 1] @ (np.sqrt(lmb) * rng.standard_normal(lmb.size))
            d = G @ Rt + 80.0 + c + s * rng.standard_normal(s.size)
            x0c = [np.log(400.0), np.log(1000e3)]
            pn = Problem(G, d, s, Y, degrees, z, correlated=True, mantle=False)
            _, zn = fit(pn, np.array(x0c))
            ps = Problem(G, d, s, Y, degrees, z, correlated=True, mantle=True)
            th, zs = fit(ps, np.array(x0c + [0.05, np.log(0.1), np.log(400.0)]))
            _, pm, pc, _, _ = ps.solve(th)
            i = slice(ps.nc, ps.nc + ps.nr)
            R, sd = pm[i], np.sqrt(np.diag(pc[i, i]))
            cover = float(np.mean(np.abs(R - Rt) <= 2 * sd))
            rms_err = float(np.sqrt(np.mean((R - Rt) ** 2)))
            row = dict(case=case, seed=seed, logBF=float(zs - zn), coverage_2sd=cover, rms_error_R=rms_err,
                       hyper={k: float(v) for k, v in th.items()}, R=R.tolist(), sd=sd.tolist())
            out.append(row)
            print(case, seed, f"logBF {zs - zn:8.1f}  coverage {cover:.2f}  rms err {rms_err:.3f}", flush=True)
    tag = f"_{only_case}_{seeds[0]}" if only_case else ""
    (OUT / f"synthetic_{Path(gname).stem}{tag}.json").write_text(json.dumps(dict(R_true=R_true.tolist(), nodes_km=z.tolist(), runs=out), indent=1))


if __name__ == "__main__":
    main()
