"""Geoid-only single scaling with different lithosphere corrections in the datum (written by Claude, 23 Nov 2026).

The datum is d = N_obs - N_lith. The product uses WINTERC-G's lithosphere (surface to its LAB). Here N_lith is
replaced by each alternative behind the product's variances (CRUST1.0 / ECM1 crust, four seismic LAB models), by
WINTERC-G cut at 105 km, and by no correction at all, to see how much the geoid-only R(z) depends on it.
Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/geoid_datum_test.py core_<...>.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import invert_core as ic  # noqa: E402

c = np.load(ic.OUT / sys.argv[1])
z = c["nodes_km"]
g = np.load(ic.PRODUCT)
base_d, var, power, keep, lk = ic.geoid_data()
cut105 = np.load(Path(__file__).resolve().parents[3] / "ch2_density/geoid/cache/N_lith_cut105.npy")
alts = {"WINTERC-G to LAB (product)": np.zeros(g["l"].size)}
alts.update({k.replace("delta_", "swap "): g[k] for k in g.files if k.startswith("delta_")})
alts["WINTERC-G cut at 105 km"] = cut105 - g["N_lith"]
alts["no lithosphere correction"] = -g["N_lith"]
res = {}
for name, delta in alts.items():
    d = (g["d"] - delta)[keep]
    power_d = np.array([np.mean(d[lk == L] ** 2) for L in lk])
    out = {}
    for label, mantle in (("null", False), ("scaling", True)):
        p = ic.Joint(z, None, (c["G_geoid"][keep], d, var, power_d), mantle)
        th, logZ = p.fit()
        out[label] = logZ
        if mantle:
            _, pm, pc, _ = p.solve(th)
            R, sd = pm[: z.size], np.sqrt(np.diag(pc))[: z.size]
            out.update(R=R.tolist(), sd=sd.tolist(), hyper=th)
    res[name] = dict(lnBF=out["scaling"] - out["null"], R105=out["R"][0], sd105=out["sd"][0], R=out["R"], hyper=out["hyper"])
    r = res[name]
    print(f"{name:32s} lnBF {r['lnBF']:6.1f}  R(105) {r['R105']:+.2f} +- {r['sd105']:.2f}  R(200..600) "
          + " ".join(f"{v:+.2f}" for v in r["R"][1:6]) + f"  tau {r['hyper']['tau']:.2f}", flush=True)
(ic.OUT / "geoid_datum_test.json").write_text(json.dumps(res, indent=1))
