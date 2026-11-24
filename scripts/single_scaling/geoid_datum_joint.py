"""Joint topography + geoid single scaling with the geoid datum's lithosphere cut at WINTERC-G's LAB (the product)
or at 105 km (consistent with the core model's flow top). Written by Claude, 23 Nov 2026.
Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/geoid_datum_joint.py core_<...>.npz {lab,105}
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import invert_core as ic  # noqa: E402

c = np.load(ic.OUT / sys.argv[1]); which = sys.argv[2]
z = c["nodes_km"]
g = np.load(ic.PRODUCT)
_, var, _, keep, lk = ic.geoid_data()
d = g["d"] if which == "lab" else g["N_obs"] - np.load(Path(__file__).resolve().parents[3] / "ch2_density/geoid/cache/N_lith_cut105.npy")
d = d[keep]
power = np.array([np.mean(d[lk == L] ** 2) for L in lk])
raw = np.loadtxt(ic.DATA)[ic.N_STEPHENSON:]
Y = ic.harmonics_at(*np.loadtxt(ic.DATA)[:, [1, 0]].T, ic.LC, ic.OUT / f"Y_points_L{ic.LC}.npy")[ic.N_STEPHENSON:]
deg = np.concatenate([[l] * (2 * l + 1) for l in range(ic.LC + 1)])
lat, lon, dd, s, (Gh, Yt) = ic.thin(raw[:, 1], raw[:, 0], raw[:, 2] * 1e3, raw[:, 3] * 1e3, [c["G_points"][ic.N_STEPHENSON:], Y[:, deg >= 1]])
topo = (Gh, dd, s, Yt, deg[deg >= 1])
geoid = (c["G_geoid"][keep], d, var, power)
out = {}
for label, mantle in (("null", False), ("scaling", True)):
    p = ic.Joint(z, topo, geoid, mantle)
    th, logZ = p.fit()
    out[label] = logZ
    if mantle:
        _, pm, pc, _ = p.solve(th)
        iR = slice(p.nc, p.nc + p.nr)
        R, sd = pm[iR], np.sqrt(np.diag(pc))[iR]
        pr = c["G_geoid"][keep] @ R
        r0 = dd - pm[-1]; prt = Gh @ R; W = 1 / s ** 2
        out.update(R=R.tolist(), sd=sd.tolist(), hyper=th,
                   topo_VR=float(1 - np.sum(W * (r0 - prt) ** 2) / np.sum(W * r0 ** 2)),
                   geoid_VR=float(1 - np.sum((d - pr) ** 2 / var) / np.sum(d ** 2 / var)))
out["lnBF"] = out["scaling"] - out["null"]
print(which, f"lnBF {out['lnBF']:.1f}", {k: round(v, 3) for k, v in out["hyper"].items()}, f"topoVR {out['topo_VR']:.2f} geoidVR {out['geoid_VR']:.2f}")
print("R(z):", " ".join(f"{zz:.0f}:{r:+.2f}" for zz, r in zip(z, out["R"])))
(ic.OUT / f"geoid_datum_joint_{which}.json").write_text(json.dumps(out, indent=1))
