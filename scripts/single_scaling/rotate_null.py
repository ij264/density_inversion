"""Rotation null for the core topography result (written by Claude, 19 Nov 2026).

Rotate the tomography-predicted topography of every scaling node by the same random rotation, so the depth
structure and spectra are kept but the geography is not, and refit. If the real log Bayes factor is far above the
rotated ones, the fit comes from where the predicted structure is, not from its spectrum.

Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/rotate_null.py core_<...>.npz first count
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyshtools as pysh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_inversion.forward.examples import cilm_to_components  # noqa: E402
from invert import DATA, LC, N_STEPHENSON, OUT, harmonics_at  # noqa: E402
from invert_core import Joint, thin  # noqa: E402


def main():
    gname, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    c = np.load(OUT / gname)
    z, H = c["nodes_km"], c["h_cilm"]
    lt = H.shape[-1] - 1
    allp = np.loadtxt(DATA)
    Y = harmonics_at(allp[:, 1], allp[:, 0], LC, OUT / f"Y_points_L{LC}.npy")[N_STEPHENSON:]
    raw = allp[N_STEPHENSON:]
    deg = np.concatenate([[l] * (2 * l + 1) for l in range(LC + 1)])
    f = OUT / f"rotation_null_{Path(gname).stem}.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    for k in range(first, first + count):
        rng = np.random.default_rng(1000 + k)
        a, g = rng.uniform(0, 360, 2)
        b = np.degrees(np.arccos(rng.uniform(-1, 1)))
        G = np.empty((raw.shape[0], z.size))
        for i in range(z.size):
            rot = pysh.SHCoeffs.from_array(H[i], normalization="ortho", csphase=1).rotate(a, b, g, degrees=True).coeffs
            G[:, i] = Y[:, : (lt + 1) ** 2] @ cilm_to_components(rot, lt)
        lat, lon, d, s, (Gt, Yt) = thin(raw[:, 1], raw[:, 0], raw[:, 2] * 1e3, raw[:, 3] * 1e3, [G, Y[:, deg >= 1]])
        topo = (Gt, d, s, Yt, deg[deg >= 1])
        _, zn = Joint(z, topo, None, False).fit()
        th, zs = Joint(z, topo, None, True).fit()
        rows.append(dict(k=k, euler=[float(a), float(b), float(g)], logBF=float(zs - zn)))
        print(k, round(zs - zn, 2), flush=True)
        f.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
