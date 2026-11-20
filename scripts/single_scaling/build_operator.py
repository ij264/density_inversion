"""Single 1D velocity-to-density scaling: the linear map R(z) -> predicted residual depth at the data points.

Written by Claude (AI assistant), 19 Nov 2026, for the rerun of the Chapter 3 (thesis numbering) single-scaling
experiment with the current forward model.

delta rho_m(r, x) = rho_0(r) R(r) dlnVs(r, x), below the 105 km lid base, with R on hat functions in depth.
Each hat i gives a mantle density field; its load and direct geoid come from the cached flow kernels by radial
quadrature (MantleFlow.kernels: q_lm = sum w K_q[l](r) drho_lm(r), the same for K_N), and then go through the same
ocean + plate + geoid solve as ForwardModel's mantle block. h = w - N is evaluated at the data points with
pygeoinf's point-evaluation operator. Lithosphere at the reference (x_L = 0), dK = 0, h_s = 0.

Usage:
  PYTHONPATH=src ~/pgenv/bin/python scripts/single_scaling/build_operator.py \
      --visc jump30 --tomo sl2013sv.s40rts.31.m.ab --lmax 64 --Te 40e3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from density_inversion.forward.examples import cilm_to_components
from density_inversion.forward.mantle import MantleFlow, viscosity_jump
from density_inversion.forward.model import ForwardModel

DEV = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DEV / "data" / "server_src"))
from hc_io import read_hc_sh_short  # noqa: E402

TOMO_DIR = DEV / "data" / "hc" / "tomography"
DATA = DEV / "data" / "hoggard" / "global.xyz"  # Stephenson (2024) continental bins (first 4119 rows) + Holdt (2022)
OUT = Path(__file__).resolve().parents[2] / "outputs" / "single_scaling"
NODES_KM = np.concatenate([[105.0], np.arange(200.0, 2801.0, 100.0)])  # 28 nodes


def viscosity(name):
    if name == "jump30":
        return viscosity_jump(30.0, 670e3)
    if name == "jump10":
        return viscosity_jump(10.0, 670e3)
    if name == "jump100":
        return viscosity_jump(100.0, 670e3)
    if name == "const":  # density_space needs a jump to split the ball; 1 + 1e-9 is constant for all purposes
        return viscosity_jump(1.0 + 1e-9, 670e3)
    raise ValueError(name)


def hats(depth_km):
    """(n_nodes, n_depths) piecewise-linear basis in depth; flat below the deepest node, zero above the first."""
    z = np.asarray(depth_km, float)
    B = np.zeros((NODES_KM.size, z.size))
    for i, zi in enumerate(NODES_KM):
        e = np.zeros(NODES_KM.size)
        e[i] = 1.0
        B[i] = np.interp(z, NODES_KM, e, left=0.0, right=e[-1])
    B[:, z < NODES_KM[0] - 1e-6] = 0.0
    return B


def tomography_at(model, depth_km):
    """dlnVs as a fraction, (n_depths, 2, L+1, L+1), linear in depth between the model's layers, clamped at its ends."""
    d, c = model["depths_km"], model["cilm_layers"]
    o = np.argsort(d)
    d, c = d[o], c[o]
    out = np.empty((len(depth_km),) + c.shape[1:])
    for k, z in enumerate(depth_km):
        z = float(np.clip(z, d[0], d[-1]))
        j = min(max(1, int(np.searchsorted(d, z))), len(d) - 1)
        t = (z - d[j - 1]) / (d[j] - d[j - 1])
        out[k] = (1 - t) * c[j - 1] + t * c[j]
    out[:, 0, 0, 0] = 0.0  # degree 0 carries no topography or geoid (mass conservation)
    return out / 100.0


def rho0_on_kernel_grid(mf):
    s = mf.shell
    eta = s.viscosity(1)
    i_jump = int(np.flatnonzero(np.diff(eta) != 0)[0]) + 1
    idx = np.concatenate([np.arange(0, i_jump + 1), np.arange(i_jump, s.n)])
    return np.asarray(s.rho)[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visc", default="jump30")
    ap.add_argument("--tomo", default="sl2013sv.s40rts.31.m.ab")
    ap.add_argument("--lmax", type=int, default=64)
    ap.add_argument("--Te", type=float, default=40e3)
    a = ap.parse_args()
    t0 = time.time()

    mf = MantleFlow(viscosity=viscosity(a.visc), lab_depth=105e3, lmax=a.lmax)
    fm = ForwardModel(mf, Te=a.Te)
    Q, k = fm.Q, mf.kernels
    L = a.lmax
    print(f"forward model built ({time.time() - t0:.0f} s)", flush=True)

    r, w = k["r"], k["weights"]
    depth = (float(k["R"]) - r) / 1e3
    rho0 = rho0_on_kernel_grid(mf)
    tomo = read_hc_sh_short(TOMO_DIR / a.tomo)
    lt = tomo["lmax"]
    dv = tomography_at(tomo, depth)  # (n_r, 2, lt+1, lt+1)
    B = hats(depth)  # (n_nodes, n_r)

    pts = np.loadtxt(DATA)
    lat, lon = pts[:, 1], pts[:, 0]
    E = Q.point_evaluation_operator(np.column_stack([lat, lon]), route="dense")
    deg = np.concatenate([[l] * (2 * l + 1) for l in range(L + 1)])

    ap_ = fm._apply
    cols_h, cols_N, coef_h, coef_N = [], [], [], []
    for i in range(NODES_KM.size):
        # drho_i(r) = rho0(r) B_i(r) dlnVs(r): quadrature against the kernels, degree by degree
        f = (w * rho0 * B[i])[:, None, None, None] * dv  # (n_r, 2, lt+1, lt+1)
        q = np.zeros((2, L + 1, L + 1))
        D = np.zeros((2, L + 1, L + 1))
        for l in range(1, min(lt, L) + 1):
            q[:, l, : l + 1] = np.tensordot(k["K_q"][l - 1], f[:, :, l, : l + 1], axes=(0, 0))
            D[:, l, : l + 1] = np.tensordot(k["K_N"][l - 1], f[:, :, l, : l + 1], axes=(0, 0))
        qc, Dc = cilm_to_components(q, L), cilm_to_components(D, L)
        rr = Dc + ap_(fm.G_w @ fm.P_H, qc)
        N = ap_(fm.ocean.inverse, rr)
        wH = ap_(fm.P_H, qc) + ap_(fm.P_H @ fm.ocean.load, N)
        h = ap_(fm.truncate, wH) - N
        coef_h.append(h)
        coef_N.append(N)
        cols_h.append(ap_(E, h))
        print(f"node {NODES_KM[i]:6.0f} km: |h| {np.sqrt((h[deg >= 2] ** 2).sum() / (4 * np.pi)):8.2f} m rms  "
              f"({time.time() - t0:.0f} s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"G_{a.tomo.replace('.ab', '')}_{a.visc}_L{L}_Te{a.Te / 1e3:g}.npz"
    np.savez(
        f,
        G_points=np.array(cols_h).T, h_coef=np.array(coef_h), N_coef=np.array(coef_N),
        nodes_km=NODES_KM, lat=lat, lon=lon, lmax=L, tomo=a.tomo, visc=a.visc, Te=a.Te,
        notes=np.array("G_points: (n_points, n_nodes) predicted residual depth [m] per unit R at each node; "
                       "h_coef/N_coef: sphere components (orthonormal, unit radius) of h and N per node [m]"),
    )
    print(f"saved {f} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
