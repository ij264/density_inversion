"""Core forward model (stage 1): flow kernels with a water-loaded free surface, no plate, no land/sea.

Written by Claude (AI assistant), 19 Nov 2026.

For a mantle density field below the 105 km flow top, degree by degree (everything is diagonal):
  q_lm = sum_r W K_q[l](r) drho_lm(r)        rock load on the flow top (flowkernels normal stress), Pa
  D_lm = sum_r W K_N[l](r) drho_lm(r)        geoid of the anomaly and the deflected CMB, without the flow top's mass
  The lithosphere has no strength and rides on the flow top, so the whole column is lifted by w; its mass change
  (rho_LAB - rho_w) w is put at the surface, and an ocean covers the surface.
  Static ocean (default, agreed with David 23 Nov 2026): the sea surface does not follow the geoid, so
    force up = force down:  q = (rho_LAB - rho_w) g w
    geoid:                  N = D + S_l(R) (rho_LAB - rho_w) w,  S_l(R) = 4 pi G R / ((2l+1) g0)
  and the topography is still measured from the geoid, h = w - N.
  --ocean geoid restores the ocean that follows the geoid (the earlier runs):
    q = (rho_LAB - rho_w) g w + rho_w g0 N,  N = D + S [(rho_LAB - rho_w) w + rho_w N]
    so N = (D + S q / g) / (1 + rho_w S (g0/g - 1)),  w = (q - rho_w g0 N) / ((rho_LAB - rho_w) g).
Putting the deflected mass at the flow top instead (flowkernels' own K_h0, K_N_fk) misplaces it by 105 km, which
changes the compensated geoid by factors of 0.2-3 at low degree.
Outputs, per scaling node: h at every residual-depth point (via orthonormal real harmonics, csphase +1), and the
geoid in the convention of products/geoid_for_inversion.npz (ortho, csphase -1, radius a, degrees 2-25).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_operator import NODES_KM, OUT, TOMO_DIR, hats, read_hc_sh_short, rho0_on_kernel_grid, tomography_at, viscosity  # noqa: E402
from density_inversion.forward.mantle import MantleFlow  # noqa: E402
from invert import DATA, LC, harmonics_at  # noqa: E402

PRODUCT = Path(__file__).resolve().parents[3] / "products" / "geoid_for_inversion.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visc", default="jump30")
    ap.add_argument("--tomo", default="sl2013sv.s40rts.31.m.ab")
    ap.add_argument("--ocean", choices=("static", "geoid"), default="static")
    a = ap.parse_args()
    mf = MantleFlow(viscosity=viscosity(a.visc), lab_depth=105e3, lmax=40)
    k = mf.kernels
    r, wq = k["r"], k["weights"]
    R, g0, r_top, g_top = float(k["R"]), float(k["g0"]), float(k["r_lab"]), float(k["g_lab"])
    rho_lab, rho_w = float(k["rho_lab"]), float(k["rho_water"])
    G_const = mf.shell.constants.G
    depth = (R - r) / 1e3
    rho0 = rho0_on_kernel_grid(mf)
    tomo = read_hc_sh_short(TOMO_DIR / a.tomo)
    lt = tomo["lmax"]
    dv = tomography_at(tomo, depth)
    B = hats(depth)

    raw = np.loadtxt(DATA)
    Y = harmonics_at(raw[:, 1], raw[:, 0], LC, OUT / f"Y_points_L{LC}.npy")[:, : (lt + 1) ** 2]
    from density_inversion.forward.examples import cilm_to_components

    gp = np.load(PRODUCT)
    gl, gm, a_ref = gp["l"], gp["m"], float(gp["radius"])

    Gh, GN, hcoef, Ncoef = [], [], [], []
    for i in range(NODES_KM.size):
        f = (wq * rho0 * B[i])[:, None, None, None] * dv
        q = np.zeros((2, lt + 1, lt + 1))
        D = np.zeros((2, lt + 1, lt + 1))
        for l in range(1, lt + 1):
            q[:, l, : l + 1] = np.tensordot(k["K_q"][l - 1], f[:, :, l, : l + 1], axes=(0, 0))
            D[:, l, : l + 1] = np.tensordot(k["K_N"][l - 1], f[:, :, l, : l + 1], axes=(0, 0))
        # no plate, ocean everywhere, deflection mass at the surface: diagonal in degree
        ls = np.arange(lt + 1)[None, :, None]
        S = 4 * np.pi * G_const * R / ((2 * ls + 1) * g0)
        if a.ocean == "static":
            w = q / ((rho_lab - rho_w) * g_top)
            N = D + S * (rho_lab - rho_w) * w
        else:
            N = (D + S * q / g_top) / (1 + rho_w * S * (g0 / g_top - 1))
            w = (q - rho_w * g0 * N) / ((rho_lab - rho_w) * g_top)
        # degree 1, centre-of-mass frame (as flowkernels and fem_1d): no external dipole, so N_1 = 0 and
        # the force balance gives w_1 = q_1 / ((rho_LAB - rho_w) g); h = w - N is independent of the frame
        N[:, 1, :] = 0.0
        w[:, 1, :] = q[:, 1, :] / ((rho_lab - rho_w) * g_top)
        h = w - N
        Gh.append(Y @ cilm_to_components(h, lt))
        hcoef.append(h)
        Ncoef.append(N)
        # geoid in the product's convention: csphase -1 multiplies order m by (-1)^m; radius R -> a scales by (R/a)^(l-1)
        col = np.empty(gl.size)
        for j, (L, M) in enumerate(zip(gl, gm)):
            v = N[1, L, -M] if M < 0 else N[0, L, M]
            col[j] = v * (-1.0) ** abs(M) * (R / a_ref) ** (L - 1) if L <= lt else 0.0
        GN.append(col)
        print(f"node {NODES_KM[i]:6.0f} km  |h| {np.sqrt(np.sum(h ** 2) / (4 * np.pi)):7.1f} m  |N| {np.sqrt(np.sum(N ** 2) / (4 * np.pi)):6.2f} m", flush=True)
    f = OUT / f"core_{a.tomo.replace('.ab', '')}_{a.visc}{'' if a.ocean == 'static' else '_geoidocean'}.npz"
    np.savez(f, G_points=np.array(Gh).T, G_geoid=np.array(GN).T, h_cilm=np.array(hcoef), N_cilm=np.array(Ncoef), nodes_km=NODES_KM,
             geoid_l=gl, geoid_m=gm, tomo=a.tomo, visc=a.visc)
    print("saved", f)


if __name__ == "__main__":
    main()
