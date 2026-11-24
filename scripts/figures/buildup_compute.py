"""Build-up of the forward problem, stage by stage: each stage's contribution to topography and the geoid.

Written by Claude, 24 Nov 2026, for the meeting with David. It does not change the package: David's lithosphere
(each column ends at its own LAB, its mass counted in excess of the asthenosphere, rho_LAB; static ocean) is
built here by subclassing ReferenceLithosphere and Geoid, until task B4 puts it in the package.

Stages (one mantle density throughout: the tomography scaled by the R(z) recovered from residual topography):
  1  core model: no plate, water everywhere, static ocean, moved mass at the surface   (core operator file)
  2  elastic plate on land and sea (Te = 40 km), same mantle load; moved mass (rho_LAB - rho_w C) w at the surface
  3  reference lithosphere: q_ref = -g (M - K0), M the excess mass to the LAB; geoid = structure + lift
     and the mantle term with the moved mass distributed through the columns (sea-level frame)
  4  sea level: the reference topography with K = <M> minus that with K0
  5  lithospheric perturbations, an example: a 10 km crustal root under Tibet, a depleted lid under the
     Canadian Shield (-30, -15, 0 kg/m^3 at the three lid nodes)
Output: outputs/figures_buildup/buildup.npz, components (pygeoinf order: orthonormal, csphase +1) at lmax 50.

Asthenosphere (--asth): 'const' counts the excess over a constant rho_LAB (PREM at the flow top), as the model was
first written down; '1d' counts it over the asthenosphere the column displaces at each depth, rho_a(z), WINTERC-G's own
1D density (WINTERC_T_rho_1D.z), which rises with pressure by ~0.9 kg/m^3 per km below 100 km. With 'const', deep
cratonic roots count as heavy because they are compressed, not because they are dense. In both, the plate and the
base of the lift keep rho_LAB (a few per cent of the restoring force; see the notes).

Usage: PYTHONPATH=src ~/pgenv/bin/python scripts/figures/buildup_compute.py [--asth {const,1d}]
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "single_scaling"))
from build_operator import viscosity  # noqa: E402
from density_inversion.forward.examples import cilm_to_components  # noqa: E402
from density_inversion.forward.geoid import Geoid  # noqa: E402
from density_inversion.forward.lithosphere import ReferenceLithosphere  # noqa: E402
from density_inversion.forward.mantle import MantleFlow  # noqa: E402
from density_inversion.forward.plate import OceanLoadedPlate  # noqa: E402

OUT = ROOT / "outputs" / "figures_buildup"
CORE = ROOT / "outputs" / "single_scaling" / "core_sl2013sv.s40rts.31.m_jump30.npz"
POST = ROOT / "outputs" / "single_scaling" / "posterior_core_sl2013sv.s40rts.31.m_jump30_topo.npz"
LMAX, TE = 50, 40e3
WINTERC_1D = ROOT.parent / "data" / "WINTERC-G_ver5.4_distribution" / "WINTERC_T_rho_1D.z"


class Asthenosphere:
    """rho_a(z) (kg/m^3) at depth z below the column's surface, and its integral F(z) = int_0^z rho_a, exact for the
    piecewise-linear profile. 'const' is rho_LAB everywhere."""

    def __init__(self, kind, rho_lab):
        if kind == "const":
            self.z, self.rho = np.array([0.0, 1000e3]), np.array([rho_lab, rho_lab])
        else:
            t = np.loadtxt(WINTERC_1D)
            z, rho = -t[:, 0] * 1e3, t[:, 2]
            keep = z >= 0
            self.z, self.rho = z[keep], rho[keep]
            if self.z[0] > 0:
                self.z, self.rho = np.r_[0.0, self.z], np.r_[self.rho[0], self.rho]
        seg = np.diff(self.z) * 0.5 * (self.rho[1:] + self.rho[:-1])
        self.F_knots = np.r_[0.0, np.cumsum(seg)]

    def __call__(self, z):
        return np.interp(z, self.z, self.rho)

    def integral(self, z):
        z = np.asarray(z, float)
        i = np.clip(np.searchsorted(self.z, z) - 1, 0, self.z.size - 2)
        dz = z - self.z[i]
        return self.F_knots[i] + dz * (self.rho[i] + 0.5 * dz * (self.rho[i + 1] - self.rho[i]) / (self.z[i + 1] - self.z[i]))


class LabColumn(ReferenceLithosphere):
    """David's column: bedrock to the LAB (WINTERC-G's lab_depth, taken as below sea level like its density
    tables); its mass is counted in excess of asthenosphere of density rho_LAB, so columns of different length
    compare directly and nothing below the LAB enters."""

    def __init__(self, *a, rho_lab, asth, **kw):
        super().__init__(*a, **kw)
        self.rho_lab = float(rho_lab)
        self.asth = asth

    def structure(self, lat, lon):
        s = super().structure(lat, lon)  # t_l there is T - t_c; replaced below
        from density_inversion.forward.lithosphere import gll

        lab = s["lab"]
        t_l = lab - s["moho"]
        if np.any(t_l <= 0):
            raise ValueError("Moho below the LAB somewhere")
        x, w = gll(self.lid_nodes)
        s.update(t_l=t_l, node_weights=[t_l / 2 * wn for wn in w],
                 node_depths=[s["moho"] + t_l / 2 * (1 + xn) for xn in x])
        return s

    def lid_intervals(self, lat, lon):
        s = self.structure(lat, lon)
        top, bottom = s["moho"], s["lab"]
        knots = self.sample("mantle_depths", lat, lon)
        edges = [top] + [np.clip(k, top, bottom) for k in knots] + [bottom]
        return [(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def column_mass(self, lat, lon, n_quad=None):
        """Excess mass per unit area over the asthenosphere it displaces:
        rho_i t_i + int_0^{z_L} (rho - rho_a(z)) dz, z below the column's own surface."""
        s = self.structure(lat, lon)
        lid = np.zeros_like(s["t_l"])
        for a, b in self.lid_intervals(lat, lon):
            lid += (b - a) * 0.5 * (self.mantle_density(a, lat, lon) + self.mantle_density(b, lat, lon))
        ice = self.RHO_ICE * self.sample("ice_thickness", lat, lon)
        z_base = s["lab"] + s["bed"]  # the LAB below the column's own surface
        return s["rho_c"] * s["t_c"] + lid + ice - self.asth.integral(z_base)


class LabGeoid(Geoid):
    """The lift sheets end at the LAB; the reference structure's masses are excesses over rho_LAB."""

    def deflection_sheets(self, LA, LO, frame="sealevel"):
        ref = self.ref
        s = ref.structure(LA, LO)
        C = np.asarray(self.ocean(LA, LO), float)
        ice = ref.sample("ice_thickness", LA, LO)
        a = (ice > self.ICE_MIN).astype(float)
        above = a * ref.RHO_ICE + (1 - a) * self.rho_water * C
        out = [(-s["bed"] - ice, a * (ref.RHO_ICE - self.rho_water * C)),
               (-s["bed"], s["rho_c"] - above),
               (s["moho"], ref.mantle_density(s["moho"], LA, LO) - s["rho_c"])]
        for top, bot in ref.lid_intervals(LA, LO):
            out.append((0.5 * (top + bot), ref.mantle_density(bot, LA, LO) - ref.mantle_density(top, LA, LO)))
        out.append((s["lab"], self.rho_lab - ref.mantle_density(s["lab"], LA, LO)))
        if frame == "column":
            out = [(d + s["bed"], c) for d, c in out]
        return out

    def reference_layers(self, LA, LO, n_crust=8, n_piece=6, n_ice=2):
        ref, ra = self.ref, self.ref.asth
        s = ref.structure(LA, LO)
        ice = ref.sample("ice_thickness", LA, LO)
        out = []
        for n, top, bot, rho in ((n_ice, -ice, 0.0 * ice, lambda z: ref.RHO_ICE + 0.0 * z),
                                 (n_crust, 0.0 * ice, s["t_c"], lambda z: s["rho_c"] - ra(z))):
            x, w = np.polynomial.legendre.leggauss(n)
            for xq, wq in zip(x, w):
                z = top + (bot - top) * (1 + xq) / 2
                out.append((z, wq * (bot - top) / 2 * rho(z)))
        x, w = np.polynomial.legendre.leggauss(n_piece)
        for top, bot in ref.lid_intervals(LA, LO):
            for xq, wq in zip(x, w):
                z = top + (bot - top) * (1 + xq) / 2
                out.append((z + s["bed"], wq * (bot - top) / 2 * (ref.mantle_density(z, LA, LO) - ra(z + s["bed"]))))
        return out


def cap(LA, LO, lat0, lon0, radius_deg):
    """A smooth cap: cos^2 taper from 1 at the centre to 0 at radius_deg."""
    la, lo, la0, lo0 = map(np.radians, (LA, LO, lat0, lon0))
    d = np.degrees(np.arccos(np.clip(np.sin(la) * np.sin(la0) + np.cos(la) * np.cos(la0) * np.cos(lo - lo0), -1, 1)))
    return np.where(d < radius_deg, np.cos(0.5 * np.pi * d / radius_deg) ** 2, 0.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asth", choices=("const", "1d"), default="1d")
    kind = ap.parse_args().asth
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mf = MantleFlow(viscosity=viscosity("jump30"), lab_depth=105e3, lmax=LMAX)
    k = mf.kernels
    R, g0, g, rho_lab, rho_w = (float(k[x]) for x in ("R", "g0", "g_lab", "rho_lab", "rho_water"))
    Q = mf.load_space()
    ref = LabColumn(g, R, LMAX, R - float(k["r_lab"]), rho_lab=rho_lab, asth=Asthenosphere(kind, rho_lab))
    geo = LabGeoid(mf, ref, ocean=ref.ocean)
    plate = OceanLoadedPlate.from_mantle(mf, Te=TE, ocean=ref.ocean)
    H = ref._fine(Q)
    LA, LO = ref._grid(H)
    down = H.degree_transfer_operator(Q)
    a = lambda op, c: op.codomain.to_components(op(op.domain.from_components(c)))
    P_H = plate.operator(Q, W=H)
    C = np.asarray(ref.ocean(LA, LO), float)
    print(f"setup {time.time() - t0:.1f}s", flush=True)

    # ---- mantle: load q and direct geoid D from the core operator (static ocean), for the scaled tomography
    core, R_z = np.load(CORE), np.load(POST)["R_mean"]
    lt = core["h_cilm"].shape[-1] - 1
    h1c = np.tensordot(R_z, core["h_cilm"], axes=(0, 0))
    N1c = np.tensordot(R_z, core["N_cilm"], axes=(0, 0))
    pad = lambda c: np.pad(c, ((0, 0), (0, LMAX - lt), (0, LMAX - lt)))
    h1, N1 = cilm_to_components(pad(h1c), LMAX), cilm_to_components(pad(N1c), LMAX)
    w1 = h1 + N1
    deg = np.asarray(Q.degrees, float)
    S_b = np.where(deg >= 2, 4 * np.pi * float(mf.shell.constants.G) * R / ((2 * deg + 1) * g0), 0.0)
    q_m = (rho_lab - rho_w) * g * w1
    D_m = N1 - S_b * (rho_lab - rho_w) * w1
    D_m[deg < 2] = 0.0

    # ---- stage 2: plate on land and sea; moved mass at the surface
    w2H = a(P_H, q_m)
    w2 = a(down, w2H)
    surf = geo.sheets(Q, [(0.0 * LA, rho_lab - rho_w * C)], W=H)
    N2 = D_m + a(surf, w2H)
    h2 = w2 - N2
    print(f"stage 2 {time.time() - t0:.1f}s", flush=True)

    # ---- stage 3: the reference lithosphere, and the mantle with the moved mass through the columns
    G_w = geo.deflection_operator(Q, W=H)
    G_w_col = geo.deflection_operator(Q, W=H, frame="column")
    N3m = D_m + a(G_w, w2H)
    h3m = w2 - N3m
    q_ref = ref.reference_load(Q)  # -g (M - <M>)
    N_struct = Q.to_components(geo.sheets(Q, geo.reference_layers(LA, LO), W=H)(H.from_components(
        H.to_components(H.from_grid_values(np.ones(tuple(H.grid_shape)))))))
    print(f"structure geoid {time.time() - t0:.1f}s", flush=True)

    def reference(K):
        q = q_ref.copy()
        q[0] += g * K * np.sqrt(4 * np.pi)
        wH = a(P_H, q)
        N_lift = a(G_w_col, wH)
        N = N_struct + N_lift
        return a(down, wH) - N, N, N_lift, wH

    h0, *_ = reference(0.0)
    u = reference(1.0)[0] - h0
    up = Q.degree_transfer_operator(H)
    grid = lambda comp: np.asarray(H.grid_values(up(Q.from_components(comp))))
    bed = ref.field(Q, "bed_elevation")
    _, wlat = np.polynomial.legendre.leggauss(LA.shape[0])
    wq = np.broadcast_to(wlat[:, None], LA.shape)
    gu, gr = grid(u), grid(bed - h0)
    K0 = float(np.sum(wq * gu * gr) / np.sum(wq * gu * gu))
    h_ref, N_ref, N_lift, w_refH = reference(K0)
    print(f"stage 3 {time.time() - t0:.1f}s  K0 - <M> = {K0:.3e} kg/m^2", flush=True)

    # ---- stage 4: K = <M> instead of K0
    h_meanM = h0

    # ---- stage 5: lithospheric perturbations (example)
    A_L = ref.operator(Q)
    G_L = geo.lithosphere_operator(Q)
    labels = [s[0] for s in ref.summands()]
    fields = {lab_: np.zeros(LA.shape) for lab_ in labels}
    fields["dMoho"] = -10e3 * cap(LA, LO, 33.0, 88.0, 8.0)  # Moho 10 km deeper under Tibet (upward is positive)
    shield = cap(LA, LO, 60.0, -95.0, 15.0)
    for n, v in enumerate((-30.0, -15.0, 0.0)):
        fields[f"drho lid node {n}"] = v * shield
    xL = np.concatenate([a(down, H.to_components(H.from_grid_values(fields[lab_]))) for lab_ in labels])
    q_L = a(A_L, xL)
    wLH = a(P_H, q_L)
    N_L = a(G_L, xL) + a(G_w, wLH)
    h_L = a(down, wLH) - N_L
    print(f"stage 5 {time.time() - t0:.1f}s", flush=True)

    # ---- checks: mass balance of the reference (sum of lift contrasts = rho_LAB - rho_w C), Airy limit
    sheets = geo.deflection_sheets(LA, LO)
    tot = sum(c for _, c in sheets)
    print("max |sum of lift contrasts - (rho_LAB - rho_w C)| =", float(np.max(np.abs(tot - (rho_lab - rho_w * C)))))
    M = ref.column_mass(LA, LO)
    airy = (K0 + np.sum(wq * M) / np.sum(wq) - M) / (rho_lab - rho_w * C)  # local isostasy with the same K0
    rms = lambda f: float(np.sqrt(np.sum(wq * f ** 2) / np.sum(wq)))
    airy_Q = grid(a(down, H.to_components(H.from_grid_values(airy))))
    print("rms (area-weighted): bed %.0f m, h_ref %.0f m, Airy %.0f m, h_ref - Airy %.0f m, h_ref - bed %.0f m" % (
        rms(grid(bed)), rms(grid(h_ref)), rms(airy_Q), rms(grid(h_ref) - airy_Q), rms(grid(h_ref - bed))))

    np.savez(OUT / f"buildup_{kind}.npz", asth=kind, lmax=LMAX, h1=h1, N1=N1, h2=h2, N2=N2, w1=w1, w2=w2, h3m=h3m, N3m=N3m,
             h_ref=h_ref, N_ref=N_ref, N_lift=N_lift, N_struct=N_struct, h_meanM=h_meanM, K0=K0, bed=bed,
             h_L=h_L, N_L=N_L, R_z=R_z, nodes_km=core["nodes_km"],
             lab=a(down, H.to_components(H.from_grid_values(ref.sample("lab_depth", LA, LO)))),
             C=a(down, H.to_components(H.from_grid_values(C))))
    print(f"saved {OUT / f'buildup_{kind}.npz'}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
