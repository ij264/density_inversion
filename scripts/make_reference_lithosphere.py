"""Build the reference lithosphere (src/density_inversion/data/reference_lithosphere.npz) from WINTERC-G v5.4.

    python scripts/make_reference_lithosphere.py /path/to/WINTERC-G_ver5.4_distribution

Run once; the output is committed, so the package needs nothing outside the repository. Every field is on the
1 degree grid of the ocean function: rows are cells centred on 89.5N ... 89.5S, columns on 0.5E ... 359.5E.
The 0.5 degree WINTERC-G grids are averaged over the four half-degree cells in each 1 degree cell; the LAB and
its density jump, on WINTERC-G's triangular grid (about 225 km spacing), are interpolated by inverse distance
from the four nearest columns.

Fields (SI units; depths positive down from sea level, elevations positive up):
  bed_elevation   bedrock surface without ice (m)                     -ETOPO2_km_depth_Bed.xyz
  ice_thickness   bedrock depth minus ice-surface depth, >= 0 (m)     ETOPO2_km_depth_Bed - Ice;
                  zero where the ice floats (below)
  floating_ice    1 where the ice floats (ice < 1 m excluded): rho_ice I < rho_w (-bed). A floating shelf and
                  the water under it weigh what the same depth of open water does, so these cells are ocean
                  (audit L5: the cavity had been counted as ice, and the restoring force as air)
  moho_depth      crust-mantle boundary (m)                           Global_Moho_WINTERC-G.xyz
  rho_crust       average crustal density, constant in depth          rho_c_out.xyz
  rho_mantle      mantle density at the knots of mantle_depths below  rho_submoho_out.xyz, rho_{20..400}km_out.xyz
  mantle_depths   the knot depths: Moho, 20, 36, 56, 80, 110, 150, 200, 260, 330, 400 km; a knot above the Moho
                  is moved to the Moho, as WINTERC-G's own layering does (README, layers 4-13); linear between
                  knots. Knots below 110 km let the lid's base be moved deeper (audit fix 6).
  lab_depth       thermal lithosphere-asthenosphere boundary (m)      WINTERC-G_LAB.lis
  drho_lab        density just below the LAB minus just above (kg/m^3), from the 2 km profiles in
                  WINTERC-G_Density.lis: rho(LAB + 1 km) - rho(LAB - 1 km)
"""

import sys
from pathlib import Path

import numpy as np

OUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "density_inversion"
    / "data"
    / "reference_lithosphere.npz"
)
LAT = 89.5 - np.arange(180)
LON = 0.5 + np.arange(360)


def half_degree(path):
    """A 0.5 degree xyz file (lon, lat, value) -> 1 degree cell means."""
    lon, lat, v = np.loadtxt(path, unpack=True)
    i = np.round((89.75 - lat) / 0.5).astype(int)
    j = np.round(np.mod(lon, 360.0) / 0.5).astype(int)
    g = np.full((360, 720), np.nan)
    g[i, j] = v
    if np.isnan(g).any():
        raise ValueError(f"{path} does not fill the grid")
    return g.reshape(180, 2, 360, 2).mean(axis=(1, 3))


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def from_triangles(lon, lat, v, k=4):
    """Inverse-distance interpolation from scattered columns to the 1 degree cell centres."""
    from scipy.spatial import cKDTree

    tree = cKDTree(unit(lat, lon))
    LA, LO = np.meshgrid(LAT, LON, indexing="ij")
    d, idx = tree.query(unit(LA, LO).reshape(-1, 3), k=k)
    w = 1.0 / np.maximum(d, 1e-9) ** 2
    return (np.sum(w * v[idx], axis=1) / np.sum(w, axis=1)).reshape(LA.shape)


def main(src):
    src = Path(src)
    km = 1e3
    # the ETOPO2 "depth" files hold depth below sea level, positive down (the README's sign is the other way,
    # but the values are not: 4.2 at the North Pole is the Arctic Ocean's depth)
    bed_depth = half_degree(src / "ETOPO2_km_depth_Bed.xyz") * km
    ice_depth = half_degree(src / "ETOPO2_km_depth_Ice.xyz") * km
    bed = -bed_depth
    ice = np.maximum(bed_depth - ice_depth, 0.0)
    floating = (ice > 1.0) & (bed < 0) & (910.0 * ice < 1030.0 * (-bed))
    ice = np.where(floating, 0.0, ice)
    moho = half_degree(src / "Global_Moho_WINTERC-G.xyz") * km
    rho_c = half_degree(src / "rho_c_out.xyz")
    names = ["submoho"] + [
        f"{d}km" for d in (20, 36, 56, 80, 110, 150, 200, 260, 330, 400)
    ]
    rho_m = np.stack([half_degree(src / f"rho_{n}_out.xyz") for n in names])
    knots = np.array([0.0, 20, 36, 56, 80, 110, 150, 200, 260, 330, 400]) * km
    depths = np.maximum(knots[:, None, None], moho[None])
    depths[0] = moho

    lab = np.loadtxt(src / "WINTERC-G_LAB.lis")
    col, lab_lon, lab_lat, lab_km = lab.T
    dens = np.loadtxt(src / "WINTERC-G_Density.lis", usecols=(0, 3, 4))
    jump = np.empty(col.size)
    for n, (c, z) in enumerate(zip(col, lab_km)):
        p = dens[dens[:, 0] == c]
        below = np.interp(
            z + 1.0, -p[:, 1], p[:, 2]
        )  # depths in the file are negative, listed downwards
        above = np.interp(z - 1.0, -p[:, 1], p[:, 2])
        jump[n] = below - above
    lab_depth = from_triangles(lab_lon, lab_lat, lab_km * km)
    drho_lab = from_triangles(lab_lon, lab_lat, jump)

    np.savez_compressed(
        OUT,
        lat=LAT,
        lon=LON,
        bed_elevation=bed.astype(np.float32),
        ice_thickness=ice.astype(np.float32),
        floating_ice=floating.astype(np.float32),
        moho_depth=moho.astype(np.float32),
        rho_crust=rho_c.astype(np.float32),
        rho_mantle=rho_m.astype(np.float32),
        mantle_depths=depths.astype(np.float32),
        lab_depth=lab_depth.astype(np.float32),
        drho_lab=drho_lab.astype(np.float32),
        source=np.array(
            "WINTERC-G v5.4 (Fullea et al. 2021); see scripts/make_reference_lithosphere.py"
        ),
    )
    w = np.cos(np.radians(LAT))[:, None] * np.ones((1, 360))
    mean = lambda f: float(np.sum(w * f) / np.sum(w))
    print(f"wrote {OUT}")
    print(
        f"area means: Moho {mean(moho) / km:.1f} km, crust {mean(rho_c):.0f} kg/m^3, LAB {mean(lab_depth) / km:.1f} km, "
        f"LAB jump {mean(drho_lab):.1f} kg/m^3 (range {drho_lab.min():.1f} to {drho_lab.max():.1f}), "
        f"ice {ice.max() / km:.2f} km max, floating ice {mean(floating) * 100:.2f} % of the area, bed elevation {bed.min() / km:.1f} to {bed.max() / km:.1f} km"
    )


if __name__ == "__main__":
    main(sys.argv[1])
