"""Task 4 fix check: reference() with the crust/mantle cell split by volume fraction (so each column's mass
matches column_mass() to O(dz^2)), convergence of the NET reference geoid in n_layers."""
import numpy as np
import t45_sheets as T45
Q, geo, ref, H, LA, LO, s, deg = T45.Q, T45.geo, T45.ref, T45.H, T45.LA, T45.LO, T45.s, T45.deg

def reference_split(n):
    out = np.zeros(Q.dim); dz = ref.T / n
    wl = np.cos(np.radians(LA))
    for k in range(n):
        z0, z1 = k * dz, (k + 1) * dz
        fc = np.clip((s["t_c"] - z0) / dz, 0, 1)              # crust fraction of the cell
        zm = np.maximum(0.5 * (z1 + np.maximum(z0, s["t_c"])), z0)  # midpoint of the mantle part
        rho = fc * s["rho_c"] + (1 - fc) * ref.mantle_density(zm - s["bed"], LA, LO)
        sig = dz * (rho - np.sum(wl * rho) / np.sum(wl))
        out += geo.sheet_factor(Q, geo.R - 0.5 * (z0 + z1)) * T45.comps(sig)
    ice = ref.RHO_ICE * ref.sample("ice_thickness", LA, LO)
    return out + geo.sheet_factor(Q, geo.R) * T45.comps(ice)

Nw = T45.apply(T45.Gw, T45.h_ref)
N = {n: reference_split(n) + Nw for n in (120, 240, 960)}
for n in (120, 240):
    print(f"split-cell reference, net, n={n} vs 960: {T45.band(N[n], N[960])}")
print("repo reference(960) net vs split-cell(960) net:", T45.band(geo.reference(Q, 960) + Nw, N[960]))
