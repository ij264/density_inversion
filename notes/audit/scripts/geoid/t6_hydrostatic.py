"""Task 6: size of the hydrostatic-flattening correction at l=2 (and 4) vs the model's degree-2 output.
Orthonormal components (integral Y^2 = 1 on the unit sphere) = 4pi-normalised coefficient * sqrt(4 pi)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel
from density_inversion.forward import examples as ex

R = 6.371e6   # for geodetic coefficients use a = 6378137 m; difference is 0.1 %
a = 6378137.0
on = lambda Cbar: a * Cbar * np.sqrt(4 * np.pi)
C20_obs = -484.1555e-6       # Chambat et al. 2010 (tide/atmosphere corrected)
C20_hyd_ch = -479.06e-6      # Chambat et al. 2010
C20_hyd_nk = -479.73e-6      # Nakiboglu 1982 as quoted by Chambat et al. 2010
J2_grs80, J4_grs80 = 1.08263e-3, -2.37091222e-6
C20_grs80, C40_grs80 = -J2_grs80 / np.sqrt(5), -J4_grs80 / 3
C40_hyd = 2.96e-6 / 3        # J4_hyd = -2.96e-6 (Chambat et al. 2010)
print(f"non-hydrostatic C20 (Chambat): {C20_obs - C20_hyd_ch:.3e} -> orthonormal geoid component {on(C20_obs - C20_hyd_ch):.1f} m"
      f" (pole {a*(C20_obs-C20_hyd_ch)*np.sqrt(5):.1f} m)")
print(f"non-hydrostatic C20 (Nakiboglu): {C20_obs - C20_hyd_nk:.3e} -> {on(C20_obs - C20_hyd_nk):.1f} m;  Chambat-Nakiboglu {on(C20_hyd_nk - C20_hyd_ch):.1f} m")
print(f"GRS80-referenced geoid C20 residual: {C20_obs - C20_grs80:.3e} -> {on(C20_obs - C20_grs80):.2f} m  (=> correction to apply: {on(C20_grs80 - C20_hyd_ch):.1f} m)")
print(f"l=4: GRS80 C40 {C40_grs80:.3e}, hydrostatic C40 {C40_hyd:.3e}: correction {on(C40_grs80 - C40_hyd):.2f} m")
print(f"un-referenced ('full') geoid C20 -> {on(C20_obs):.0f} m")
# observed nonhydrostatic l=2 other coefficients for scale (EGM2008-like values)
print(f"C22=2.4393e-6 -> {on(2.4393e-6):.1f} m, S22=-1.4003e-6 -> {on(-1.4003e-6):.1f} m")

L = 12
mf = MantleFlow(lmax=L); fm = ForwardModel(mf, Te=40e3)
X = fm.X; Rm = float(mf.kernels["R"])
x_m = ex.project(X, ex.power_law(Rm, X.layer_bounds[0][0], X.layer_bounds[-1][1], rms=3.0, lmax=L, seed=0))
p = fm.parts(x_m)
deg = np.asarray(fm.Q.degrees)
for k in ("mantle", "deflection", "reference", "total"):
    v = p["geoid"][k][deg == 2]
    print(f"model l=2 geoid '{k}': components {np.round(v, 1)}  norm {np.linalg.norm(v):.1f} m")
