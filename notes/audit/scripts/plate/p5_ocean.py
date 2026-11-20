"""Audit step 2: ocean function vs reference-lithosphere bed elevation and ice; density at the LAB."""
import numpy as np
from density_inversion.forward.plate import ocean_function, sample_ocean_function
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere

d = np.load("src/density_inversion/data/reference_lithosphere.npz")
lat_r, lon_r = d["lat"], d["lon"]
bed, ice = d["bed_elevation"].astype(float), d["ice_thickness"].astype(float)
print("cells with 0<ice<1 m (interpolation noise):", np.mean((ice>0)&(ice<1)))
ice = np.where(ice > 1.0, ice, 0.0)
LA, LO = np.meshgrid(lat_r, lon_r, indexing="ij")
C = sample_ocean_function(LA, LO)  # same cell centres
w = np.cos(np.radians(LA)); A = lambda m: float(np.sum(w * m) / np.sum(w))
print("area-weighted ocean fraction  C mean:", A(C), " bed<0:", A(bed < 0), " bed<0 & no ice:", A((bed < 0) & (ice == 0)))
print("fractional C cells (0<C<1) area:", A((C > 0) & (C < 1)))
wet = C >= 0.5
print("disagreement C>=0.5 vs bed<0 (area):", A(wet != (bed < 0)))
print("   C>=0.5 but bed>=0:", A(wet & (bed >= 0)), "  C<0.5 but bed<0:", A(~wet & (bed < 0)))
iceb = (ice > 0) & (bed < 0)
print("ice-covered with bed<0: area", A(iceb), " mean C there", float(np.sum(w*C*iceb)/np.sum(w*iceb)))
print("   of which C=0 (treated as land/air):", A(iceb & (C == 0)), "  C>0:", A(iceb & (C > 0)))
grounded_float = None
# floating ice test: ice thickness * 917/1030 > -bed  -> afloat (ice shelves)
afloat = (ice > 0) & (bed < 0) & (ice * 917 / 1030 < -bed)
print("   ice afloat (shelves) area:", A(afloat), " mean C:", float(np.sum(w*C*afloat)/max(np.sum(w*afloat),1e-30)))
for nm, (la0, la1, lo0, lo1) in {"Caspian": (37, 47, 46, 55), "Dead Sea/Jordan": (31, 32, 35, 36),
                                 "Great Lakes": (41, 49, 267, 284), "Black Sea": (41, 46, 28, 41),
                                 "Antarctica interior": (-90, -70, 0, 360), "Greenland": (60, 83, 287, 340)}.items():
    m = (LA >= la0) & (LA <= la1) & (LO >= lo0) & (LO <= lo1)
    print(f"  {nm:20s} mean C {np.mean(C[m]):.2f}; frac bed<0 {np.mean(bed[m]<0):.2f}; frac ice>0 {np.mean(ice[m]>0):.2f}; min bed {bed[m].min():.0f}")
# orientation: correlation of C with bed<0 under both lat orders
_, _, Craw = ocean_function()
print("corr(Craw rows N->S, bed<0 S->N flipped):", np.corrcoef(Craw.ravel(), (bed[::-1] < 0).ravel())[0, 1],
      " unflipped:", np.corrcoef(Craw.ravel(), (bed < 0).ravel())[0, 1])

mf = MantleFlow(lmax=50)
ref = ReferenceLithosphere.from_mantle(mf)
rho_bot = ref.mantle_density(ref.T, LA, LO)
rho_lab = float(mf.kernels["rho_lab"])
print(f"rho_LAB (PREM, 105 km) = {rho_lab:.1f}; WINTERC-G density at T={ref.T/1e3:.0f} km: mean {A(rho_bot):.1f}, min {rho_bot.min():.1f}, max {rho_bot.max():.1f}")
print("   restoring-term relative change if WINTERC-G mean used: sea", (A(rho_bot)-rho_lab)/(rho_lab-1030), " land", (A(rho_bot)-rho_lab)/rho_lab)
