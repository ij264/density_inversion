"""Task 4: Geoid.reference() convergence in n_layers; reference geoid (structure + its deflection)
against (a) the linear lifted column with all contrasts at their own depths, (b) the finite-amplitude
lifted column (every layer at its true radius R + h - z). Also the effect of the sheet errors on the
NET geoid after compensation, for the reference and for the mantle term."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.geoid import Geoid
from density_inversion.forward.plate import OceanLoadedPlate
import t45_sheets as T45  # reuses its grid, helpers and h_ref (LMAX from argv default 50)

Q, ref, geo, H, LA, LO, s, deg = T45.Q, T45.ref, T45.geo, T45.H, T45.LA, T45.LO, T45.s, T45.deg
nrm, band, exact_sheets, apply = T45.nrm, T45.band, T45.exact_sheets, T45.apply
R, T = geo.R, ref.T
h_ref = T45.h_ref
_, hg = T45.fields["h_ref"]

print("\n=== reference(): n_layers convergence (l>=2) ===")
Nr = {n: geo.reference(Q, n) for n in (120, 240, 480, 960)}
for n in (120, 240, 480):
    print(f" n={n}: |N(n)-N(960)|/|N(960)| = {nrm(Nr[n]-Nr[960])/nrm(Nr[960]):.2e};  by band {band(Nr[n], Nr[960])}")
Nw = apply(T45.Gw, h_ref)
net = {n: Nr[n] + Nw for n in Nr}
for n in (120, 240, 480):
    print(f" NET N_ref (structure + G_w h_ref): n={n} vs 960: {band(net[n], net[960])}")
print(f" compensation: |N_ref_net|/|N_structure| = {nrm(net[960])/nrm(Nr[960]):.3f}")
print(" rms N_ref_net (m) =", np.sqrt(np.sum(net[960][deg>=2]**2)/(4*np.pi)), " rms structure-only =", np.sqrt(np.sum(Nr[960][deg>=2]**2)/(4*np.pi)))

# column-mass consistency: the reference() columns vs column_mass() used for the load
dz = T / 120
rho_mid = []
M120 = np.zeros(LA.shape)
for z in (np.arange(120) + 0.5) * dz:
    rho = np.where(z < s["t_c"], s["rho_c"], ref.mantle_density(z - s["bed"], LA, LO))
    M120 += rho * dz
Mq = ref.column_mass(LA, LO) - ref.RHO_ICE * ref.sample("ice_thickness", LA, LO)
d = M120 - Mq
print(f" column mass (n=120 midpoint) minus column_mass(): rms {np.sqrt(np.mean(d**2)):.1f} kg/m^2, max {np.abs(d).max():.1f}  (column mass ~{Mq.mean():.3e})")

print("\n=== linear lifted column with own-depth contrasts vs repo, NET reference geoid ===")
N_lin = Nr[960] + exact_sheets(T45.lifted_column(hg, "column"))
print(" repo net vs exact-linear net:", band(net[960], N_lin))

# finite amplitude: every layer at R + h - z, water and mantle infill as signed sheets
n = 480
dz = T / n
sh = []
for z in (np.arange(n) + 0.5) * dz:
    rho = np.where(z < s["t_c"], s["rho_c"], ref.mantle_density(z - s["bed"], LA, LO))
    sh.append((z - hg, rho * dz))
C = np.asarray(geo.ocean(LA, LO), float)
sh.append((-hg / 2, -geo.rho_water * C * hg))                 # water between R+h and R (signed)
sh.append((T - hg / 2, geo.rho_lab * hg))                     # mantle filling below the column
sh.append((-hg, ref.RHO_ICE * ref.sample("ice_thickness", LA, LO)))  # ice on the lifted surface
N_fin = exact_sheets(sh)
print(" repo net vs exact finite-amplitude net:", band(net[960], N_fin))
print(" exact-linear vs exact-finite:", band(N_lin, N_fin))
