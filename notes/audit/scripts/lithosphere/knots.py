import numpy as np
from density_inversion.forward.lithosphere import ReferenceLithosphere
ref=ReferenceLithosphere(9.91,6370e3,12,105e3)
d=ref.data; rho=d['rho_mantle']; kn=d['mantle_depths']; moho=d['moho_depth']
for i,z in enumerate([20,36,56]):
    m=moho>z*1e3
    print(f'knot {z} km above Moho in {m.mean():.2%} cells; max |rho_knot - rho_submoho| there: {np.abs(rho[i+1]-rho[0])[m].max():.2f}; partial-cell (moho within 1 km) ',np.abs(rho[i+1]-rho[0])[m & (moho<z*1e3+2e3)].max() if (m & (moho<z*1e3+2e3)).any() else None)
LA,LO=np.meshgrid(d['lat'],d['lon'],indexing='ij')
just=ref.mantle_density(moho+1.0,LA,LO)
print('rho 1 m below Moho minus rho_submoho: max abs',np.abs(just-rho[0]).max())
# monotonic knots?
print('knots non-decreasing:',np.all(np.diff(kn,axis=0)>=0))
# density below 110 km knot -> constant extrapolation; lid base depth max
s=ref.structure(LA,LO); print('max lid base depth bsl (km)',(ref.T-s['bed']).max()/1e3)
# 1 deg averaging of Moho vs knots consistency: knots computed from averaged moho? 
