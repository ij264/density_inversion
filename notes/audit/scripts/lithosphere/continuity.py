import numpy as np, time
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
t=time.time(); mf=MantleFlow(lmax=12); print('mf',time.time()-t)
k=mf.kernels
for n in ['R','r_lab','rho_lab','g_lab','g0','rho_water']: print(n,float(k[n]))
ref=ReferenceLithosphere.from_mantle(mf)
d=ref.data; LA,LO=np.meshgrid(d['lat'],d['lon'],indexing='ij')
W=np.cos(np.radians(LA)); W/=W.sum(); mean=lambda f: float((W*f).sum())
s=ref.structure(LA,LO)
rho_bot=ref.mantle_density(ref.T - s['bed'],LA,LO)   # what lid uses at its base (below sea level T-bed)
rho_bot_geoid=ref.mantle_density(ref.T,LA,LO)          # what geoid.deflection_sheets uses
rho_lab=float(k['rho_lab'])
print('lid base density: mean',mean(rho_bot),'std',np.sqrt(mean((rho_bot-mean(rho_bot))**2)),'min',rho_bot.min(),'max',rho_bot.max())
jump=rho_lab-rho_bot
print('jump rho_lab - rho_lid_base: mean',mean(jump),'pct 5/50/95',np.percentile(jump,[5,50,95]))
print('geoid uses rho at T below SEA LEVEL instead of T-bed: max diff',np.abs(rho_bot-rho_bot_geoid).max(),'mean abs',mean(np.abs(rho_bot-rho_bot_geoid)))
# PREM check at 105 km
x=(6371e3-105e3)/6371e3; print('PREM LID formula at 105 km', 1e3*(2.6910+0.6924*x))
# mean 110 km knot vs PREM at 110
print('mean WG rho at 110 knot',mean(d['rho_mantle'][-1]), 'PREM 110', 1e3*(2.6910+0.6924*(6261/6371)))
# Implied load if the base jump matters: the plate w lifts the column; mass at base rho_lab vs real rho_bot
h=s['bed']
print('jump*bed (kg/m2) mean abs',mean(np.abs(jump*h)),' -> topography equiv (m) mean abs',mean(np.abs(jump*h))/(rho_lab-1030*0.7))
# mean lid density, crust
print('mean rho_c',mean(s['rho_c']),'mean drho_moho',mean(s['drho_moho']),'min drho_moho',s['drho_moho'].min())
