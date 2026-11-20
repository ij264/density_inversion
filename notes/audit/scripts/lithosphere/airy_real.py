"""Airy check with the real reference and no plate (Te=0): uniform Moho deepening by 10 km through the 3c operator and
the ocean-loaded plate; compare with w = (rho_top - rho_c) r / (rho_LAB - rho_w C) pointwise."""
import numpy as np, pyshtools as pysh
from pygeoinf.symmetric_space import sphere
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.plate import OceanLoadedPlate, sample_ocean_function
from density_inversion.forward.mantle import MantleFlow
mf=MantleFlow(lmax=12); k=mf.kernels
g=float(k['g_lab']); R=float(k['R']); rl=float(k['rho_lab']); rw=float(k['rho_water']); T=R-float(k['r_lab'])
L=40
Q=sphere.Sobolev(L,2.0,0.05)
ref=ReferenceLithosphere(g,R,L,T)
op=ref.operator(Q)
plate=OceanLoadedPlate(0.0, rl-rw, g, R, L, rho_water=rw)
P=plate.operator(Q)
r=10e3
x=np.zeros(op.domain.dim); i=ref.index('dMoho'); x[i*Q.dim]=-r*np.sqrt(4*np.pi)
q=op(op.domain.from_components(x)); w=P(q)
wc=Q.to_components(w)
# evaluate on a fine grid
H=sphere.Sobolev(4*L,2.0,0.05,grid='GLQ')
LA,LO=ref._grid(H)
wc_f=np.zeros(H.dim); wc_f[:Q.dim]=wc
wg=H.to_grid_values(H.from_components(wc_f)) if hasattr(H,'to_grid_values') else np.asarray(H.from_components(wc_f).data)
s=ref.structure(LA,LO); C=sample_ocean_function(LA,LO)
expect=(s['drho_moho'])*r/(rl-rw*C)
# interior mask: points whose C and drho_moho are constant within +-5 deg
from scipy.ndimage import uniform_filter
def flat(F,n):
    return (uniform_filter(F,size=n,mode='wrap')==F)
n=int(round(10/(360/LA.shape[1])))
mask=np.ones_like(C,bool)
Cg=C; mask&= np.abs(uniform_filter(Cg,size=n,mode='wrap')-Cg)<1e-9
wl=np.cos(np.radians(LA))
rel=(wg-expect)/expect
print('grid',LA.shape,'interior fraction (area)',(wl*mask).sum()/wl.sum())
print('pointwise rel error interior: median',np.median(np.abs(rel[mask])),'95%',np.percentile(np.abs(rel[mask]),95))
print('area-mean w: model',(wl*wg).sum()/wl.sum(),' expected',(wl*expect).sum()/wl.sum())
# smooth comparison: project expected to degree L and compare
ec=H.to_components(H.from_grid_values(expect))[:Q.dim]
print('spectral rel diff (degree<=L):',np.linalg.norm(wc-ec)/np.linalg.norm(ec))
# land vs ocean means
land=(C==0)&mask; ocean=(C==1)&mask
print('land interior: model mean',np.average(wg[land],weights=wl[land]),' expected',np.average(expect[land],weights=wl[land]))
print('ocean interior: model mean',np.average(wg[ocean],weights=wl[ocean]),' expected',np.average(expect[ocean],weights=wl[ocean]))
print('Airy with WINTERC-G lid-base density instead of PREM rho_LAB (~3340): factor',(rl-rw)/(3340.5-rw),'ocean,',rl/3340.5,'land')
