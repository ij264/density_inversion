import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
mf=MantleFlow(lmax=12); ref=ReferenceLithosphere.from_mantle(mf)
d=ref.data; LA,LO=np.meshgrid(d['lat'],d['lon'],indexing='ij')
W=np.cos(np.radians(LA)); W/=W.sum(); A=lambda m: float((W*m).sum())
s=ref.structure(LA,LO); lab=s['lab']; T=ref.T; bed=s['bed']
print('area LAB(bsl) > T:',A(lab>T),' LAB > T - bed (below lid base):',A(lab>T-bed), ' LAB>150km',A(lab>150e3),' LAB>200km',A(lab>200e3))
print('area LAB < Moho?',A(lab<s['moho']), ' LAB within lid',A((lab>s['moho'])&(lab<=T-bed)))
dl=s['drho_lab']
print('drho_lab: mean',A(dl),' mean where LAB>T-bed',A(dl*(lab>T-bed))/A(lab>T-bed),' |drho_lab|<2 area',A(np.abs(dl)<2),' negative area',A(dl<0))
# share of dLAB "mass" placed: geoid sheet at mean(lab)
print('mean LAB depth used for geoid sheet',A(lab)/1e3,'km; median',np.median(lab)/1e3,'; std',np.sqrt(A((lab-A(lab))**2))/1e3)
# geoid sheet factor error from using mean depth: S_l(r) ~ (r/R)^(l+2): ratio for LAB at 250 km vs 105 km
R=6370e3
for l in [2,10,20,40]:
    print(' l',l,'sheet factor ratio (R-250km)/(R-104.75km):',((R-250e3)/(R-104.75e3))**(l+2), ' (R-60km)/(R-104.75):',((R-60e3)/(R-104.75e3))**(l+2))
