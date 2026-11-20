import numpy as np
R='/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/newrepo/density-inversion'
C=np.loadtxt(R+'/src/density_inversion/data/ocean_function.csv',delimiter=',')
d=np.load(R+'/src/density_inversion/data/reference_lithosphere.npz')
bed=d['bed_elevation'].astype(float); I=d['ice_thickness'].astype(float); lat=d['lat'][:,None]*np.ones((1,360)); lon=np.ones((180,1))*d['lon'][None]
W=np.cos(np.radians(lat)); W/=W.sum(); A=lambda m: float((W*m).sum())
rw,ri,rl=1030.,910.,3370.36
float_=(I>1)&(bed<0)&(ri*I<rw*(-bed))     # ice lighter than the water column it sits in -> cannot be grounded
grounded_sub=(I>1)&(bed<0)&~float_
print('ice cells over sub-sea bed: floating-by-flotation area',A(float_),'n',float_.sum(),' grounded area',A(grounded_sub))
for c,lab in ((0,'C=0'),(1,'C>0')):
    m=float_&((C>0) if c else (C==0))
    if m.sum()==0: continue
    extra = ri*I - (0 if c else rw*(-bed))       # model mass minus hydrostatic (water-equivalent) truth
    den = rl-rw*C
    dw = extra/den
    print(f' floating {lab}: n {m.sum()}, area {A(m):.2e}; extra mass median {np.median(extra[m]):.3e} kg/m2, w error median {np.median(dw[m]):.0f} m, max |.| {np.abs(dw[m]).max():.0f} m')
    if not c:
        # response to perturbations: restoring rho_LAB instead of rho_LAB - rho_w
        print(f'   restoring contrast {rl:.0f} instead of {rl-rw:.0f}: perturbation response too small by {1-(rl-rw)/rl:.0%}')
m=(I>1)&(np.abs(lat)<60)
print('ice >1 m at |lat|<60: n',m.sum(), 'cells', list(zip(lat[m][:8],lon[m][:8],I[m][:8].round(1),bed[m][:8].round(0))))
print('ice > 0 but < 1 m area (float32 noise)',A((I>0)&(I<1)))
