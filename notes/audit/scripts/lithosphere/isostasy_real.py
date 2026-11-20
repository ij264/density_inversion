"""No-plate isostatic elevation of the real WINTERC-G columns (to T below the column's own surface) vs ETOPO bed,
for the code's convention (Moho below sea level) and the alternative (Moho below bedrock surface)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.plate import sample_ocean_function
mf=MantleFlow(lmax=12); k=mf.kernels; rho_lab=float(k['rho_lab']); rw=1030.
ref=ReferenceLithosphere.from_mantle(mf)
d=ref.data; LA,LO=np.meshgrid(d['lat'],d['lon'],indexing='ij')
W=np.cos(np.radians(LA)); W/=W.sum(); mean=lambda f: float((W*f).sum())
C=sample_ocean_function(LA,LO); Cb=(d['bed_elevation']<0).astype(float)
bed=d['bed_elevation'].astype(float)
M=ref.column_mass(LA,LO)
def lid_mass(top_bsl, base_bsl, n=40):
    xg,wg=np.polynomial.legendre.leggauss(n); out=0
    for xq,wq in zip(xg,wg):
        z=top_bsl+(base_bsl-top_bsl)/2*(1+xq); out=out+wq*(base_bsl-top_bsl)/2*ref.mantle_density(z,LA,LO)
    return out
ice=910*d['ice_thickness']
# alternative: Moho depth below bedrock surface -> Moho bsl = moho - bed
moho=d['moho_depth'].astype(float)
M_alt=d['rho_crust']*moho + lid_mass(moho-bed, ref.T-bed) + ice
def fit(M,Cuse,label):
    den=rho_lab-rw*Cuse
    # e*den = K - M ; choose K by least squares vs bed
    K=mean((bed*den+M)/den**2)/mean(1/den**2)
    e=(K-M)/den
    e0=-(M-mean(M))/den
    r=e-bed
    A=np.vstack([e.ravel()*np.sqrt(W).ravel(),np.sqrt(W).ravel()]).T
    slope=np.linalg.lstsq(np.vstack([e.ravel(),np.ones(e.size)]).T*np.sqrt(W).ravel()[:,None],bed.ravel()*np.sqrt(W).ravel(),rcond=None)[0]
    print(f"{label}: K-<M> = {K-mean(M):.3e} kg/m2; rms(e-bed) {np.sqrt(mean(r**2)):.0f} m; corr {np.corrcoef(e.ravel(),bed.ravel())[0,1]:.3f}; "
          f"regress bed = {slope[0]:.3f} e + {slope[1]:.0f}; land rms {np.sqrt(mean(r**2*(Cuse<.5))/mean(Cuse<.5)):.0f}, ocean rms {np.sqrt(mean(r**2*(Cuse>.5))/mean(Cuse>.5)):.0f}")
    print(f"   with K=<M> (code's zero-mean): mean e0 {mean(e0):.0f} m vs mean bed {mean(bed):.0f}; rms(e0-bed) {np.sqrt(mean((e0-bed)**2)):.0f} m")
    return e,e0
e,e0=fit(M,C,'code (Moho bsl), ocean fn C')
fit(M_alt,C,'alt (Moho below bed)')
print('crust thickness difference alt-code = -bed: area-mean abs',mean(np.abs(bed)),'m; land mean',mean(bed*(bed>0))/mean(bed>0),'ocean mean',mean(-bed*(bed<0))/mean(bed<0))
dM=M_alt-M
print('column mass difference alt-code: mean abs',mean(np.abs(dM)),'kg/m2; ocean mean',mean(dM*(bed<0))/mean(bed<0),'land mean',mean(dM*(bed>0))/mean(bed>0))
print('  -> topography difference equiv: ocean',mean(dM*(bed<0))/mean(bed<0)/(rho_lab-rw),'m; land',mean(dM*(bed>0))/mean(bed>0)/rho_lab,'m')
# effect of choosing K=<M> rather than the K that fits sea level: difference in degree>0 pattern
K=mean((bed*(rho_lab-rw*C)+M)/(rho_lab-rw*C)**2)/mean(1/(rho_lab-rw*C)**2)
de=(K-mean(M))/(rho_lab-rw*C)
de=de-mean(de)
print('K choice: K-<M>',K-mean(M),' land-ocean step in w from the constant: ', (K-mean(M))*(1/rho_lab-1/(rho_lab-rw)),'m; rms of non-constant part',np.sqrt(mean(de**2)))
np.savez('/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/audit/lithosphere/isostasy_real.npz',e=e,e0=e0,bed=bed)
