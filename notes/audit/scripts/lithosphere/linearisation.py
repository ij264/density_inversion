"""First-order column-mass change vs exact, real WINTERC-G columns (density-depth profile fixed, Moho moves through it)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
mf=MantleFlow(lmax=12); ref=ReferenceLithosphere.from_mantle(mf); rl=float(mf.kernels['rho_lab'])
cols={'Tibet':(32.5,88.5),'Altiplano':(-17.5,292.5),'Pacific':(-10.5,210.5),'MAR':(0.5,335.5),'Canadian shield':(60.5,265.5),'E Antarctica':(-80.5,100.5),'Iceland':(64.5,341.5)}
def integ(f,a,b,n=4000):
    z=np.linspace(a,b,n+1); v=f(z); return np.sum(0.5*(v[1:]+v[:-1])*np.diff(z))
print(f"{'column':16s} {'case':26s} {'linear':>10s} {'exact':>10s} {'err kg/m2':>10s} {'rel':>7s} {'dw(m)':>7s}")
worst=0
for name,(la,lo) in cols.items():
    lat,lon=np.array([la]),np.array([lo])
    s=ref.structure(lat,lon); bed=s['bed'][0]; moho=s['moho'][0]; rc=s['rho_c'][0]; dm=s['drho_moho'][0]; tl=s['t_l'][0]
    rho=lambda z_bsl: ref.mantle_density(z_bsl, np.full(z_bsl.shape,la), np.full(z_bsl.shape,lo))
    base=ref.T-bed  # lid base, depth below sea level
    C=1.0 if bed<0 else 0.0
    for du in (-10e3, +10e3):   # dMoho positive up
        lin=dm*du
        if du<0: exact=integ(lambda z: rc-rho(z), moho, moho-du)*-1*-1  # crust replaces lid between moho and moho+10
        else: exact=integ(lambda z: rho(z)-rc, moho-du, moho) if moho-du>-bed else np.nan
        # sign: mass change = +int(rc-rho) for thickening
        if du<0: exact=integ(lambda z: rc-rho(z), moho, moho-du)
        err=lin-exact
        print(f"{name:16s} {'dMoho %+.0f km'%(du/1e3):26s} {lin:10.3e} {exact:10.3e} {err:10.2e} {err/exact:7.1%} {err/(rl-1030*C):7.1f}")
        worst=max(worst,abs(err/exact))
    for dr in (-30.,30.):
        lin=dr*tl; exact=dr*tl
        # combined with Moho thickening 10 km (fixed profile): lid shrinks by 10 km
        lin2=dm*(-10e3)+dr*tl; exact2=integ(lambda z: rc-rho(z), moho, moho+10e3)+dr*(tl-10e3)
        err=lin2-exact2
        print(f"{name:16s} {'lid %+.0f & Moho -10 km'%dr:26s} {lin2:10.3e} {exact2:10.3e} {err:10.2e} {err/exact2:7.1%} {err/(rl-1030*C):7.1f}")
        lin3=dm*(-10e3)+dr*s['t_c'][0]; exact3=integ(lambda z: rc-rho(z), moho, moho+10e3)+dr*(s['t_c'][0]+10e3)
        err=lin3-exact3
        print(f"{name:16s} {'crust %+.0f & Moho -10 km'%dr:26s} {lin3:10.3e} {exact3:10.3e} {err:10.2e} {err/exact3:7.1%} {err/(rl-1030*C):7.1f}")
print('pure lid +-30 alone: first order is exact (linear in drho); error 0')
