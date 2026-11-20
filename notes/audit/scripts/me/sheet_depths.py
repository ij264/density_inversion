import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
R=6370000.; G=6.67e-11; g0=9.820015821809946
A=2000; s=400e3/R
x,wq=leggauss(400); th=np.arccos(x); f=A*np.exp(-th**2/(2*s**2))
L=50; ls=np.arange(L+1)
fl=np.array([(2*l+1)/2*np.sum(wq*f*eval_legendre(l,x)) for l in ls])  # f = sum fl P_l
rc,rL,rw,rlab=2938.42,3352.344,1030.,3370.361
rbase_true=3335.4674
w_l=-(rL-rc)/(rlab-rw)*fl
def N(sig_l,depth):
    r=R-depth; S=4*np.pi*G*r*(r/R)**(ls+1)/((2*ls+1)*g0)
    m=ls>=2; return np.sum((S*sig_l)[m])   # value at theta=0, P_l(1)=1
def total(dm,ds,dmoho_lift,dmid,dbase,rbase,label):
    moho=N((rL-rc)*fl,dm)
    lift=N((rc-rw)*w_l,ds)+N((rL-rc)*w_l,dmoho_lift)+N((rbase-rL)*w_l,dmid)+N((rlab-rbase)*w_l,dbase)
    print(f'{label:45s} moho {moho:8.3f} lift {lift:8.3f} total {moho+lift:7.3f}')
total(14461.3,4960.6,14461.3,0.5*(14461.3+109960.6),109960.6,rbase_true,'true depths (lid lumped mid)')
total(22871.9,0.0,22871.9,0.5*(22871.9+105000),105000.,3331.609,'package depths (mean Moho, surface at R)')
total(14461.3,0.0,14461.3,0.5*(14461.3+105000),105000.,3331.609,'local Moho, surface sheet at R')
total(14461.3,4960.6,14461.3,0.5*(14461.3+105000),105000.,3331.609,'local Moho, seafloor sheet, base 105')
