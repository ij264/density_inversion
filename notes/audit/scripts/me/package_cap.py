import numpy as np, pyshtools as pysh
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel
from density_inversion.forward import examples as ex
L=50
mf=MantleFlow(lmax=L)
for Te in (0.0, 40e3):
    fm=ForwardModel(mf,Te=Te)
    Q=fm.Q; R=float(mf.kernels['R']); deg=np.asarray(Q.degrees)
    glat,glon=ex.dh_lat_lon(2*(4*L+1))
    GLA,GLO=np.meshgrid(np.radians(glat),np.radians(glon),indexing='ij')
    la,lo=0.0,np.radians(200.)
    cosd=np.sin(GLA)*np.sin(la)+np.cos(GLA)*np.cos(la)*np.cos(GLO-lo)
    grid=2000*np.exp(-0.5*(R*np.arccos(np.clip(cosd,-1,1))/400e3)**2)
    cap=ex.cilm_to_components(pysh.expand.SHExpandDH(grid,norm=4,sampling=2,csphase=1,lmax_calc=L),L)
    ref=fm.ref; xL=np.zeros(fm.lithosphere_space().dim); i=ref.index('dMoho'); xL[i*Q.dim:(i+1)*Q.dim]=cap
    p=fm.parts(None,xL,None)
    def at(c):
        lat,lon,G=ex.to_grid(c,L); j=np.argmin(abs(lat-0)); k=np.argmin(abs(lon-200)); return G[j,k]
    w=p['topography']['lithosphere']
    Nl=fm._apply(fm.G_L,None) if False else None
    NL=fm.Q.to_components(fm.G_L(fm.G_L.domain.from_components(xL)))
    Nw=fm._apply(fm.G_w,w)
    print(f'Te={Te/1e3:.0f} km: w centre {at(w):.2f} m (l>=1), {at(np.where(deg>=0,w,0)):.2f}; N_lith {at(NL):.4f} m, N_defl {at(Nw):.4f} m, total {at(NL+Nw):.4f} m')
