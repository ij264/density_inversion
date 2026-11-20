import numpy as np, sys
R='/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/newrepo/density-inversion'
d=np.load(R+'/src/density_inversion/data/reference_lithosphere.npz')
LAT=d['lat']; w=np.cos(np.radians(LAT))[:,None]*np.ones((1,360)); W=w/w.sum()
mean=lambda f: float((W*f).sum())
for k in d.files:
    a=d[k]
    if a.ndim>=2: print(k,a.shape,a.dtype,np.nanmin(a),np.nanmax(a), mean(a) if a.ndim==2 else '')
    else: print(k,a.shape, a if a.size<5 else (a[:3],a[-3:]))
bed=d['bed_elevation'];ice=d['ice_thickness'];moho=d['moho_depth'];lab=d['lab_depth']
print('frac area LAB>105km',float((W*(lab>105e3)).sum()), 'LAB>110', float((W*(lab>110e3)).sum()))
print('frac area bed<0', float((W*(bed<0)).sum()))
print('ice>0 area',float((W*(ice>0)).sum()), 'ice>0 & bed<0', float((W*((ice>0)&(bed<0))).sum()))
# where is Moho small vs bed
oc=bed<-2000
print('ocean (bed<-2km) moho mean, min, max',moho[oc].mean(),moho[oc].min(),moho[oc].max())
print('ocean moho - (-bed) i.e. crust thickness if moho below sea level: mean',(moho+bed)[oc].mean(), 'min',(moho+bed)[oc].min())
cont=bed>500
print('continents bed>0.5km moho mean',moho[cont].mean(), 'moho+bed mean', (moho+bed)[cont].mean())
# Tibet
i=int(90-31.5); j=int(88.5)
print('Tibet cell lat',LAT[i],'lon',d['lon'][j],'bed',bed[i,j],'moho',moho[i,j],'lab',lab[i,j])
for name,(la,lo) in {'Himalaya/Tibet':(32.5,88.5),'Andes Altiplano':(-17.5,292.5),'Pacific':(-10.5,210.5),'Mid-Atlantic ridge':(0.5,335.5),'Antarctica E':(-80.5,100.5),'Greenland':(72.5,320.5),'Canadian shield':(60.5,265.5),'North pole':(89.5,0.5), 'Iceland':(64.5,341.5)}.items():
    i=int(np.floor(90-la)); j=int(np.floor(lo))
    print(f"{name:20s} bed {bed[i,j]:8.0f} ice {ice[i,j]:6.0f} moho {moho[i,j]:7.0f} t_c {moho[i,j]+bed[i,j]:7.0f} rho_c {d['rho_crust'][i,j]:.0f} lab {lab[i,j]:7.0f} drho_lab {d['drho_lab'][i,j]:.1f} rho_m {d['rho_mantle'][:,i,j].round(0)} knots {d['mantle_depths'][:,i,j].round(0)}")
