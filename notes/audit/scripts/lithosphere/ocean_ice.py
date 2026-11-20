import numpy as np
R='/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/newrepo/density-inversion'
C=np.loadtxt(R+'/src/density_inversion/data/ocean_function.csv',delimiter=',')
d=np.load(R+'/src/density_inversion/data/reference_lithosphere.npz')
bed=d['bed_elevation'].astype(float); ice=d['ice_thickness'].astype(float)
LAT=d['lat']; W=np.cos(np.radians(LAT))[:,None]*np.ones((1,360)); W/=W.sum()
A=lambda m: float((W*m).sum())
print('C shape',C.shape,'unique',np.unique(C)[:10], 'ocean area', A(C))
print('area C=1 & bed>0 :',A((C>0.5)&(bed>0)), ' C=0 & bed<0:',A((C<0.5)&(bed<0)))
print('  of which ice>100 (C=0,bed<0):',A((C<0.5)&(bed<0)&(ice>100)),' ice<=100:',A((C<0.5)&(bed<0)&(ice<=100)))
print('C=1 & ice>100:',A((C>0.5)&(ice>100)), ' C=1&ice>100&bed<0',A((C>0.5)&(ice>100)&(bed<0)))
m=(C<0.5)&(bed<0)&(ice<=100)
print('land below sea level w/o ice: bed range',bed[m].min(),np.percentile(bed[m],[5,50]))
m2=(C>0.5)&(bed>0)
print('ocean w/ bed>0: bed pct',np.percentile(bed[m2],[50,95,100]))
# floating ice shelves: C=1 & ice>100
m3=(C>0.5)&(ice>100)
print('n cells C=1 ice>100',m3.sum(),'ice thick median',np.median(ice[m3]),'bed median',np.median(bed[m3]))
# freeboard: ice surface elevation = bed + ice
print('  ice surface elevation median',np.median(bed[m3]+ice[m3]))
# load error per area at floating shelves: model mass 910*ice + (if C=1, water via plate rho_w*(-bed)) vs truth rho_w*(-bed)
err=910*ice[m3]
print('  extra mass if C=1 at shelves (kg/m2) median',np.median(err),' => topography error ~ ', np.median(err)/(3380-1030),'m')
# grounded ice below sea level with C=1?
m4=(C>0.5)&(ice>100)&(bed+ice*910/1030>0)  # grounded (ice heavier than displaced water) 
print('C=1 but grounded ice cells',m4.sum(), A(m4))
m5=(C<0.5)&(ice>100)&(bed<0)&(bed+ice*910/1030<0)
print('C=0 but floating?',m5.sum())
