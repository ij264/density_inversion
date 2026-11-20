import numpy as np
R='/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/newrepo/density-inversion'
F='/mnt/user-data/uploads/dev/data/WINTERC-G_ver5.4_distribution/ETOPO2_km_depth_Ice.xyz'
lon,lat,v=np.loadtxt(F,unpack=True)
print('lon unique',np.unique(lon)[:4],np.unique(lon)[-3:],len(np.unique(lon)))
print('lat unique',np.unique(lat)[:3],np.unique(lat)[-3:],len(np.unique(lat)))
def at(la,lo):
    k=np.argmin((lat-la)**2+((lon-lo+180)%360-180)**2); return lon[k],lat[k],v[k]
for n,(la,lo) in {'Everest':(28,87),'Tibet':(32.5,88.5),'Pacific':(-10.5,210.5),'Dome A':(-80.5,77.5),'S Pole':(-89.75,0),'Greenland summit':(72.5,321.5),'Dead Sea':(31.5,35.5),'Mariana':(11.5,142.5)}.items():
    print(n,at(la,lo))
i=np.round((89.75-lat)/0.5).astype(int); j=np.round(np.mod(lon,360)/0.5).astype(int)
g=np.full((360,720),np.nan); g[i,j]=v
ice_depth=g.reshape(180,2,360,2).mean(axis=(1,3))*1e3
d=np.load(R+'/src/density_inversion/data/reference_lithosphere.npz')
bed=d['bed_elevation'].astype(float); ice=d['ice_thickness'].astype(float)
bed_depth=-bed
diff=bed_depth-ice_depth
print('check ice reproduction max err',np.abs(np.maximum(diff,0)-ice).max())
W=np.cos(np.radians(d['lat']))[:,None]*np.ones((1,360)); W/=W.sum()
oc=bed<0
print('ocean cells: diff stats (m): mean',diff[oc].mean(),'pct 5,50,95',np.percentile(diff[oc],[5,50,95]))
print('ocean area with ice>0 :',(W*(oc&(ice>0))).sum(),' with ice>50 m:',(W*(oc&(ice>50))).sum(), 'ice>200', (W*(oc&(ice>200))).sum())
print('land cells away from poles |lat|<60: diff pct',np.percentile(diff[(~oc)&(np.abs(d['lat'])[:,None]<60)],[1,5,50,95,99]))
# ice mass from spurious ocean ice
spur=np.where(oc&(np.abs(d['lat'])[:,None]<60),ice,0)
print('spurious ice |lat|<60 ocean: area-mean thickness',(W*spur).sum(),'m; max',spur.max(),'; equiv mass 910*: ',910*spur.max())
# true ice: Antarctica & Greenland
print('ice area total', (W*(ice>0)).sum(), ' ice>100m', (W*(ice>100)).sum())
# sign check: ocean ice-surface depth vs bed depth: if ice file were water surface 0 at sea, ice_depth would be ~0
print('ocean ice_depth mean (m)',ice_depth[oc].mean())
