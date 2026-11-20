"""Which 1-degree regridding of the 0.5-degree WINTERC-G/ETOPO2 grid (lon nodes 0,0.5,..) best matches the
package's ocean function? Compare the script's pairing (nodes 0 and 0.5 -> cell [0,1]) with a centred
stencil (0.25*n0 + 0.5*n0.5 + 0.25*n1) and with pairing (0.5,1.0)."""
import numpy as np
R='/tmp/claude-0/-home-claude/59c1e02c-0bf7-57c2-b134-24fac4332cfa/scratchpad/newrepo/density-inversion'
F='/mnt/user-data/uploads/dev/data/WINTERC-G_ver5.4_distribution/ETOPO2_km_depth_Ice.xyz'
lon,lat,v=np.loadtxt(F,unpack=True)
i=np.round((89.75-lat)/0.5).astype(int); j=np.round(np.mod(lon,360)/0.5).astype(int)
g=np.full((360,720),np.nan); g[i,j]=v
glat=g.reshape(180,2,720).mean(axis=1)            # lat pairs are fine (89.75, 89.25 -> 89.5)
script=glat.reshape(180,360,2).mean(axis=2)        # nodes (0, 0.5) -> cell [0,1]
centred=0.25*glat[:,0::2]+0.5*glat[:,1::2]+0.25*np.roll(glat[:,0::2],-1,axis=1)
later=0.5*(glat[:,1::2]+np.roll(glat[:,0::2],-1,axis=1))   # nodes (0.5, 1)
C=np.loadtxt(R+'/src/density_inversion/data/ocean_function.csv',delimiter=',')
LAT=89.5-np.arange(180); W=np.cos(np.radians(LAT))[:,None]*np.ones((1,360)); W/=W.sum()
pol=(np.abs(LAT)[:,None]<60)&np.ones((1,360),bool)
for name,f in [('script (0,0.5)',script),('centred (.25,.5,.25)',centred),('pair (0.5,1)',later)]:
    sea=(f>0).astype(float)  # depth > 0 -> below sea level (ice surface, so ice sheets count as land)
    mis=np.abs(sea-C)
    print(f'{name:22s}: area-weighted |sea(f) - C| over |lat|<60: {(W*mis*pol).sum()/(W*pol).sum():.4f}; correlation of C with -depth {np.corrcoef(C[pol],f[pol])[0,1]:.4f}')
# best shift by cross-correlating along longitude at 0.5-degree steps (half-degree resolution)
