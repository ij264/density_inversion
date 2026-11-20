"""Convergence of the 3c multiplication operators and of q_ref with the quadrature degree quad_lmax."""
import numpy as np, sys, time
from pygeoinf.symmetric_space import sphere
from density_inversion.forward.lithosphere import ReferenceLithosphere
g,R,T=9.9135,6370e3,105e3
L=int(sys.argv[1]) if len(sys.argv)>1 else 20
Q=sphere.Sobolev(L,2.0,0.05)
deg=np.asarray(Q.degrees)
rng=np.random.default_rng(1)
u_rand=rng.standard_normal(Q.dim)*(1+deg*(deg+1)*0.05**2)**-1.0   # smooth-ish random field
u_one=np.zeros(Q.dim); u_one[0]=np.sqrt(4*np.pi)
facs=[int(a) for a in sys.argv[2].split(",")] if len(sys.argv)>2 else [2,3,4,6,10]
res={}
for f in facs:
    t=time.time()
    ref=ReferenceLithosphere(g,R,L,T,quad_lmax=f*L)
    op=ref.operator(Q)
    n=len(ref.summands())
    outs=[]
    for i in range(n):
        for u in (u_one,u_rand):
            x=np.zeros(op.domain.dim); x[i*Q.dim:(i+1)*Q.dim]=u
            outs.append(Q.to_components(op(op.domain.from_components(x))))
    qref=ref.reference_load(Q)
    res[f]=(outs,qref,[s[0] for s in ref.summands()])
    print(f'N={f}L done {time.time()-t:.1f}s',flush=True)
best=res[facs[-1]]
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
def relhi(a,b):
    m=deg>=L//2; return np.linalg.norm((a-b)[m])/np.linalg.norm(b[m])
labels=best[2]
print(f'L={L}: relative error vs N={facs[-1]}L  [all degrees | degrees >= L/2]')
for f in facs[:-1]:
    outs,qref,_=res[f]
    line=[]
    for i,lab in enumerate(labels):
        e1=rel(outs[2*i],best[0][2*i]); e2=rel(outs[2*i+1],best[0][2*i+1]); e3=relhi(outs[2*i+1],best[0][2*i+1])
        line.append(f'{lab}: u=1 {e1:.1e}, rand {e2:.1e}|{e3:.1e}')
    print(f' N={f}L: q_ref {rel(qref,best[1]):.1e}|{relhi(qref,best[1]):.1e}; '+'; '.join(line))
