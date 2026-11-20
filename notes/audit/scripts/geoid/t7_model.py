"""Task 7/8: assembly checks on ForwardModel (LMAX = 12)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel

LMAX = 12
fm = ForwardModel(MantleFlow(lmax=LMAX))
Q, A = fm.Q, fm.operator
M, D = fm.model_space, fm.data_space
deg = np.asarray(Q.degrees)
rng = np.random.default_rng(11)
comp = lambda sp, v: sp.to_components(v)
apply = fm._apply

# 1. exact linearity of the operator, affine forward
x, y = rng.standard_normal(M.dim), rng.standard_normal(M.dim)
a, b = 2.7, -0.31
lhs = comp(D, A(M.from_components(a * x + b * y)))
rhs = a * comp(D, A(M.from_components(x))) + b * comp(D, A(M.from_components(y)))
print("linearity rel err:", np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
nX, nL = fm.X.dim, fm.lithosphere_space().dim
xm, xL, xs = x[:nX], x[nX:nX + nL], x[nX + nL:]
h, N = fm.forward(xm, xL, xs)
h_ref, N_ref = fm.offset()
d = comp(D, A(M.from_components(x)))
print("forward = operator + offset:", np.abs(h - d[:Q.dim] - h_ref).max(), np.abs(N - d[Q.dim:] - N_ref).max())
h0, N0 = fm.forward()
print("forward(0) == offset:", np.array_equal(h0, h_ref), np.array_equal(N0, N_ref))

# 2. parts sum and the 'reference' entry
p = fm.parts(xm, xL, xs)
print("parts sum h:", np.abs(p["topography"]["total"] - h).max() / np.abs(h).max(),
      " N:", np.abs(p["geoid"]["total"] - N).max() / np.abs(N).max())
Nstruct = fm.geoid.reference(Q)
print("parts['geoid']['reference'] vs geoid.reference(Q):", np.abs(p["geoid"]["reference"] - Nstruct).max() / np.abs(Nstruct).max())
# independent recomputation of the geoid row
Nm = apply(fm.G_m, xm)
qL = Q.to_components(fm.A_L(fm.A_L.domain.from_components(xL)))
NL = Q.to_components(fm.G_L(fm.G_L.domain.from_components(xL)))
hm = apply(fm.P, apply(fm.A_m, xm)); hL = apply(fm.P, qL)
Nind = Nm + NL + apply(fm.G_w, hm + hL) + Nstruct + apply(fm.G_w, h_ref)
print("geoid row recomputed by hand:", np.abs(Nind - N).max() / np.abs(N).max())
hind = hm + hL + xs + h_ref
print("topography row recomputed by hand:", np.abs(hind - h).max() / np.abs(h).max())

# 3. Sobolev adjoint (dot-product) tests
def adj_err(op, seed):
    r = np.random.default_rng(seed)
    u = op.domain.from_components(r.standard_normal(op.domain.dim))
    v = op.codomain.from_components(r.standard_normal(op.codomain.dim))
    l = op.codomain.inner_product(op(u), v)
    return abs(l - op.domain.inner_product(u, op.adjoint(v))) / abs(l)

blocks = dict(P=fm.P, A_m=fm.A_m, A_L=fm.A_L, G_m=fm.G_m, G_L=fm.G_L, G_w=fm.G_w, I_s=fm.I_s,
              PA_m=fm.P @ fm.A_m, GwPA_L=fm.G_w @ fm.P @ fm.A_L, composite=A)
for i, (k, op) in enumerate(blocks.items()):
    print(f"adjoint {k:10s}: {adj_err(op, 100 + i):.1e}")
# is the Sobolev inner product actually non-Euclidean (so the adjoint test bites)?
u = rng.standard_normal(Q.dim)
print("Q metric non-trivial:", Q.inner_product(Q.from_components(u), Q.from_components(u)) / np.dot(u, u))

# 4. degrees 0 and 1
print("\nDegree 0/1 content (max abs):")
for name, v in (("h_ref", h_ref), ("N_ref", N_ref), ("h(x)", h), ("N(x)", N), ("h_m", hm), ("h_L", hL), ("N_L", NL)):
    print(f"  {name:6s} l=0: {np.abs(v[deg==0]).max():.3e}   l=1: {np.abs(v[deg==1]).max():.3e}   l=2 max: {np.abs(v[deg==2]).max():.3e}")
# mean bed elevation the data carry at l=0 (orthonormal component = mean * sqrt(4 pi))
bed00 = fm.ref.field(Q, "bed_elevation")[0]
print(f"  bed elevation l=0 component {bed00:.1f} m (mean {bed00/np.sqrt(4*np.pi):.0f} m); h_ref l=0 = {h_ref[0]:.1f}")
b = fm.ref.field(Q, "bed_elevation")
m = deg >= 1
print(f"  h_ref vs bed elevation (l=1..{LMAX}): corr {np.dot(b[m], h_ref[m])/np.linalg.norm(b[m])/np.linalg.norm(h_ref[m]):.3f}, "
      f"rms ratio {np.linalg.norm(h_ref[m])/np.linalg.norm(b[m]):.3f}")
