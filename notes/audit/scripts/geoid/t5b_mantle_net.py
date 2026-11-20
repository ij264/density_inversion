"""Task 5 continued: effect of the sheet placement on the NET mantle geoid G_m x + G_w P A_m x (l <= 50),
for a random mantle density, against the deflection sheets at their own depths (sea-level frame,
where the deflected plate actually is, and column frame)."""
import numpy as np
import t45_sheets as T45

Q, mf, geo, H, deg = T45.Q, T45.mf, T45.geo, T45.H, T45.deg
apply, band, exact_sheets = T45.apply, T45.band, T45.exact_sheets
X = mf.density_space()
rng = np.random.default_rng(7)
x = rng.standard_normal(X.dim)
A = mf.operator(X, Q)
w = apply(T45.P, apply(A, x))
full = np.zeros(H.dim); full[: Q.dim] = w
wg = np.asarray(H.grid_values(H.from_components(full)))
Nm = apply(geo.mantle_operator(X, Q), x)
Nw = apply(T45.Gw, w)
net = Nm + Nw
print("cancellation |G_w w| / |net| by band:", "  ".join(
    f"l{a}-{b}: {np.linalg.norm(Nw[(deg>=a)&(deg<=b)])/np.linalg.norm(net[(deg>=a)&(deg<=b)]):.2f}"
    for a, b in ((2, 4), (5, 10), (11, 20), (21, 35), (36, 50))))
for frame in ("column", "sealevel"):
    ex = Nm + exact_sheets(T45.lifted_column(wg, frame))
    print(f"net mantle geoid, repo vs own-depth sheets ({frame}): {band(net, ex)}")
# surface sheet only at the seafloor/land surface (-bed) instead of at R
sh = [(np.full(T45.LA.shape, d), f * wg) for d, f in geo.deflection_sheets(T45.LA, T45.LO)]
sh2 = [(-T45.s["bed"], sh[0][1])] + sh[1:]
print("net mantle geoid, surface sheet at R vs at R+bed alone:", band(net, Nm + exact_sheets(sh2)))
