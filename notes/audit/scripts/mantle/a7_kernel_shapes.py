"""Task 5/6 and 4a (normalised): kernel shapes, sign changes, degree 1, grid error relative to the kernel peak."""
import numpy as np
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow, viscosity_jump, reference_model

mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40, cache=False)
full = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=1.0, lmax=10, cache=False)
uni = MantleFlow(viscosity=lambda r, R: np.where(r < r[np.argmin(abs(r - (R - 670e3)))], 1.0 + 1e-9, 1.0),
                 lab_depth=105e3, lmax=10, cache=False)
R = mf.R
for lab, m in [("LAB top, x30", mf), ("surface top, x30", full), ("LAB top, uniform", uni)]:
    k = m.kernels
    d = (R - k["r"]) / 1e3
    print(f"\n{lab}")
    for l in (2, 3, 4, 6, 8):
        for name in ("K_N_fk", "K_N"):
            K = k[name][l - 1]
            sl = slice(1, -1)
            zc = d[1:-1][np.flatnonzero(np.diff(np.sign(K[sl])) != 0)]
            print(f"  l={l} {name:6s}: shallow(300km) {np.interp(300, d[::-1], K[::-1]):+.3e}  "
                  f"deep(2000km) {np.interp(2000, d[::-1], K[::-1]):+.3e}  zero crossings km {np.round(zc).tolist()}")
        Kq = k["K_q"][l - 1]
        print(f"  l={l} K_q single-signed: {np.all(Kq[1:-1] < 0)}; peak at {d[np.argmin(Kq)]:.0f} km; "
              f"K_h0 at 300 km {np.interp(300, d[::-1], k['K_h0'][l-1][::-1]):.3e} m per kg/m^3 per m")

k = mf.kernels
d = (R - k["r"]) / 1e3
print("\nl=1 K_q (Pa per kg/m^2) at depths:", {z: round(float(np.interp(z, d[::-1], k['K_q'][0][::-1])), 3)
                                              for z in (105, 200, 670, 1500, 2500, 2891)})
print("l=2 K_q:", {z: round(float(np.interp(z, d[::-1], k['K_q'][1][::-1])), 3) for z in (105, 200, 670, 1500, 2500, 2891)})
print("g_lab", k["g_lab"])

# 4a: grid error normalised by the kernel peak
base = reference_model()
rr = np.concatenate([base.r[:1]] + [np.linspace(a, b, 5)[1:] for a, b in zip(base.r[:-1], base.r[1:])])
m4 = fk.ShellModel(r=rr, rho=np.interp(rr, base.r, base.rho), eta=np.ones_like(rr))
d_jump = R - float(k["r_jump"])
mf4 = MantleFlow(viscosity=viscosity_jump(30.0, d_jump), lab_depth=105e3, lmax=40, model=m4, cache=False)
k4 = mf4.kernels
print("\n4a: max |K(257) - K(x4)| / max|K(x4)|, on the 257 nodes")
for name in ("K_q", "K_N"):
    out = []
    for l in (1, 2, 5, 10, 20, 30, 40):
        a = k[name][l - 1]
        b = np.interp(k["r"], k4["r"], k4[name][l - 1])
        out.append(f"l={l}: {np.max(abs(a - b)) / max(np.max(abs(b)), 1e-300):.1e}")
    print(name, " ".join(out))
