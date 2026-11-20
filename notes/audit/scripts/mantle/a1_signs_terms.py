"""Task 1/2/5/6: sign of flowkernels' PHI, size of the (rho_w/rho_ref) PHI term in K_q, c_top consistency,
degree 1, sign structure of K_q and K_N."""
import numpy as np
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow, viscosity_jump

mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40, cache=False)
s, c = mf.shell, mf.shell.constants
r, R = s.r, mf.R
g = s.gravity
print("r_top km", r[-1]/1e3, "g_top", g[-1], "g0", c.G*c.earth_mass/R**2, "rho_top", s.rho[-1])
print("core: gm_core implies rho", c.gm_core/(c.G*4/3*np.pi*r[0]**3), "vs rho_core", c.rho_core)

# ---- 1. sign of PHI: rigid test. A positive anomaly at a node: attr_top>0 (geodesy sign U=-Phi>0 above a mass
blk, jump, at, ab = fk._source_terms(s, 2, fk._layer_propagators(s, 2))
print("DPHI jump sign (physics sign would be +):", np.sign(jump[100, fk.DPHI]), " attr_top sign:", np.sign(at[100]))

# ---- 2. ratio of the PHI term to sigma in K_q, by degree and depth
depths = np.array([200, 400, 660, 700, 1000, 1500, 2000, 2500, 2800]) * 1e3
idx = [np.argmin(abs(r - (R - d))) for d in depths]
print("\nratio (rho_w/rho_ref) PHI_top / sig_top  [= rho_w U / sigma_rr at r_top]")
print("l    " + " ".join(f"{d/1e3:7.0f}" for d in depths))
rat = {}
for l in [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40]:
    tr = fk.solve_degree_traces(s, l)
    sig, phi = tr[:, 1], tr[:, 2]
    ratio = (c.rho_water / c.rho_ref) * phi / sig
    rat[l] = ratio
    print(f"{l:3d} " + " ".join(f"{ratio[i]:7.4f}" for i in idx))

    # c_top check: phi_top + c_top sig_top should equal the potential of anomaly + CMB only
    # => check against the direct formula using attr_top and the CMB traces
    if l > 1:
        fac = 4*np.pi*c.G/(2*l+1)
        c_top = fac*r[-1]/g[-1]*c.rho_ref
        c_bot = -fac*r[0]/g[0]*c.rho_ref*(r[0]/r[-1])**l
        lhs = phi + c_top*sig
        rhs = at_l = fac*c.rho_ref/c.visc_ref*(r/r[-1])**l*r**2 - c_bot*(tr[:, 3] + c.rho_core/c.rho_ref*tr[:, 4])
        # rhs as above: phi + c_top sig + c_bot (sig_b + rc/rr phi_b) = attr
        print(f"     potential row residual / |attr| max: {np.max(abs(lhs-rhs))/np.max(abs(at_l)):.2e}")

# ---- land vs sea: q_sea = -(sig + rho_w U), q_land = -sig. relative difference |rho_w U / (sig + rho_w U)|
print("\nrelative change in load, land (air) vs sea, |q_land - q_sea|/|q_sea|, kernel-weighted RMS over depth")
w = np.gradient(r)
for l in [2, 4, 8, 20, 40]:
    tr = fk.solve_degree_traces(s, l)
    sig, phi = tr[:, 1], tr[:, 2]
    qs = sig + c.rho_water/c.rho_ref*phi
    print(f"l={l:2d}: L2 rel {np.sqrt(np.sum(w*(qs-sig)**2)/np.sum(w*qs**2)):.4f};"
          f" at 1000 km {abs((qs-sig)/qs)[idx[4]]:.4f}")

# ---- 5/6. sign structure
K = mf.kernels
print("\nK_q interior max (should be <=0):", K["K_q"][:, 1:-1].max(), " min", K["K_q"].min())
for l in [1, 2, 3, 4, 5, 6, 8, 10, 20]:
    kN = K["K_N"][l-1]; kfk = K["K_N_fk"][l-1]
    rr = K["r"]
    sc = np.flatnonzero(np.diff(np.sign(kN[1:-1])) != 0)
    sc2 = np.flatnonzero(np.diff(np.sign(kfk[1:-1])) != 0)
    print(f"l={l:2d} K_N sign changes at depth km {[(R-rr[i+1])/1e3 for i in sc]}, "
          f"K_N_fk (with top deflection) sign changes {[round((R-rr[i+1])/1e3) for i in sc2]}")
    if l == 1:
        print("   l=1 K_q range", K["K_q"][0].min(), K["K_q"][0].max(), " K_N", abs(K["K_N"][0]).max(),
              "K_N_fk", abs(K["K_N_fk"][0]).max())
