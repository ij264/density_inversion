"""Task 1: S_l(r) vs direct Newton quadrature for a spherical-cap mass sheet at depth.

N = -Phi/g0, Phi(x) = -G int sigma(x') r_s^2 dOmega' / |x - x'| (no spherical harmonics),
compared with sum_l S_l(r_s) sigma_l P_l(cos theta) using the repo's Geoid.sheet_factor formula
(evaluated for all l, including 0 and 1, which the repo zeroes)."""
import numpy as np
from scipy.special import eval_legendre
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.geoid import Geoid

mf = MantleFlow(lmax=12)
geo = Geoid(mf, ReferenceLithosphere.from_mantle(mf))
R, g0, G = geo.R, geo.g0, geo.G
r_s = R - 100e3
alpha = np.radians(10.0)
sig0 = 1000.0  # kg/m^2


class FakeQ:
    def __init__(self, L):
        self.degrees = np.arange(L + 1)


L = 3000
S = 4 * np.pi * G * r_s * (r_s / R) ** (np.arange(L + 1) + 1) / ((2 * np.arange(L + 1) + 1) * g0)
# the repo's function (zeroes l<2) for l>=2:
S_repo = geo.sheet_factor(FakeQ(L), r_s)
assert np.allclose(S_repo[2:], S[2:], rtol=1e-14)

x0 = np.cos(alpha)
l = np.arange(L + 1)
Pm = np.array([eval_legendre(max(j - 1, 0), x0) if j > 0 else 1.0 for j in l])
Pp = np.array([eval_legendre(j + 1, x0) for j in l])
sig_l = sig0 / 2 * (Pm - Pp)  # coefficients of sum sig_l P_l
sig_l[0] = sig0 / 2 * (1 - x0)

thetas = np.radians([0.0, 5.0, 15.0, 30.0, 60.0, 120.0])

# direct quadrature: Gauss-Legendre in theta' (on [0, alpha]) and phi' (on [0, 2pi])
nt, npf = 800, 1600
xt, wt = np.polynomial.legendre.leggauss(nt)
tp = alpha / 2 * (xt + 1); wtp = alpha / 2 * wt
pp = np.linspace(0, 2 * np.pi, npf, endpoint=False); wpp = 2 * np.pi / npf
TP, PP = np.meshgrid(tp, pp, indexing="ij")
dA = r_s ** 2 * np.sin(TP) * wtp[:, None] * wpp
print(f"cap radius 10 deg, sheet 100 km deep, sigma = {sig0} kg/m^2")
print(" theta   N_direct(m)     N_spectral(m)   rel.diff   [l>=2 only: direct-minus-(l0,l1) vs spectral]")
for th in thetas:
    xs = R * np.array([np.sin(th), 0, np.cos(th)])
    X = r_s * np.stack([np.sin(TP) * np.cos(PP), np.sin(TP) * np.sin(PP), np.cos(TP)])
    d = np.sqrt(((X - xs[:, None, None]) ** 2).sum(0))
    Phi = -G * np.sum(sig0 * dA / d)
    Nd = -Phi / g0
    Pl = eval_legendre(l, np.cos(th))
    Ns = np.sum(S * sig_l * Pl)
    Nd2 = Nd - S[0] * sig_l[0] - S[1] * sig_l[1] * np.cos(th)
    Ns2 = np.sum((S * sig_l * Pl)[2:])
    print(f"{np.degrees(th):6.1f} {Nd: .8e} {Ns: .8e} {abs(Nd-Ns)/abs(Nd):.2e}   {abs(Nd2-Ns2)/abs(Nd2):.2e}")
# sign: positive mass -> N > 0 above it
