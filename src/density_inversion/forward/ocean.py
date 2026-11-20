"""The ocean that follows the geoid (audit finding M1 / G1 / P4 / P5).

Physics
-------
Sea level is an equipotential, so where the geoid rises by N the ocean deepens by N. To first order this
does two things at sea (C = 1) and nothing on land (C = 0):

* it loads the plate with the weight of the extra water,      dq = -rho_w g C N;
* it adds the extra water's mass to the geoid,                 dN = S(R) (rho_w C N),  a sheet at sea level.

Both depend on the total geoid N, which they change, so N solves

    N = r + S(R) rho_w C N - G_w P (rho_w g C N),      i.e.   (I + T) N = r,
    T = rho_w (g G_w P - S(R)) C,

where r = (direct geoid of the masses) + G_w P (their load): the geoid the model computed before the ocean
was added. The water's own load and its own mass nearly cancel (the plate sinks under the water as the water
adds mass), so T is small: its norm is a few tenths of rho_w / rho_mean at degree 2 and falls as 1/(2l+1).
We solve (I + T) N = r by a Neumann series, N = sum_k (-T)^k r, to a relative tolerance `rtol`, and the
adjoint the same way with T's adjoint.

The deflection is then w = P (q - rho_w g C N), and the observed topography, measured from sea level
(orthometric heights, on land and at sea), is h = w - N.

This replaces the water term (rho_w / rho_ref) PHI that step 1 used to put into the load: that term was the
same water, but taken from flowkernels' geoid of a freely deflecting top at r_LAB, applied on land as well
as at sea, and its mass was never added to the geoid.

Sea level itself is the geoid with zero mean (the datum of the data); a uniform offset is the sea-level
constant of model.py.
"""

from __future__ import annotations

import numpy as np
from pygeoinf import LinearOperator


class SelfGravitatingOcean:
    """(I + T)^-1 for the ocean that follows the geoid, and the operators it is built from.

    Args:
        Q: the load / geoid space (sphere, unit radius).
        P: load (Pa) -> deflection (m), Q -> Q.
        G_w: deflection (m) -> geoid (m) of the moved column, Q -> Q.
        C: the ocean function as multiplication Q -> Q (sea 1, land 0).
        S: sigma (kg/m^2) -> geoid of a sheet at sea level, Q -> Q.
        rho_w: water density (kg/m^3).
        g: gravity at sea level (m/s^2).
        rtol: relative tolerance of the Neumann series.
    """

    def __init__(self, Q, P, G_w, C, S, rho_w, g, *, rtol=1e-12, max_terms=200):
        self.Q, self.P, self.G_w, self.C, self.S = Q, P, G_w, C, S
        self.rho_w, self.g = float(rho_w), float(g)
        self.rtol, self.max_terms = float(rtol), int(max_terms)
        self.T = self.rho_w * ((self.g * (G_w @ P)) - S) @ C
        self.load = (-self.rho_w * self.g) * C  # N -> the water's load on the plate
        self.sheet = self.rho_w * (S @ C)  # N -> the water's geoid
        self.inverse = LinearOperator.from_callables(
            Q,
            Q,
            lambda r: self._neumann(self.T, r),
            adjoint=lambda r: self._neumann(self.T.adjoint, r),
        )

    def _neumann(self, T, r):
        Q = self.Q
        rc = Q.to_components(r)
        scale = np.linalg.norm(rc)
        if scale == 0:
            return Q.from_components(rc)
        term, total = rc.copy(), rc.copy()
        for _ in range(self.max_terms):
            term = -Q.to_components(T(Q.from_components(term)))
            total += term
            if np.linalg.norm(term) <= self.rtol * scale:
                return Q.from_components(total)
        raise RuntimeError("the ocean's Neumann series did not converge")

    def contraction(self, n_iter=30, seed=0):
        """An estimate of the spectral radius of T (power iteration); the series needs it below 1."""
        Q = self.Q
        x = np.random.default_rng(seed).standard_normal(Q.dim)
        rate = 0.0
        for _ in range(n_iter):
            y = Q.to_components(self.T(Q.from_components(x)))
            rate = np.linalg.norm(y) / np.linalg.norm(x)
            x = y / np.linalg.norm(y)
        return rate
