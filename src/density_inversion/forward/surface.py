"""Step 4 of the forward problem: the surface field h_s.

Physics
-------
h_s is topography the long-wavelength model is not meant to explain: orogens, volcanic edifices, short-
wavelength crustal structure the reference lithosphere misses. It is added to the topography as it is:

    h = P (q_m + q_L + q_ref) + h_s,

with no load, no flexure and no geoid. It is a nuisance field: what separates it from dynamic topography is its
prior, which should put its power at short wavelengths (an open choice; see figures/04_surface_field.py).

pygeoinf
--------
Its space is a sphere Sobolev space like the load space (components in m). The operator into topography is the
identity, or a degree transfer if the two spaces differ in lmax.

Usage
-----
    sf = SurfaceField(order=2.0, length_scale=0.05)
    S = sf.space(lmax)
    I = sf.operator(S, Q)          # h_s -> its contribution to topography
"""

from __future__ import annotations

import numpy as np
from pygeoinf.symmetric_space import sphere


class SurfaceField:
    """The additive surface field h_s and its operator into topography."""

    def __init__(self, *, order=2.0, length_scale=0.05):
        self.order, self.length_scale = float(order), float(length_scale)

    def space(self, lmax):
        """The h_s space: a Sobolev space on the unit sphere (components are coefficients in m)."""
        return sphere.Sobolev(int(lmax), self.order, self.length_scale)

    def operator(self, S, Q):
        """h_s (in S) -> its contribution to topography (in Q): the identity, or truncation/padding."""
        if S == Q:
            return Q.spectral_operator(np.ones(Q.dim))
        return S.degree_transfer_operator(Q)


def high_pass(lmax, l0, width=5.0):
    """A smooth step in degree, ~0 below l0 and ~1 above it: a shape for an h_s prior spectrum."""
    l = np.arange(lmax + 1, dtype=float)
    return 0.5 * (1 + np.tanh((l - l0) / width))
