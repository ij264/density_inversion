"""Step 2 of the forward problem: the elastic plate, load -> deflection.

Physics
-------
A thin elastic plate of constant elastic thickness T_e and flexural rigidity D = E T_e^3 / (12 (1 - nu^2)) lies
on the mantle. A load q (Pa, positive up) deflects it by w (m, positive up):

    D nabla^4 w + drho(x) g w = q.

The restoring term is the weight of the material that fills the deflection: below it is mantle, above it is
water in the oceans and air on land, so

    drho(x) = rho_LAB - rho_w C(x),        C the ocean function (1 ocean, 0 land).

The deflection is the topography: the surface moves with the plate, and water fills the deflection at sea.
Both plates use pygeoinf's flexure operator on the sphere (Sphere.flexural_operator and its inverse):

    Op(w) = L(D L w) - L(D_eff) L w + tr(Hess D_eff Hess w) + 2 K grad D_eff . grad w - K D_eff L w + drho g w,

with L the surface Laplacian, K = 1/R^2 the Gaussian curvature and D_eff = D (1 - nu). Our sphere spaces have
radius 1 (components are orthonormal coefficients), so the rigidity is passed as D / R^4; with that scaling
the operator is exactly the one on a sphere of radius R. For constant coefficients it is diagonal:

    w_lm = F_l q_lm,   F_l = 1 / ( (D/R^4) [lam^2 - (1 - nu) lam] + drho g ),   lam = l(l+1),

which differs from the flat-plate filter 1 / (D [l(l+1)/R^2]^2 + drho g) by the curvature term, at most a few
parts in 10^5 for l <= 50 and T_e = 40 km.

Two plates, built up in order
-----------------------------
ElasticPlate        constant drho = rho_LAB - rho_w (water everywhere): the diagonal filter above.

OceanLoadedPlate    drho(x) from the ocean function, with fixed present-day coastlines, and optionally a
                    laterally varying T_e(x). The operator is not diagonal: pygeoinf solves it matrix-free by
                    conjugate gradients, preconditioned by the constant-coefficient filter. Because C(x) is
                    a step at the coasts, the solve is done on a finer Gauss-Legendre grid (degree
                    quad_lmax, 3 lmax by default), the load padded with zeros and the deflection
                    truncated back to lmax. At 3 lmax the result is converged to about 1 %; on the model's own
                    grid aliasing errors reach 20 % at coasts. With C = 1 it is the ElasticPlate.

Both take rho_LAB, rho_w, g at the LAB and the surface radius R from step 1 (MantleFlow), so with T_e = 0 and
water everywhere the plate gives back step 1's unflexed deflection exactly.

pygeoinf
--------
Domain: the load space of step 1 (a sphere Sobolev space, components = orthonormal coefficients in Pa).
Codomain: the load space again by default (components then in m); pass another sphere space W to change it.

Usage
-----
    from density_inversion.forward.mantle import MantleFlow
    from density_inversion.forward.plate import ElasticPlate, OceanLoadedPlate
    mf = MantleFlow(lmax=50)
    plate = ElasticPlate.from_mantle(mf, Te=40e3)           # water everywhere
    land_sea = OceanLoadedPlate.from_mantle(mf, Te=40e3)    # air on land
    craton = OceanLoadedPlate.from_mantle(mf, Te=lambda lat, lon: ...)   # T_e(x) in m
    Q = mf.load_space()
    P = land_sea.operator(Q)                                # q (Pa) -> w (m)
    h_dyn = P @ mf.operator()                               # mantle density -> dynamic topography
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
from pygeoinf import CGSolver, LinearOperator
from pygeoinf.symmetric_space import sphere

CG_RTOL = 1e-10

E_DEFAULT = 90e9  # Pa, as in Chapter 1
NU_DEFAULT = 0.25


def rigidity(Te, E=E_DEFAULT, nu=NU_DEFAULT):
    """Flexural rigidity D = E Te^3 / (12 (1 - nu^2)) in N m."""
    return E * Te**3 / (12 * (1 - nu**2))


@lru_cache(maxsize=4)
def ocean_function(path=None):
    """The ocean function on its 1 degree grid: (lat centres, lon centres, C), lat from 89.5N down."""
    if path is None:
        path = (
            resources.files("density_inversion") / "data" / "ocean_function.csv"
        )
    C = np.loadtxt(path, delimiter=",")
    return 89.5 - np.arange(C.shape[0]), 0.5 + np.arange(C.shape[1]), C


def sample_ocean_function(lat, lon, path=None):
    """C at points (degrees), by the 1 degree cell each point falls in."""
    _, _, C = ocean_function(path)
    i = np.clip(np.floor(90.0 - np.asarray(lat)).astype(int), 0, C.shape[0] - 1)
    j = np.floor(np.mod(np.asarray(lon), 360.0)).astype(int) % C.shape[1]
    return C[i, j]


class ElasticPlate:
    """A thin elastic plate with constant elastic thickness and a constant restoring contrast."""

    def __init__(self, Te, drho, g, R, lmax, *, E=E_DEFAULT, nu=NU_DEFAULT):
        self.Te, self.drho, self.g, self.R, self.lmax = (
            float(Te),
            float(drho),
            float(g),
            float(R),
            int(lmax),
        )
        self.E, self.nu = float(E), float(nu)
        self.D = rigidity(self.Te, self.E, self.nu)

    @classmethod
    def from_mantle(cls, mf, Te=40e3, **kw):
        """Take rho_LAB - rho_w, g at the LAB, R and lmax from a MantleFlow (step 1)."""
        k = mf.kernels
        drho = float(k["rho_lab"]) - float(k["rho_water"])
        return cls(Te, drho, float(k["g_lab"]), float(k["R"]), mf.lmax, **kw)

    def degrees(self):
        return np.arange(self.lmax + 1)

    def filter(self):
        """F_l, m per Pa, l = 0..lmax: pygeoinf's constant-coefficient symbol (see the module notes)."""
        lam = self.degrees() * (self.degrees() + 1.0)
        return 1.0 / (
            self.D / self.R**4 * (lam**2 - (1 - self.nu) * lam)
            + self.drho * self.g
        )

    def flat_filter(self):
        """The flat-plate filter 1 / (D [l(l+1)/R^2]^2 + drho g), for comparison."""
        lam = self.degrees() * (self.degrees() + 1.0)
        return 1.0 / (self.D * (lam / self.R**2) ** 2 + self.drho * self.g)

    def deflection_space(self, Q):
        """The deflection space: the load space itself (pygeoinf does not track units; components are in m)."""
        return Q

    def operator(self, Q, W=None):
        """The LinearOperator q (Pa) -> w (m), from the load space Q (a sphere space of radius 1)."""
        _check_unit_sphere(Q)
        F = Q.inverse_flexural_operator(
            self.D / self.R**4, self.nu, self.drho * self.g
        )
        if W is not None and W != Q:
            F = Q.degree_transfer_operator(W) @ F
        return F


class OceanLoadedPlate(ElasticPlate):
    """The same plate, with water filling the deflection in the oceans and air on land.

    drho is the ocean value rho_LAB - rho_w; rho_water is what is missing on land. ocean is a
    function (lat, lon) -> C in degrees, by default the package's present-day ocean function; pass
    lambda lat, lon: np.ones_like(lat) for water everywhere or zeros for air everywhere. Te is a number
    (m) or a vectorised function (lat, lon) -> T_e in m, both in degrees.
    """

    def __init__(
        self,
        Te,
        drho,
        g,
        R,
        lmax,
        *,
        rho_water,
        ocean=None,
        quad_lmax=None,
        rtol=CG_RTOL,
        **kw,
    ):
        self.Te_field = Te if callable(Te) else None
        super().__init__(0.0 if callable(Te) else Te, drho, g, R, lmax, **kw)
        self.rho_water = float(rho_water)
        self.ocean = ocean or sample_ocean_function
        self.quad_lmax = int(quad_lmax or 3 * self.lmax)
        self.rtol = float(rtol)

    @classmethod
    def from_mantle(cls, mf, Te=40e3, **kw):
        k = mf.kernels
        drho = float(k["rho_lab"]) - float(k["rho_water"])
        return cls(
            Te,
            drho,
            float(k["g_lab"]),
            float(k["R"]),
            mf.lmax,
            rho_water=float(k["rho_water"]),
            **kw,
        )

    def fine_space(self, Q):
        """The Gauss-Legendre space the solve is done on: Q's Sobolev parameters at degree quad_lmax."""
        if isinstance(Q, sphere.Sobolev):
            return sphere.Sobolev(
                self.quad_lmax, Q.order, Q.length_scale, grid="GLQ"
            )
        return sphere.Lebesgue(self.quad_lmax, grid="GLQ")

    def fields(self, H):
        """(rigidity / R^4, buoyancy drho(x) g) on H's Gauss-Legendre grid."""
        import pyshtools as pysh

        lat, lon = pysh.expand.GLQGridCoord(H.lmax)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        if LA.shape != tuple(H.grid_shape):
            raise ValueError("unexpected grid layout")
        C = np.asarray(self.ocean(LA, LO), float)
        buoyancy = self.g * (self.drho + self.rho_water * (1.0 - C))
        if self.Te_field is None:
            D = self.D
        else:
            D = rigidity(
                np.asarray(self.Te_field(LA, LO), float), self.E, self.nu
            )
        return np.asarray(D) / self.R**4, buoyancy

    def operator(self, Q, W=None):
        """The LinearOperator q (Pa) -> w (m): pad to quad_lmax, solve the flexure there, truncate to W
        (by default Q)."""
        _check_unit_sphere(Q)
        W = W or Q
        H = self.fine_space(Q)
        D, buoyancy = self.fields(H)
        Finv = H.inverse_flexural_operator(
            D, self.nu, buoyancy, solver=CGSolver(rtol=self.rtol)
        )
        return (
            H.degree_transfer_operator(W) @ Finv @ Q.degree_transfer_operator(H)
        )

    def assembled(self, Q, W=None):
        """The same operator as an explicit matrix (one solve per component), for repeated use."""
        op = self.operator(Q, W)
        n = Q.dim
        M = np.empty((op.codomain.dim, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            M[:, j] = op.codomain.to_components(op(Q.from_components(e)))
            e[j] = 0.0
        return LinearOperator.from_matrix(Q, op.codomain, M, form="components")


def _check_unit_sphere(Q):
    if not np.isclose(Q.radius, 1.0):
        raise ValueError(
            "the plate expects sphere spaces of radius 1 (the rigidity is scaled by R^4)"
        )
