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
Bending uses the flat-plate operator on the sphere, nabla^4 -> [l(l+1)/R^2]^2. The thin spherical shell adds a
membrane term E T_e / R^2 (about 0.4 % of drho g at T_e = 40 km) and replaces [l(l+1)]^2 by l(l+1)(l(l+1) - 4),
which matters only at low degree where bending is negligible; both are left out.

Two plates, built up in order
-----------------------------
ElasticPlate        constant drho = rho_LAB - rho_w (water everywhere). Diagonal in degree:

                        w_lm = F_l q_lm,      F_l = 1 / (D [l(l+1)/R^2]^2 + drho g).

OceanLoadedPlate    drho(x) from the ocean function, with fixed present-day coastlines. Writing
                    drho(x) = (rho_LAB - rho_w) + rho_w (1 - C(x)), the equation is

                        (diag(1/F_l) + g rho_w M) w = q,      M_ij = int (1 - C) Y_i Y_j dOmega,

                    a symmetric positive-definite matrix coupling the degrees (land is stiffer). M is
                    computed on a Gauss-Legendre grid, so it is exactly symmetric; the matrix is solved once.
                    With C = 1 everywhere it is the ElasticPlate.

Both take rho_LAB, rho_w, g at the LAB and the surface radius R from step 1 (MantleFlow), so with T_e = 0 and
water everywhere the plate gives back step 1's unflexed deflection exactly.

pygeoinf
--------
Domain: the load space of step 1 (a sphere Sobolev space, components = orthonormal coefficients in Pa).
Codomain: a sphere Sobolev space for the deflection (components in m), with the same order and length scale.

Usage
-----
    from density_inversion.forward.mantle import MantleFlow
    from density_inversion.forward.plate import ElasticPlate, OceanLoadedPlate
    mf = MantleFlow(lmax=50)
    plate = ElasticPlate.from_mantle(mf, Te=40e3)           # water everywhere
    land_sea = OceanLoadedPlate.from_mantle(mf, Te=40e3)    # air on land
    Q = mf.load_space()
    P = land_sea.operator(Q)                                # q (Pa) -> w (m)
    h_dyn = P @ mf.operator()                               # mantle density -> dynamic topography
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pyshtools as pysh
import scipy.linalg as sla
import scipy.sparse as sp
from pygeoinf import LinearOperator

from .examples import cilm_to_components, components_to_cilm, sphere_ordering

E_DEFAULT = 90e9  # Pa, as in Chapter 1
NU_DEFAULT = 0.25


def rigidity(Te, E=E_DEFAULT, nu=NU_DEFAULT):
    """Flexural rigidity D = E Te^3 / (12 (1 - nu^2)) in N m."""
    return E * Te**3 / (12 * (1 - nu**2))


def ocean_function(path=None):
    """The ocean function on its 1 degree grid: (lat centres, lon centres, C), lat from 89.5N down."""
    if path is None:
        path = resources.files("density_inversion") / "data" / "ocean_function.csv"
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

    def bending(self):
        """D [l(l+1)/R^2]^2 for l = 0..lmax, Pa per m."""
        l = self.degrees()
        return self.D * (l * (l + 1) / self.R**2) ** 2

    def filter(self):
        """F_l = 1 / (D [l(l+1)/R^2]^2 + drho g), m per Pa, l = 0..lmax."""
        return 1.0 / (self.bending() + self.drho * self.g)

    def deflection_space(self, Q):
        """A sphere space like the load space Q, for the deflection in m."""
        return _like(Q)

    def matrix(self, Q):
        deg, _ = sphere_ordering(Q.lmax)
        return sp.diags(self.filter()[deg])

    def operator(self, Q, W=None):
        """The LinearOperator q (Pa) -> w (m), from the load space Q."""
        W = W or self.deflection_space(Q)
        return LinearOperator.from_matrix(Q, W, self.matrix(Q), form="components")


class OceanLoadedPlate(ElasticPlate):
    """The same plate, with water filling the deflection in the oceans and air on land.

    ``drho`` is the ocean value rho_LAB - rho_w; ``rho_water`` is what is missing on land. ``ocean`` is a
    function (lat, lon) -> C in degrees, by default the package's present-day ocean function; pass
    ``lambda lat, lon: np.ones_like(lat)`` for water everywhere or zeros for air everywhere.
    """

    def __init__(
        self, Te, drho, g, R, lmax, *, rho_water, ocean=None, quad_lmax=None, **kw
    ):
        super().__init__(Te, drho, g, R, lmax, **kw)
        self.rho_water = float(rho_water)
        self.ocean = ocean or sample_ocean_function
        self.quad_lmax = int(quad_lmax or 3 * self.lmax)
        self._K = None
        self._inverse = None

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

    def land_matrix(self):
        """M_ij = int (1 - C) Y_i Y_j dOmega for the sphere components up to lmax (Gauss-Legendre quadrature)."""
        L, Lq = self.lmax, self.quad_lmax
        zero, w = pysh.expand.SHGLQ(Lq)
        lat, lon = pysh.expand.GLQGridCoord(Lq)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        land = 1.0 - self.ocean(LA, LO)
        n = (L + 1) ** 2
        M = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            cilm = np.zeros((2, Lq + 1, Lq + 1))
            cilm[:, : L + 1, : L + 1] = components_to_cilm(e, L)
            grid = pysh.expand.MakeGridGLQ(cilm, zero, norm=4, csphase=1)
            out = pysh.expand.SHExpandGLQ(
                land * grid, w, zero, norm=4, csphase=1, lmax_calc=L
            )
            M[:, j] = cilm_to_components(out, L)
            e[j] = 0.0
        return 0.5 * (M + M.T)  # symmetric to rounding already; remove the last bit

    def system(self):
        """K = diag(1/F_l) + g rho_w M: the plate equation K w = q."""
        if self._K is None:
            deg, _ = sphere_ordering(self.lmax)
            self._K = (
                np.diag(1.0 / super().filter()[deg])
                + self.g * self.rho_water * self.land_matrix()
            )
        return self._K

    def matrix(self, Q):
        if Q.lmax != self.lmax:
            raise ValueError("the load space must have the plate's lmax")
        if self._inverse is None:
            K = self.system()
            self._inverse = sla.cho_solve(sla.cho_factor(K), np.eye(K.shape[0]))
        return self._inverse


def _like(Q):
    """A new sphere space with the same type, lmax, radius and Sobolev parameters as Q."""
    from pygeoinf.symmetric_space import sphere

    if isinstance(Q, sphere.Sobolev):
        return sphere.Sobolev(Q.lmax, Q.order, Q.length_scale, radius=Q.radius)
    return sphere.Lebesgue(Q.lmax, radius=Q.radius)
