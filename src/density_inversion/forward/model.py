"""Step 6 of the forward problem: everything assembled into one operator.

    (drho_m, x_L, h_s, dK)  ->  (h, N)

For each source X of load (mantle density, lithospheric unknowns, the sea-level constant), with load
operator L_X and direct geoid D_X:

    N_X = (I + T)^-1 (D_X + G_w P L_X)            the geoid, with the ocean following it (ocean.py)
    w_X = P (L_X - rho_w g C N_X)                 the deflection under its load and the water's
    h_X = w_X - N_X                               topography from sea level (orthometric), the observable

and h = sum_X h_X x_X + h_s + h_ref,  N = sum_X N_X x_X + N_ref, with

    A_m   mantle density -> load on the plate (step 1)            G_m  its geoid (anomaly + CMB, step 5)
    A_L   lithospheric unknowns -> load (step 3c)                  G_L  their geoid (mass sheets, step 5)
    P     load -> deflection (step 2, land and sea)                G_w  the deflected plate's geoid (step 5)
    U     dK (kg/m^2) -> the uniform load g dK (the sea-level constant, below)
    T     the ocean's self-gravitation (ocean.py)
    h_s   the surface field (step 4): added to h, no load, no geoid

The fixed reference terms are the same equations for the WINTERC-G lithosphere's load q_ref + U K0 and
its own geoid N_ref(structure).

The sea-level constant
----------------------
Each column floats where (rho_LAB - rho_w C) w = K - M (no plate): K is the column mass that floats exactly
at sea level. The reference load q_ref = -g (M - <M>) takes K = <M>, but the mean column (mostly oceanic)
sits ~2.4 km below sea level. Because the restoring contrast differs at sea and on land, a wrong K is not a
uniform shift: it moves oceans ~1.45 times as far as continents, a step along every coast with power at
all degrees (audit finding L1 / P12 / G8). K is one number: K0 is fitted once to observed bed elevation
(least squares over columns that hold the whole lithosphere, `sea_level_constant`) and used in the offset;
dK, the departure from it, is an unknown with prior mean 0 and a broad standard deviation SEA_LEVEL_SIGMA.
In the operator dK is a uniform pressure g dK on the plate.

The ocean that follows the geoid
--------------------------------
See ocean.py (audit finding M1 / G1 / P4 / P5). The water term that step 1 used to put into the mantle
load is gone; the water is now loaded where it is (at sea), with the model's own geoid, and its mass is in
the geoid.

Dynamic topography is the mantle column of the topography row, h_m = w_m - N_m.

Degrees 0 and 1
---------------
The geoid has no degree 0 or 1 (mass conservation; centre-of-mass frame). Topography keeps both: degree 0 is
what the sea-level constant fits, and degree 1 is frame-dependent (the plate's masses sit at other radii
than flowkernels' top, so the model's own centre of mass is not exactly flowkernels'; about 4 % of h at l = 1
for the mantle term, audit G7). Degree-1 topography should therefore not be used as data.

Approximations kept (audit; sizes for L <= 50)
----------------------------------------------
* first order in every perturbation (finite-amplitude interface terms ~(l+2) dr / 2R: 4 % at l = 50 for a
  10 km Moho change; cross terms Moho x density 3-11 % for 10 km and 30 kg/m^3);
* isostatic balance per unit area rather than per steradian (moves a compensated lithospheric geoid by up to
  ~15 %, because it is a small remainder of large terms);
* g0 = GM / R^2 = 9.820 for the geoid, against 9.882 from the model's own mass (0.6 %, audit M8);
* one g in the plate and the loads (plate.py);
* the mantle geoid from flowkernels' free-top flow: the plate's masses enter its self-gravitation only through
  the top boundary's mass, which the plate puts back to within 0.6 % at l = 2.

The model is affine: forward(x) = A(x) + offset. A is a pygeoinf BlockLinearOperator from the direct sum
(mantle density space X, lithospheric space, h_s space, the 1-d space of dK) to the direct sum (topography
space, geoid space), both copies of the load space Q.

Usage
-----
    fm = ForwardModel(MantleFlow(lmax=50))
    A = fm.operator                          # the linear part
    h, N = fm.forward(x_m, x_L, x_s, x_K)    # components, offsets included
    parts = fm.parts(x_m, x_L, x_s, x_K)     # every contribution separately (for figures)
    D = fm.dynamic_topography                # h_m
"""

from __future__ import annotations

import numpy as np
from pygeoinf import BlockLinearOperator, EuclideanSpace, LinearOperator

from .geoid import Geoid
from .lithosphere import ReferenceLithosphere
from .ocean import SelfGravitatingOcean
from .plate import OceanLoadedPlate
from .surface import SurfaceField

SEA_LEVEL_SIGMA = (
    1.0e6  # kg/m^2: prior standard deviation of dK, ~400 m of elevation at sea
)


class ForwardModel:
    """Mantle density, lithospheric perturbations, the surface field and the sea-level constant -> total
    topography and the geoid."""

    def __init__(
        self,
        mf,
        *,
        Te=40e3,
        include_reference=True,
        X=None,
        Q=None,
        surface=None,
        ocean_rtol=1e-12,
    ):
        self.mf = mf
        self.X = X or mf.density_space()
        self.Q = Q or mf.load_space()
        self.ref = ReferenceLithosphere.from_mantle(mf)
        self.plate = OceanLoadedPlate.from_mantle(mf, Te=Te, ocean=self.ref.ocean)
        self.geoid = Geoid(mf, self.ref, ocean=self.ref.ocean)
        self.surface = surface or SurfaceField()
        self.include_reference = bool(include_reference)

        Q, geo = self.Q, self.geoid
        H = self.ref._fine(Q)
        self.H = H
        self.truncate = H.degree_transfer_operator(Q)
        # deflections stay on the fine space until multiplied by the column's contrasts (audit G2)
        self.P_H = self.plate.operator(Q, W=H)
        self.P = self.truncate @ self.P_H
        self.A_m = mf.operator(self.X, Q)
        self.A_L = self.ref.operator(Q)
        self.G_m = geo.mantle_operator(self.X, Q)
        self.G_L = geo.lithosphere_operator(Q)
        self.G_w = geo.deflection_operator(Q, W=H)  # perturbations: sea-level frame
        self.G_w_col = geo.deflection_operator(
            Q, W=H, frame="column"
        )  # the reference's lift
        self.S = self.surface.space(Q.lmax)
        self.I_s = self.surface.operator(self.S, Q)
        self.K_space = EuclideanSpace(1)
        self.U = self.uniform_load_operator()

        LA, LO = geo._on_fine_grid(Q)
        C = np.asarray(geo.ocean(LA, LO), float)
        self.C = (
            H.degree_transfer_operator(Q)
            @ H.multiplication_operator(C)
            @ Q.degree_transfer_operator(H)
        )
        ocean = lambda G_w: SelfGravitatingOcean(
            Q,
            self.P_H,
            G_w,
            self.C,
            geo.sheet(Q, geo.R),
            geo.rho_water,
            geo.g0,
            rtol=ocean_rtol,
        )
        self.ocean = ocean(self.G_w)  # perturbations
        self.ocean_col = ocean(
            self.G_w_col
        )  # the reference, linearised about the undeflected column

        # each source: (geoid N_X, deflection w_X, topography h_X) as operators from its own space
        Binv, P, G_w, load = self.ocean.inverse, self.P_H, self.G_w, self.ocean.load
        self.blocks = {}
        for name, L_X, D_X in (
            ("mantle", self.A_m, self.G_m),
            ("lithosphere", self.A_L, self.G_L),
            ("sea_level", self.U, None),
        ):
            r = G_w @ P @ L_X if D_X is None else D_X + G_w @ P @ L_X
            N_X = Binv @ r
            w_X = P @ L_X + P @ load @ N_X
            self.blocks[name] = dict(N=N_X, w=w_X, h=self.truncate @ w_X - N_X)

        b = self.blocks
        zero = Q.spectral_operator(np.zeros(Q.dim)) @ self.I_s
        self.dynamic_topography = b["mantle"]["h"]
        self.dynamic_deflection = b["mantle"]["w"]
        self.operator = BlockLinearOperator(
            [
                [
                    b["mantle"]["h"],
                    b["lithosphere"]["h"],
                    self.I_s,
                    b["sea_level"]["h"],
                ],
                [b["mantle"]["N"], b["lithosphere"]["N"], zero, b["sea_level"]["N"]],
            ]
        )
        self._offset = None
        self._K0 = None

    # ---------------------------------------------------------------- the sea-level constant
    def uniform_load_operator(self):
        """dK (kg/m^2, a 1-vector) -> the uniform load g dK (Pa) as components of Q.

        The orthonormal Y_00 is 1/sqrt(4 pi) on the unit sphere, so a constant c has component c sqrt(4 pi).
        """
        Q = self.Q
        M = np.zeros((Q.dim, 1))
        M[0, 0] = self.ref.g * np.sqrt(4 * np.pi) * Q.radius
        return LinearOperator.from_matrix(self.K_space, Q, M, form="components")

    def _reference(self, K):
        """(h, N, w) of the reference lithosphere for sea-level constant K (kg/m^2 relative to <M>)."""
        Q, a = self.Q, self._apply
        q = self.ref.reference_load(Q) + a(self.U, np.array([K]))
        r = self._N_structure() + a(self.G_w_col @ self.P_H, q)
        N = a(self.ocean_col.inverse, r)
        w = a(self.P_H, q + a(self.ocean_col.load, N))  # on the fine space
        return a(self.truncate, w) - N, N, w

    def _N_structure(self):
        if getattr(self, "_N_struct", None) is None:
            self._N_struct = self.geoid.reference(self.Q)
        return self._N_struct

    def sea_level_constant(self):
        """K0 - <M> (kg/m^2): the fit of the reference topography to observed bed elevation.

        The reference topography is affine in K: h(K) = h(0) + K u. K0 = sum w u (b - h(0)) / sum w u^2 on
        the fine Gauss-Legendre grid, w the quadrature (area) weights. Only columns whose whole lithosphere
        lies inside the 105 km column are used (WINTERC-G LAB above the column base): where the LAB is
        deeper, the dense cold root below the column is missing from the reference, and those columns sit
        too high by an amount that grows with LAB depth (+0.1 km for LAB 100-150 km, +0.8 km beyond 200 km).
        Fitting them too would bias K0 by that missing mass, which belongs to the mantle unknown.
        """
        if self._K0 is None:
            Q, ref = self.Q, self.ref
            H = ref._fine(Q)
            up = Q.degree_transfer_operator(H)
            grid = lambda comp: np.asarray(H.grid_values(up(Q.from_components(comp))))
            b = ref.field(Q, "bed_elevation")
            h0 = self._reference(0.0)[0]
            u = self._reference(1.0)[0] - h0
            LA, LO = ref._grid(H)
            s = ref.structure(LA, LO)
            inside = s["lab"] <= ref.T - s["bed"]
            _, wlat = np.polynomial.legendre.leggauss(LA.shape[0])
            w = np.broadcast_to(wlat[:, None], LA.shape) * inside
            gu, gr = grid(u), grid(b - h0)
            self._K0 = float(np.sum(w * gu * gr) / np.sum(w * gu * gu))
        return self._K0

    # ---------------------------------------------------------------- spaces
    @property
    def model_space(self):
        return self.operator.domain

    @property
    def data_space(self):
        return self.operator.codomain

    def lithosphere_space(self):
        return self.A_L.domain

    # ---------------------------------------------------------------- offsets
    def offset(self):
        """(h_ref, N_ref) as components: the reference lithosphere's topography and geoid (zero if excluded)."""
        if self._offset is None:
            if not self.include_reference:
                z = np.zeros(self.Q.dim)
                self._offset = (z, z.copy(), np.zeros(self.H.dim))
            else:
                self._offset = self._reference(self.sea_level_constant())
        return self._offset[:2]

    # ---------------------------------------------------------------- evaluation
    @staticmethod
    def _apply(op, comp):
        return op.codomain.to_components(op(op.domain.from_components(comp)))

    def model_vector(self, x_m=None, x_L=None, x_s=None, x_K=None):
        """A vector of the model space from components (zeros where None); x_K is dK in kg/m^2."""
        M = self.model_space
        parts = [x_m, x_L, x_s, None if x_K is None else np.atleast_1d(x_K)]
        dims = [self.X.dim, self.lithosphere_space().dim, self.S.dim, 1]
        comp = np.concatenate(
            [
                np.zeros(d) if p is None else np.asarray(p, float)
                for p, d in zip(parts, dims)
            ]
        )
        return M.from_components(comp)

    def forward(self, x_m=None, x_L=None, x_s=None, x_K=None):
        """(h, N) as components, reference offsets included."""
        d = self.data_space.to_components(
            self.operator(self.model_vector(x_m, x_L, x_s, x_K))
        )
        h, N = d[: self.Q.dim], d[self.Q.dim :]
        h_ref, N_ref = self.offset()
        return h + h_ref, N + N_ref

    def parts(self, x_m=None, x_L=None, x_s=None, x_K=None):
        """Every contribution separately, as components.

        Topography by source: dynamic (mantle), lithosphere, reference (with K0), surface, sea_level (dK).
        Geoid by mass: mantle (anomaly + CMB), lithosphere (perturbation sheets), deflection (the moved
        columns, all loads), reference (the reference lithosphere's own structure), ocean (the water that
        follows the geoid). Both sum to `forward`.
        """
        Q, a = self.Q, self._apply
        z = np.zeros(Q.dim)
        zH = np.zeros(self.H.dim)
        self.offset()
        h_ref, N_ref, w_ref = self._offset
        xs = dict(
            mantle=x_m,
            lithosphere=x_L,
            sea_level=None if x_K is None else np.atleast_1d(np.asarray(x_K, float)),
        )
        h, N, w = {}, {}, {}
        for name, x in xs.items():
            if x is None:
                h[name], N[name], w[name] = z, z, zH
            else:
                blk = self.blocks[name]
                h[name], N[name], w[name] = (a(blk[k], x) for k in ("h", "N", "w"))
        h_s = a(self.I_s, x_s) if x_s is not None else z
        N_total = N["mantle"] + N["lithosphere"] + N["sea_level"] + N_ref
        w_pert = w["mantle"] + w["lithosphere"] + w["sea_level"]
        geoid = dict(
            mantle=a(self.G_m, x_m) if x_m is not None else z,
            lithosphere=a(self.G_L, x_L) if x_L is not None else z,
            deflection=a(self.G_w, w_pert) + a(self.G_w_col, w_ref),
            reference=self._N_structure() if self.include_reference else z,
            ocean=a(self.ocean.sheet, N_total),
        )
        geoid["total"] = sum(geoid.values())
        topography = dict(
            dynamic=h["mantle"],
            lithosphere=h["lithosphere"],
            reference=h_ref,
            surface=h_s,
            sea_level=h["sea_level"],
        )
        topography["total"] = sum(topography.values())
        return dict(topography=topography, geoid=geoid)
