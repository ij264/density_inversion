"""Step 5 of the forward problem: the geoid, from every mass the model moves.

Physics
-------
The geoid is N = -Phi / g0 at the surface (physics-sign potential, nabla^2 Phi = 4 pi G rho; g0 surface
gravity), so a mass excess raises it. Five kinds of mass contribute, each at its own radius:

1. the mantle density anomaly drho_m and the deflection of the CMB it drives: from flowkernels, with the
   mass of the deflected top of the shell taken out (step 1's kernel K_N; see forward/mantle.py);
2. the lithospheric perturbations of step 3 (crust and lid densities, Moho and LAB displacements), each a
   mass sheet sigma(x) = f(x) u(x) at its own local depth, f the factor field that makes its load;
3. the plate's deflection w: lifting the whole column by w puts a sheet Drho_i(x) w(x) on every density
   contrast i of the column, each at its own local depth:

       ice top   rho_ice - rho_w C(x)                (where there is ice)
       surface   rho_c(x) - rho_ice or rho_w C(x)    (rock for ice; for water at sea; for air on land)
       Moho      rho_lid,top(x) - rho_c(x)
       lid       the lid's own gradient, one sheet per piece between WINTERC-G's knots, at its middle
                 (exact: the density is linear on each piece)
       base      rho_LAB - rho_lid,bottom(x)         (T below the column's own surface)

   which add up to rho_LAB - rho_w C(x), the same contrast the plate's restoring force uses;
4. the reference lithosphere itself, a fixed term;
5. the ocean that follows the geoid (forward/ocean.py).

A sheet sigma (kg/m^2) at radius r(x) gives

    N_lm = 4 pi G R / ((2l + 1) g0) [ sigma(x) (r(x) / R)^(l+2) ]_lm,

which for a constant radius is S_l(r) sigma_lm with S_l(r) = 4 pi G r (r / R)^(l+1) / ((2l + 1) g0), and at
r = R the textbook 3 sigma / ((2l + 1) rho_mean). Degrees 0 and 1 are set to zero (mass conservation and the
centre-of-mass frame). Masses above sea level (r > R) use the same exterior formula: the geoid is then the
exterior potential continued down to R, which is what a geoid computed from a gravity model's coefficients
is too.

Variable radius, exactly (audit E2 / G3 / L10 / G5)
---------------------------------------------------
With eps(x) = ln(r(x) / R) about a centre eps0, (r/R)^(l+2) = e^((l+2) eps0) sum_k ((l+2)(eps - eps0))^k / k!.
So a set of sheets is sum_k diag_k(l) [ F_k(x) u(x) ]_lm with F_k = sum_i Drho_i (eps_i - eps0)^k: one
multiplication and one transform per term, however many sheets, truncated where the next term is below
`tol` at the highest degree. The earlier version put each sheet at its mean depth (the Moho at 22.9 km
everywhere); because a compensated geoid is a small remainder of large terms, that gave errors of 50-70 %
for compensated Moho changes.

Two frames
----------
* Perturbations (lithospheric unknowns, dynamic deflections) move columns that already stand at their
  observed place, so their sheets are at depths below sea level ("sealevel" frame: the surface at the
  observed bed elevation).
* The reference lithosphere is linearised about the undeflected column, whose top is at sea level; its own
  structure and its lift by w_ref use depths below the column's surface ("column" frame).

The deflection is kept on the fine grid until it has been multiplied by the contrasts (audit G2): w's
degrees above lmax carry mass that, multiplied by a laterally varying contrast, lands in degrees <= lmax.

pygeoinf
--------
Every operator maps into the load/deflection sphere space Q (components then in m of geoid); the deflection
operator takes w on the fine space H (or Q).

Usage
-----
    geo = Geoid(mf, ref)                         # mf: MantleFlow, ref: ReferenceLithosphere
    G_m = geo.mantle_operator(X, Q)              # drho_m -> N
    G_L = geo.lithosphere_operator(Q)            # step 3c unknowns -> N
    G_w = geo.deflection_operator(Q, W=H)        # plate deflection w (on H) -> N
    N_ref = geo.reference(Q)                     # the reference lithosphere's own structure
"""

from __future__ import annotations

from math import factorial

import numpy as np
from pygeoinf import RowLinearOperator

from .. import operators as po
from .plate import sample_ocean_function


class Geoid:
    """The geoid operators for one mantle (step 1) and one reference lithosphere (step 3c)."""

    ICE_MIN = 1.0  # m: ice thinner than this (interpolation noise in the data) is treated as none

    def __init__(self, mf, ref, *, ocean=None, tol=1e-10):
        k = mf.kernels
        self.mf, self.ref = mf, ref
        self.R = float(k["R"])
        self.g0 = float(k["g0"])
        self.G = float(mf.shell.constants.G)
        self.r_top = float(k["r_lab"])
        self.rho_lab = float(k["rho_lab"])
        self.rho_water = float(k["rho_water"])
        self.ocean = ocean or sample_ocean_function
        self.tol = float(tol)

    # ---------------------------------------------------------------- sheets
    def sheet_factor(self, Q, radius):
        """S_l(r) per component of Q (m of geoid per kg/m^2); zero at degrees 0 and 1."""
        l = np.asarray(Q.degrees, float)
        r = float(radius)
        S = 4 * np.pi * self.G * r * (r / self.R) ** (l + 1) / ((2 * l + 1) * self.g0)
        return np.where(l >= 2, S, 0.0)

    def sheet(self, Q, radius, field=None):
        """sigma -> N for a sheet at a constant `radius`; with `field`, u -> N for sigma = field(x) u(x)."""
        S = Q.spectral_operator(self.sheet_factor(Q, radius))
        if field is None:
            return S
        H = self.ref._fine(Q)
        up, down = Q.degree_transfer_operator(H), H.degree_transfer_operator(Q)
        return S @ down @ H.multiplication_operator(np.asarray(field, float)) @ up

    def sheets(self, Q, layers, W=None):
        """u -> N for sigma_i = Drho_i(x) u(x) at depth d_i(x) below sea level, summed over i.

        layers: [(depth, contrast)] as arrays on the fine grid of Q. u is in W: Q (default) or the fine space
        H itself. Exact for laterally varying depth (series in ln(r/R) about a common centre, see the module
        docstring)."""
        H = self.ref._fine(Q)
        l = np.asarray(Q.degrees, float)
        eps = [np.log((self.R - np.asarray(d, float)) / self.R) for d, _ in layers]
        lo = min(float(e.min()) for e in eps)
        hi = max(float(e.max()) for e in eps)
        eps0 = 0.5 * (lo + hi)
        x = (Q.lmax + 2) * 0.5 * (hi - lo)
        n_terms = 1
        while x**n_terms / factorial(n_terms) > self.tol and n_terms < 60:
            n_terms += 1
        base = np.where(
            l >= 2,
            4
            * np.pi
            * self.G
            * self.R
            * np.exp((l + 2) * eps0)
            / ((2 * l + 1) * self.g0),
            0.0,
        )
        down = H.degree_transfer_operator(Q)
        up = None if (W is not None and W == H) else Q.degree_transfer_operator(H)
        total = None
        for k in range(n_terms):
            F = sum(
                np.asarray(c, float) * (e - eps0) ** k for (_, c), e in zip(layers, eps)
            )
            op = (
                Q.spectral_operator(base * (l + 2) ** k / factorial(k))
                @ down
                @ H.multiplication_operator(F)
            )
            if up is not None:
                op = op @ up
            total = op if total is None else total + op
        return total

    def _on_fine_grid(self, Q):
        return self.ref._grid(self.ref._fine(Q))

    # ---------------------------------------------------------------- 1. mantle
    def mantle_operator(self, X, Q):
        """drho_m (step 1's density space) -> N: the anomaly and the CMB it deflects."""
        return po.density_operator(X, self.mf.kernels, "N", Q)

    # ---------------------------------------------------------------- 2. lithospheric perturbations
    def lithosphere_depths(self, LA, LO):
        """Local depth (m below sea level) of each step-3c summand's mass, in summand order."""
        s = self.ref.structure(LA, LO)
        out = []
        for label, _, _ in self.ref.summands():
            if label == "dMoho":
                out.append(s["moho"])
            elif label == "dLAB":
                out.append(s["lab"])
            elif label == "drho crust":
                out.append(s["moho"] - s["t_c"] / 2)
            else:
                out.append(s["node_depths"][int(label.rsplit(" ", 1)[1])])
        return out

    def lithosphere_operator(self, Q):
        """(dMoho, dLAB, drho crust, drho lid nodes) -> N, the same summands as ReferenceLithosphere.operator,
        each sheet at its own local depth."""
        LA, LO = self._on_fine_grid(Q)
        fields = self.ref.factor_fields(LA, LO)
        depths = self.lithosphere_depths(LA, LO)
        return RowLinearOperator(
            [self.sheets(Q, [(d, f)]) for f, d in zip(fields, depths)]
        )

    # ---------------------------------------------------------------- 3. the plate's deflection
    def deflection_sheets(self, LA, LO, frame="sealevel"):
        """[(depth, contrast field)] of the column lifted by unit w. Depths below sea level ("sealevel") or
        below the column's own surface ("column")."""
        ref = self.ref
        s = ref.structure(LA, LO)
        C = np.asarray(self.ocean(LA, LO), float)
        ice = ref.sample("ice_thickness", LA, LO)
        a = (ice > self.ICE_MIN).astype(float)
        above = (
            a * ref.RHO_ICE + (1 - a) * self.rho_water * C
        )  # what sits on the bedrock
        base = ref.T - s["bed"]
        out = [
            (-s["bed"] - ice, a * (ref.RHO_ICE - self.rho_water * C)),
            (-s["bed"], s["rho_c"] - above),
            (s["moho"], ref.mantle_density(s["moho"], LA, LO) - s["rho_c"]),
        ]
        for top, bot in ref.lid_intervals(LA, LO):
            out.append(
                (
                    0.5 * (top + bot),
                    ref.mantle_density(bot, LA, LO) - ref.mantle_density(top, LA, LO),
                )
            )
        out.append((base, self.rho_lab - ref.mantle_density(base, LA, LO)))
        if frame == "column":
            out = [(d + s["bed"], c) for d, c in out]
        elif frame != "sealevel":
            raise ValueError(f"unknown frame {frame!r}")
        return out

    def deflection_operator(self, Q, W=None, frame="sealevel"):
        """w (m, in W: Q by default, or the fine space) -> N: the masses moved when the plate deflects."""
        LA, LO = self._on_fine_grid(Q)
        return self.sheets(Q, self.deflection_sheets(LA, LO, frame), W=W)

    # ---------------------------------------------------------------- 4. the reference lithosphere
    def reference_layers(self, LA, LO, n_crust=6, n_piece=3, n_ice=2):
        """[(depth below the column's surface, mass per unit area)] sheets that integrate the reference
        column exactly to quadrature accuracy: ice above the surface, crust, the lid piece by piece (linear
        density, Gauss-Legendre in each piece). Their masses add up to column_mass."""
        ref = self.ref
        s = ref.structure(LA, LO)
        ice = ref.sample("ice_thickness", LA, LO)  # all of it, as column_mass counts it
        out = []
        for n, top, bot, rho in (
            (n_ice, -ice, 0.0 * ice, lambda z: ref.RHO_ICE + 0.0 * z),
            (n_crust, 0.0 * ice, s["t_c"], lambda z: s["rho_c"] + 0.0 * z),
        ):
            x, w = np.polynomial.legendre.leggauss(n)
            for xq, wq in zip(x, w):
                z = top + (bot - top) * (1 + xq) / 2
                out.append((z, wq * (bot - top) / 2 * rho(z)))
        x, w = np.polynomial.legendre.leggauss(n_piece)
        for top, bot in ref.lid_intervals(LA, LO):
            for xq, wq in zip(x, w):
                z = top + (bot - top) * (1 + xq) / 2  # below sea level
                out.append(
                    (z + s["bed"], wq * (bot - top) / 2 * ref.mantle_density(z, LA, LO))
                )
        return out

    def reference(self, Q, **kw):
        """The geoid of the reference lithosphere's own structure (components of Q, m), undeflected: the
        column's top at sea level. Its lift by the reference deflection is the deflection operator in the
        column frame (model.py)."""
        H = self.ref._fine(Q)
        LA, LO = self.ref._grid(H)
        op = self.sheets(Q, self.reference_layers(LA, LO, **kw), W=H)
        return Q.to_components(op(H.from_components(self._ones(H))))

    @staticmethod
    def _ones(H):
        return H.to_components(H.from_grid_values(np.ones(tuple(H.grid_shape))))
