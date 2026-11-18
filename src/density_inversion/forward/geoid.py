"""Step 5 of the forward problem: the geoid, from every mass the model moves.

Physics
-------
The geoid is N = -Phi / g0 at the surface (physics-sign potential, nabla^2 Phi = 4 pi G rho; g0 surface
gravity), so a mass excess raises it. Four kinds of mass contribute, each at its own radius:

1. the mantle density anomaly drho_m and the deflection of the CMB it drives: from flowkernels, with the
   self-gravity of the deflected top of the shell taken out (step 1's kernel K_N; see forward/mantle.py);
2. the lithospheric perturbations of step 3 (crust and lid densities, Moho and LAB displacements), each a
   mass sheet sigma(x) = f(x) u(x) at its mean depth, f the same factor field that makes its load;
3. the plate's deflection w: lifting the whole reference column by w puts a sheet Drho_i(x) w(x) on every
   density contrast i of the column, from the surface to the base of the plate,

       surface   rho_c(x) - rho_w C(x)             (rock for water at sea, for air on land)
       Moho      rho_lid,top(x) - rho_c(x)
       lid       rho_lid,bottom(x) - rho_lid,top(x)  (the lid's own density gradient, at mid-lid)
       base      rho_LAB - rho_lid,bottom(x)        (at the flow top)

   which add up to rho_LAB - rho_w C(x), the same contrast the plate's restoring force uses;
4. the reference lithosphere itself (its departure from the spherical mean), a fixed term.

A sheet sigma_lm (kg/m^2) at radius r gives

    N_lm = S_l(r) sigma_lm,     S_l(r) = 4 pi G r (r / R)^(l+1) / ((2l + 1) g0),

which at r = R is the textbook 3 sigma / ((2l + 1) rho_mean). Degrees 0 and 1 are set to zero (mass
conservation and the centre-of-mass frame).

Approximations (first order): a sheet on a laterally varying interface (Moho, LAB, nodes) is placed at the
interface's mean radius; the mantle part keeps flowkernels' self-gravitating flow, which saw a freely deflecting
top rather than the plate.

pygeoinf
--------
Every operator maps into the load/deflection sphere space Q (components then in m of geoid). The total is

    N = G_m drho_m + G_L x_L + G_w w + N_ref,     w = P (A_m drho_m + A_L x_L + q_ref).

Usage
-----
    geo = Geoid(mf, ref)                   # mf: MantleFlow, ref: ReferenceLithosphere
    G_m = geo.mantle_operator(X, Q)        # drho_m -> N
    G_L = geo.lithosphere_operator(Q)      # step 3c unknowns -> N
    G_w = geo.deflection_operator(Q)       # plate deflection w -> N
    N_ref = geo.reference(Q)               # components of the reference lithosphere's geoid
"""

from __future__ import annotations

import numpy as np
from pygeoinf import RowLinearOperator

from .. import operators as po
from .plate import sample_ocean_function


class Geoid:
    """The geoid operators for one mantle (step 1) and one reference lithosphere (step 3c)."""

    def __init__(self, mf, ref, *, ocean=None):
        k = mf.kernels
        self.mf, self.ref = mf, ref
        self.R = float(k["R"])
        self.g0 = float(k["g0"])
        self.G = float(mf.shell.constants.G)
        self.r_top = float(k["r_lab"])
        self.rho_lab = float(k["rho_lab"])
        self.rho_water = float(k["rho_water"])
        self.ocean = ocean or sample_ocean_function

    # ---------------------------------------------------------------- sheets
    def sheet_factor(self, Q, radius):
        """S_l(r) per component of Q (m of geoid per kg/m^2); zero at degrees 0 and 1."""
        l = np.asarray(Q.degrees, float)
        r = float(radius)
        S = 4 * np.pi * self.G * r * (r / self.R) ** (l + 1) / ((2 * l + 1) * self.g0)
        return np.where(l >= 2, S, 0.0)

    def sheet(self, Q, radius, field=None):
        """sigma -> N for a sheet at `radius`; with `field`, u -> N for sigma = field(x) u(x)."""
        S = Q.spectral_operator(self.sheet_factor(Q, radius))
        if field is None:
            return S
        H = self.ref._fine(Q)
        up, down = Q.degree_transfer_operator(H), H.degree_transfer_operator(Q)
        return S @ down @ H.multiplication_operator(np.asarray(field, float)) @ up

    def _on_fine_grid(self, Q):
        return self.ref._grid(self.ref._fine(Q))

    # ---------------------------------------------------------------- 1. mantle
    def mantle_operator(self, X, Q):
        """drho_m (step 1's density space) -> N: the anomaly and the CMB it deflects."""
        return po.density_operator(X, self.mf.kernels, "N", Q)

    # ---------------------------------------------------------------- 2. lithospheric perturbations
    def lithosphere_depths(self):
        """Mean depth (m below sea level) of each step-3c summand's mass, in summand order."""
        LA, LO = np.meshgrid(self.ref.data["lat"], self.ref.data["lon"], indexing="ij")
        s = self.ref.structure(LA, LO)
        w = np.cos(np.radians(LA))
        mean = lambda f: float(np.sum(w * f) / np.sum(w))
        out = []
        for label, _, _ in self.ref.summands():
            if label == "dMoho":
                out.append(mean(s["moho"]))
            elif label == "dLAB":
                out.append(mean(s["lab"]))
            elif label == "drho crust":
                out.append(mean(s["moho"] - s["t_c"] / 2))
            else:
                out.append(mean(s["node_depths"][int(label.rsplit(" ", 1)[1])]))
        return out

    def lithosphere_operator(self, Q):
        """(dMoho, dLAB, drho crust, drho lid nodes) -> N, the same summands as ReferenceLithosphere.operator."""
        fields = self.ref.factor_fields(*self._on_fine_grid(Q))
        depths = self.lithosphere_depths()
        return RowLinearOperator(
            [self.sheet(Q, self.R - d, f) for f, d in zip(fields, depths)]
        )

    # ---------------------------------------------------------------- 3. the plate's deflection
    def deflection_sheets(self, LA, LO):
        """[(depth below sea level, contrast field)] of the reference column lifted by unit w."""
        s = self.ref.structure(LA, LO)
        C = np.asarray(self.ocean(LA, LO), float)
        rho_top = self.ref.sample("rho_mantle", LA, LO)[0]
        rho_bot = self.ref.mantle_density(self.ref.T, LA, LO)
        w = np.cos(np.radians(LA))
        mean = lambda f: float(np.sum(w * f) / np.sum(w))
        moho = mean(s["moho"])
        return [
            (0.0, s["rho_c"] - self.rho_water * C),
            (moho, rho_top - s["rho_c"]),
            (0.5 * (moho + self.ref.T), rho_bot - rho_top),
            (self.ref.T, self.rho_lab - rho_bot),
        ]

    def deflection_operator(self, Q):
        """w (m) -> N: the masses moved when the plate deflects."""
        LA, LO = self._on_fine_grid(Q)
        ops = [self.sheet(Q, self.R - d, f) for d, f in self.deflection_sheets(LA, LO)]
        total = ops[0]
        for op in ops[1:]:
            total = total + op
        return total

    # ---------------------------------------------------------------- 4. the reference lithosphere
    def reference(self, Q, n_layers=120):
        """The geoid of the reference lithosphere's departure from its global mean, depth by depth
        (components of Q, m). Midpoint sheets every T / n_layers in each column, plus the ice at the surface.
        """
        ref = self.ref
        H = ref._fine(Q)
        LA, LO = ref._grid(H)
        s = ref.structure(LA, LO)
        wlat = np.cos(np.radians(LA))
        out = np.zeros(Q.dim)
        dz = ref.T / n_layers
        for z in (
            np.arange(n_layers) + 0.5
        ) * dz:  # depth below the column's own surface
            rho = np.where(
                z < s["t_c"], s["rho_c"], ref.mantle_density(z - s["bed"], LA, LO)
            )
            sigma = dz * (rho - np.sum(wlat * rho) / np.sum(wlat))
            comp = H.to_components(H.from_grid_values(sigma))[: Q.dim]
            out += self.sheet_factor(Q, self.R - z) * comp
        ice = ref.RHO_ICE * ref.sample("ice_thickness", LA, LO)
        out += (
            self.sheet_factor(Q, self.R)
            * H.to_components(H.from_grid_values(ice))[: Q.dim]
        )
        return out
