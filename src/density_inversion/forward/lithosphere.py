"""Step 3 of the forward problem: lithospheric loads, layer densities and interfaces -> load on the plate.

Physics
-------
The lithosphere is a stack of layers (sediment, upper crust, lower crust, lithospheric mantle) between the
surface and the LAB. Two kinds of perturbation change the mass of a column:

* a density perturbation drho_j(r, x) inside layer j;
* a perturbation dr_i(x) of interface i (positive up), which swaps the material above it for the material
  below it over a height dr_i: extra mass Drho_i dr_i per unit area, Drho_i = rho_below - rho_above.

The plate is thin, so its load is the weight of the extra mass in each column, wherever the mass sits:

    q_L(x) = -g [ sum_j int_layer drho_j dr  +  sum_i Drho_i dr_i ]          [Pa, positive up].

Extra mass pushes the plate down. Products of perturbations are dropped (first order).

The interface perturbations are changes relative to the plate: thickening the crust by moving the Moho down
by r is dr_Moho = -r, and the plate's deflection then moves the whole column, interfaces included. With no
plate this reproduces Airy isostasy (see the tests).

Nodal fields
------------
Within layer j the density perturbation is nodal in radius: drho_j(r, x) = sum_n drho_jn(x) l_n(r), with
l_n the Lagrange polynomials on n Gauss-Lobatto-Legendre points across the layer. Each nodal value
drho_jn(x) is a field on the sphere (kg/m^3). A layer with one node is uniform in depth. The column mass is

    int_layer drho_j dr = sum_n w_jn drho_jn(x),       w_jn = the GLL weights scaled to the layer (m),

exact because l_n has degree n - 1. Each nodal value is also a mass sheet at its node's radius, which is how
step 5 (the geoid) will see it.

Stages
------
3a  interface perturbations, constant contrasts Drho_i:   q = -g Drho_i dr_i, degree by degree.
3b  layer density perturbations on the mean layer thicknesses:   q = -g sum_n w_jn drho_jn.
3c  (later) laterally varying reference thicknesses and contrasts: multiplication operators on the sphere.
3d  (later) the load of the reference structure itself, a fixed term.

The mean structure here is a PLACEHOLDER for tests (round numbers near PREM's crust); 3c and 3d replace it with
the reference lithosphere model.

pygeoinf
--------
Model space: a direct sum of sphere Sobolev spaces, one per perturbed interface (m) and one per layer node
(kg/m^3), labelled. Codomain: the load space of step 1 (Pa). The operator is a RowLinearOperator, each block
a multiple of the identity, so it adds to step 1's load before the plate:

    h = P (A_mantle drho_m + A_lith x_L).

Usage
-----
    from density_inversion.forward.lithosphere import Lithosphere
    lith = Lithosphere.placeholder(mf)            # four layers, perturbed sediment base, Moho and LAB
    L = lith.operator(mf.load_space())            # (dr_i, drho_jn) -> q (Pa)
    L.domain.labels                               # names of the summands
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre
from pygeoinf import RowLinearOperator


@dataclass(frozen=True)
class Layer:
    """A lithospheric layer between two depths (m below the surface), with its mean density (kg/m^3) and the
    number of GLL nodes across it (1 = uniform in depth)."""

    name: str
    top: float
    bottom: float
    density: float
    nodes: int = 1

    @property
    def thickness(self):
        return self.bottom - self.top


def gll(n):
    """Gauss-Lobatto-Legendre points and weights on [-1, 1]; n = 1 gives the midpoint with weight 2."""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n < 1:
        raise ValueError("a layer needs at least one node")
    P = legendre.Legendre.basis(n - 1)
    x = np.concatenate([[-1.0], np.sort(P.deriv().roots().real), [1.0]])
    w = 2.0 / (n * (n - 1) * P(x) ** 2)
    return x, w


def lagrange(nodes, x):
    """Values of the Lagrange polynomials on `nodes` at points x: array (len(x), len(nodes))."""
    nodes, x = np.asarray(nodes, float), np.atleast_1d(np.asarray(x, float))
    out = np.ones((x.size, nodes.size))
    for j, xj in enumerate(nodes):
        for k, xk in enumerate(nodes):
            if k != j:
                out[:, j] *= (x - xk) / (xj - xk)
    return out


class Lithosphere:
    """Layer densities and interface perturbations -> load on the plate, on the mean layer thicknesses."""

    def __init__(self, layers, below_density, g, R, *, interfaces=None):
        """layers: Layer objects from the top down, each starting where the last ends; below_density: the
        density under the bottom layer (the mantle at the LAB); interfaces: names of the interfaces that may
        move, each the base of a layer, e.g. "base of sediment" (default: every base but the surface).
        """
        self.layers = list(layers)
        for a, b in zip(self.layers, self.layers[1:]):
            if not np.isclose(a.bottom, b.top):
                raise ValueError(f"{a.name} and {b.name} do not meet")
        self.below_density, self.g, self.R = float(below_density), float(g), float(R)
        names = [f"base of {layer.name}" for layer in self.layers]
        self.interface_names = list(interfaces) if interfaces is not None else names
        unknown = set(self.interface_names) - set(names)
        if unknown:
            raise ValueError(f"no such interface: {sorted(unknown)}")

    # ---------------------------------------------------------------- structure
    @classmethod
    def placeholder(cls, mf, *, lab_nodes=3, lithospheric_mantle_density=3340.0, **kw):
        """A round-number structure for tests, its LAB at step 1's flow top. Sediment 0-1 km at 2300,
        upper crust 1-15 km at 2700, lower crust 15-35 km at 2900, lithospheric mantle to the LAB.
        """
        k = mf.kernels
        lab = float(k["R"]) - float(k["r_lab"])
        layers = [
            Layer("sediment", 0.0, 1e3, 2300.0),
            Layer("upper crust", 1e3, 15e3, 2700.0),
            Layer("lower crust", 15e3, 35e3, 2900.0),
            Layer(
                "lithospheric mantle", 35e3, lab, lithospheric_mantle_density, lab_nodes
            ),
        ]
        kw.setdefault(
            "interfaces",
            ["base of sediment", "base of lower crust", "base of lithospheric mantle"],
        )
        return cls(layers, float(k["rho_lab"]), float(k["g_lab"]), float(k["R"]), **kw)

    def interface(self, name):
        """(depth (m), Drho = rho_below - rho_above) of an interface."""
        i = [f"base of {layer.name}" for layer in self.layers].index(name)
        above = self.layers[i].density
        below = (
            self.layers[i + 1].density
            if i + 1 < len(self.layers)
            else self.below_density
        )
        return self.layers[i].bottom, below - above

    def nodes(self, layer):
        """(depths (m), weights (m)) of a layer's GLL nodes, from the top down."""
        x, w = gll(layer.nodes)
        h = layer.thickness / 2
        return layer.top + h * (1 + x), h * w

    def summands(self):
        """[(label, kind, depth (m), factor)]: one per summand of the model space. The factor turns the
        summand into a surface density (kg/m^2): Drho_i for an interface, the node's weight for a layer.
        """
        out = []
        for name in self.interface_names:
            depth, drho = self.interface(name)
            out.append((f"d{name}", "interface", depth, drho))
        for layer in self.layers:
            depths, weights = self.nodes(layer)
            for n, (d, w) in enumerate(zip(depths, weights)):
                label = f"drho {layer.name}" + (f" node {n}" if layer.nodes > 1 else "")
                out.append((label, "layer", d, w))
        return out

    # ---------------------------------------------------------------- pygeoinf
    def operator(self, Q):
        """(dr_i, drho_jn) -> q (Pa), from a direct sum of copies of Q to Q."""
        blocks = [
            Q.spectral_operator(np.full(Q.dim, -self.g * f))
            for *_, f in self.summands()
        ]
        return RowLinearOperator(blocks)

    def space(self, Q):
        """The model space: the operator's domain, with its summands labelled."""
        from pygeoinf import DirectSum

        return DirectSum(
            [Q] * len(self.summands()), labels=[s[0] for s in self.summands()]
        )

    def index(self, label):
        return [s[0] for s in self.summands()].index(label)


# ===================================================================================== 3c and 3d: WINTERC-G
#
# The reference lithosphere is WINTERC-G v5.4 (Fullea et al. 2021), built into data/reference_lithosphere.npz
# by scripts/make_reference_lithosphere.py. It has one crustal layer (bedrock to Moho, one density per column)
# and mantle densities at fixed depths. Step 3 owns the column from the surface to the flow top of step 1
# (a sphere T = 105 km down); everything below is step 1's. So the layers are
#
#     crust        surface to Moho(x)          thickness t_c(x) = Moho depth + bedrock elevation
#     mantle lid   Moho(x) to T below the      thickness t_l(x) = T - t_c(x), GLL nodes across it
#                  surface
#
# and the perturbed interfaces are the Moho (contrast rho_lid_top(x) - rho_crust(x)) and WINTERC-G's LAB
# (its density jump drho_lab(x), about 8 kg/m^3 on average), the LAB a mass sheet at its own depth even where it
# lies below T (first order). Columns are measured from their own surface, as the plate sees them; depths
# into WINTERC-G's density tables are depths below sea level, d = z - bedrock elevation.
#
# 3c: each perturbation's load is a field times the perturbation, -g f(x) u(x), a multiplication on the sphere.
#     It is done on a Gauss-Legendre grid at quad_lmax (3 lmax by default) and truncated, as the plate does.
# 3d: the reference structure's own load, q_ref = -g (M(x) - mean M), M the column mass to T plus the ice.


# ---------------------------------------------------------------- band-limited 1 degree fields
# The data are 1 degree cell values: steps at every cell edge, with power at every degree. Sampled point by
# point on a Gauss-Legendre grid they alias (audit L3 / P9: 3 % of the reference load, 8 % in its upper
# degrees, and no monotone convergence with the grid). Instead each field is projected onto spherical
# harmonics up to the data's own resolution (degree 89; the projection is done by quadrature on a 0.25 degree
# Gauss-Legendre grid, accurate to about 1e-3 at the coastlines), tapered from degree TAPER_FROM to 89 to
# soften the ringing at sharp edges, and then evaluated on whatever Gauss-Legendre grid the model works on.
# Products of band-limited fields are then integrated exactly once the grid is fine enough, and degrees
# <= 2 lmax of each field are all that a product can bring into degrees <= lmax, so nothing the model
# keeps is changed by the taper for lmax <= 35, and degrees above 70 are touched only by the taper.
BANDLIMIT_LMAX = 89
TAPER_FROM = 70
_PROJECTION_LMAX = 359


def _taper():
    l = np.arange(BANDLIMIT_LMAX + 1)
    t = np.clip((l - TAPER_FROM) / (BANDLIMIT_LMAX - TAPER_FROM), 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * t))


def project_cells(grid):
    """A 180 x 360 grid of 1 degree cell values (rows from 89.5N, columns from 0.5E) -> tapered spherical-
    harmonic coefficients (pyshtools 4-pi normalisation) up to BANDLIMIT_LMAX."""
    import pyshtools as pysh

    grid = np.asarray(grid, float)
    lat, lon = pysh.expand.GLQGridCoord(_PROJECTION_LMAX)
    i = np.clip(np.floor(90.0 - lat).astype(int), 0, grid.shape[0] - 1)
    j = np.floor(np.mod(lon, 360.0)).astype(int) % grid.shape[1]
    fine = grid[i[:, None], j[None, :]]
    zero, w = pysh.expand.SHGLQ(_PROJECTION_LMAX)
    cilm = pysh.expand.SHExpandGLQ(fine, w=w, zero=zero, lmax_calc=BANDLIMIT_LMAX)
    return cilm * _taper()[None, :, None]


def glq_degree(lat, lon):
    """n if (lat, lon) is the full Gauss-Legendre grid of degree n (as meshgrid, indexing 'ij'), else None."""
    import pyshtools as pysh

    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim != 2 or lat.shape != lon.shape:
        return None
    n = lat.shape[0] - 1
    if lat.shape[1] != 2 * n + 1:
        return None
    glat, glon = pysh.expand.GLQGridCoord(n)
    if np.allclose(lat[:, 0], glat) and np.allclose(lon[0], glon):
        return n
    return None


def synthesise(cilm, n):
    """Band-limited coefficients -> values on the Gauss-Legendre grid of degree n."""
    import pyshtools as pysh

    zero, _ = pysh.expand.SHGLQ(n)
    c = np.zeros((2, n + 1, n + 1))
    m = min(n, cilm.shape[1] - 1)
    c[:, : m + 1, : m + 1] = cilm[:, : m + 1, : m + 1]
    return pysh.expand.MakeGridGLQ(c, zero=zero, lmax=n)


def reference_data(path=None):
    """The reference lithosphere fields (see scripts/make_reference_lithosphere.py)."""
    from importlib import resources

    if path is None:
        path = (
            resources.files("density_inversion") / "data" / "reference_lithosphere.npz"
        )
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


class ReferenceLithosphere:
    """Laterally varying lithosphere from WINTERC-G: perturbation operators (3c) and the reference load (3d)."""

    RHO_ICE = 910.0  # WINTERC-G's README (layer 2) gives 910; Fullea et al. (2021, s4.3) quote 950 (audit L6)

    def __init__(
        self,
        g,
        R,
        lmax,
        flow_top_depth,
        *,
        lid_nodes=3,
        quad_lmax=None,
        data=None,
        lab_interface=False,
        bandlimit=True,
    ):
        self.g, self.R, self.lmax = float(g), float(R), int(lmax)
        self.T = float(flow_top_depth)
        self.lid_nodes = int(lid_nodes)
        self.quad_lmax = int(quad_lmax or 3 * self.lmax)
        # dLAB, the LAB as a displaceable interface, is off by default (audit L2 / fix 6): where WINTERC-G's
        # LAB lies below the lid (28 % of the surface) its mass is inside the flow domain and was loaded as if
        # it were in the plate (5-35 % too much); where it lies inside the lid it duplicates the lid nodes.
        # The lid's density at its nodes carries the LAB's effect above T; the mantle unknown carries it below.
        self.lab_interface = bool(lab_interface)
        self.data = data or reference_data()
        self.bandlimit = bool(bandlimit)
        self._cilm = {}

    @classmethod
    def from_mantle(cls, mf, **kw):
        k = mf.kernels
        return cls(
            float(k["g_lab"]),
            float(k["R"]),
            mf.lmax,
            float(k["R"]) - float(k["r_lab"]),
            **kw,
        )

    # ---------------------------------------------------------------- fields at points
    def sample(self, name, lat, lon):
        """A field at points (degrees). On a full Gauss-Legendre grid (the model's grids) the band-limited
        field (see project_cells); elsewhere the value of the 1 degree cell each point falls in.
        """
        n = glq_degree(lat, lon) if self.bandlimit else None
        if n is not None:
            return self._sample_bandlimited(name, n)
        return self._sample_cells(name, lat, lon)

    def _coefficients(self, name, layer=None):
        key = (name, layer)
        if key not in self._cilm:
            f = self.data[name] if layer is None else self.data[name][layer]
            self._cilm[key] = project_cells(f)
        return self._cilm[key]

    def _sample_bandlimited(self, name, n):
        f = self.data[name]
        if name == "mantle_depths":
            # knot depths are max(constant, Moho): rebuilt from the band-limited Moho so they stay ordered
            moho = self._sample_bandlimited("moho_depth", n)
            consts = [float(np.max(f[k])) for k in range(1, f.shape[0])]
            return np.stack([moho] + [np.maximum(c, moho) for c in consts])
        if f.ndim == 3:
            return np.stack(
                [synthesise(self._coefficients(name, k), n) for k in range(f.shape[0])]
            )
        return synthesise(self._coefficients(name), n)

    def _sample_cells(self, name, lat, lon):
        f = self.data[name]
        i = np.clip(np.floor(90.0 - np.asarray(lat)).astype(int), 0, 179)
        j = np.floor(np.mod(np.asarray(lon), 360.0)).astype(int) % 360
        return np.asarray(f[..., i, j], float)

    def ocean(self, lat, lon):
        """The ocean function the model uses: the present-day one, and 1 where the ice floats (a floating
        shelf and its cavity weigh what open water does; audit L5)."""
        from .plate import sample_ocean_function

        n = glq_degree(lat, lon) if self.bandlimit else None
        if n is not None:
            if ("ocean", None) not in self._cilm:
                from .plate import ocean_function

                self._cilm[("ocean", None)] = project_cells(ocean_function()[2])
            C = synthesise(self._cilm[("ocean", None)], n)
        else:
            C = np.asarray(sample_ocean_function(lat, lon), float)
        if "floating_ice" not in self.data:
            return C
        f = self.sample("floating_ice", lat, lon)
        return C + f * (1.0 - C)

    def mantle_density(self, depth, lat, lon):
        """WINTERC-G mantle density at depth below sea level (m), linear between its knots, per column."""
        knots = self.sample("mantle_depths", lat, lon)  # (6, ...)
        rho = self.sample("rho_mantle", lat, lon)
        depth = np.broadcast_to(np.asarray(depth, float), knots.shape[1:])
        out = np.where(depth <= knots[0], rho[0], rho[-1])
        for k in range(knots.shape[0] - 1):
            a, b = knots[k], knots[k + 1]
            inside = (depth > a) & (depth <= b) & (b > a)
            frac = np.where(b > a, (depth - a) / np.where(b > a, b - a, 1.0), 0.0)
            out = np.where(inside, rho[k] + frac * (rho[k + 1] - rho[k]), out)
        return out

    def structure(self, lat, lon):
        """Thicknesses, contrasts and node weights at points: a dict of arrays."""
        bed = self.sample("bed_elevation", lat, lon)
        moho = self.sample("moho_depth", lat, lon)
        t_c = moho + bed
        t_l = self.T - t_c
        if np.any(t_l <= 0):
            raise ValueError("the crust reaches the flow top somewhere")
        rho_c = self.sample("rho_crust", lat, lon)
        rho_top = self.sample("rho_mantle", lat, lon)[0]
        x, w = gll(self.lid_nodes)
        return dict(
            bed=bed,
            moho=moho,
            t_c=t_c,
            t_l=t_l,
            rho_c=rho_c,
            drho_moho=rho_top - rho_c,
            drho_lab=self.sample("drho_lab", lat, lon),
            lab=self.sample("lab_depth", lat, lon),
            node_weights=[t_l / 2 * wn for wn in w],
            node_depths=[moho + t_l / 2 * (1 + xn) for xn in x],  # below sea level
        )

    def lid_intervals(self, lat, lon):
        """[(top, bottom)] depths below sea level (m) of the pieces of the lid between WINTERC-G's density
        knots, clipped to the lid (Moho to T below the column's surface). The density is linear on each
        piece, so integrals over a piece are exact with its end values. The pieces cover the lid exactly
        (some have zero thickness)."""
        s = self.structure(lat, lon)
        top, bottom = s["moho"], self.T - s["bed"]
        knots = self.sample("mantle_depths", lat, lon)
        edges = [top] + [np.clip(k, top, bottom) for k in knots] + [bottom]
        return [(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def column_mass(self, lat, lon, n_quad=None):
        """Mass per unit area from the surface to T below it, plus the ice (kg/m^2).

        Exact: the lid's density is piecewise linear between knots, integrated piece by piece (audit G4;
        n_quad is accepted for compatibility and ignored)."""
        s = self.structure(lat, lon)
        lid = np.zeros_like(s["t_l"])
        for a, b in self.lid_intervals(lat, lon):
            lid += (
                (b - a)
                * 0.5
                * (self.mantle_density(a, lat, lon) + self.mantle_density(b, lat, lon))
            )
        ice = self.RHO_ICE * self.sample("ice_thickness", lat, lon)
        return s["rho_c"] * s["t_c"] + lid + ice

    # ---------------------------------------------------------------- pygeoinf
    def _fine(self, Q):
        from pygeoinf.symmetric_space import sphere

        if isinstance(Q, sphere.Sobolev):
            return sphere.Sobolev(self.quad_lmax, Q.order, Q.length_scale, grid="GLQ")
        return sphere.Lebesgue(self.quad_lmax, grid="GLQ")

    def _grid(self, H):
        import pyshtools as pysh

        lat, lon = pysh.expand.GLQGridCoord(H.lmax)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        if LA.shape != tuple(H.grid_shape):
            raise ValueError("unexpected grid layout")
        return LA, LO

    def summands(self):
        """[(label, kind, field name)]: dMoho, (dLAB if lab_interface), the crust's density, the lid's nodes."""
        out = [("dMoho", "interface", "drho_moho")]
        if self.lab_interface:
            out.append(("dLAB", "interface", "drho_lab"))
        out.append(("drho crust", "layer", "t_c"))
        out += [
            (f"drho lid node {n}", "layer", ("node_weights", n))
            for n in range(self.lid_nodes)
        ]
        return out

    def factor_fields(self, lat, lon):
        """The field f(x) of each summand (kg/m^2 per unit of the summand), in summand order."""
        s = self.structure(lat, lon)
        out = []
        for _, _, key in self.summands():
            out.append(s[key[0]][key[1]] if isinstance(key, tuple) else s[key])
        return out

    def operator(self, Q):
        """(dMoho, dLAB, drho crust, drho lid nodes) -> q (Pa): a RowLinearOperator of multiplications."""
        if not np.isclose(Q.radius, 1.0):
            raise ValueError("expected a sphere space of radius 1")
        H = self._fine(Q)
        up, down = Q.degree_transfer_operator(H), H.degree_transfer_operator(Q)
        fields = self.factor_fields(*self._grid(H))
        return RowLinearOperator(
            [down @ H.multiplication_operator(-self.g * f) @ up for f in fields]
        )

    def space(self, Q):
        from pygeoinf import DirectSum

        return DirectSum(
            [Q] * len(self.summands()), labels=[s[0] for s in self.summands()]
        )

    def index(self, label):
        return [s[0] for s in self.summands()].index(label)

    def reference_load(self, Q):
        """3d: q_ref = -g (M - mean M), the load of the reference structure itself, as components of Q (Pa)."""
        return self._analyse(
            Q, lambda LA, LO: -self.g * self.column_mass(LA, LO), zero_mean=True
        )

    def field(self, Q, name):
        """A stored 2-D field, or 'bed_elevation' etc., as components of Q (degree <= Q.lmax)."""
        return self._analyse(Q, lambda LA, LO: self.sample(name, LA, LO))

    def _analyse(self, Q, f, zero_mean=False):
        H = self._fine(Q)
        values = f(*self._grid(H))
        comp = H.to_components(H.from_grid_values(values))[: Q.dim].copy()
        if zero_mean:
            comp[0] = 0.0
        return comp
