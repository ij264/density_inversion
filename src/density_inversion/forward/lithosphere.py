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

    RHO_ICE = 910.0

    def __init__(
        self, g, R, lmax, flow_top_depth, *, lid_nodes=3, quad_lmax=None, data=None
    ):
        self.g, self.R, self.lmax = float(g), float(R), int(lmax)
        self.T = float(flow_top_depth)
        self.lid_nodes = int(lid_nodes)
        self.quad_lmax = int(quad_lmax or 3 * self.lmax)
        self.data = data or reference_data()

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
        """A 2-D field at points (degrees), by the 1 degree cell each point falls in."""
        f = self.data[name]
        i = np.clip(np.floor(90.0 - np.asarray(lat)).astype(int), 0, 179)
        j = np.floor(np.mod(np.asarray(lon), 360.0)).astype(int) % 360
        return np.asarray(f[..., i, j], float)

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

    def column_mass(self, lat, lon, n_quad=40):
        """Mass per unit area from the surface to T below it, plus the ice (kg/m^2)."""
        s = self.structure(lat, lon)
        xg, wg = np.polynomial.legendre.leggauss(n_quad)
        lid = np.zeros_like(s["t_l"])
        for xq, wq in zip(xg, wg):
            z = s["t_c"] + s["t_l"] / 2 * (1 + xq)  # depth below the surface
            lid += wq * s["t_l"] / 2 * self.mantle_density(z - s["bed"], lat, lon)
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
        """[(label, kind, field name)]: dMoho, dLAB, the crust's density, the lid's nodes."""
        out = [
            ("dMoho", "interface", "drho_moho"),
            ("dLAB", "interface", "drho_lab"),
            ("drho crust", "layer", "t_c"),
        ]
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
