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
