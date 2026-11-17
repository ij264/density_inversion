"""Step 1 of the forward problem: mantle density -> the load that mantle flow puts on the base of the plate.

Physics
-------
The viscous mantle runs from the CMB to a sphere at the reference LAB depth (free slip at both boundaries).
A density anomaly drho_m(r, theta, phi) in it drives instantaneous flow, which puts a normal stress on the
base of the lithosphere. flowkernels solves this for a radial viscosity eta(r), one degree at a time, with the
LAB sphere as the top of its shell. The load on the plate, positive upwards, is

    q_lm = int_{r_cmb}^{r_LAB} K^q_l(r) drho_m,lm(r) dr          [Pa],

    K^q_l(r) = -(eta_ref / r_LAB) (sigma_rr + (rho_w / rho_ref) PHI)   at the top of the shell,

with sigma_rr and PHI the non-dimensional traces of flowkernels (stress positive in tension, physics-sign
potential). The PHI term is the self-gravity of the deflected boundary and of the anomaly, as in flowkernels'
own surface-topography kernel. By construction

    q / ((rho_LAB - rho_w) g_LAB) = flowkernels' surftopo kernel of the truncated shell,

i.e. q is (density contrast) x g x (the deflection the LAB would have with no plate). A dense anomaly gives
q < 0 (the plate is pulled down). Step 2, the elastic plate, turns q into a deflection.

pygeoinf
--------
Model space: drho_m on a layered Sobolev ball (pygeoinf.ball.Layered) from the CMB to the LAB sphere, split at
the viscosity jump so density may jump there (density_inversion.operators.density_space). Codomain: a sphere
space whose components are the orthonormal coefficients of q in Pa (sphere.Sobolev, radius 1, order 2 and
length scale 0.05 R by default). Both spaces are Sobolev. The operator is block-diagonal in degree.

Usage
-----
    from density_inversion.forward.mantle import MantleFlow, viscosity_jump
    mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=40)
    X = mf.density_space()                 # drho_m, kg/m^3
    A = mf.operator(X)                     # drho_m -> q (Pa)
    mf.kernels["K_q"]                      # (degree, node) kernel, Pa per (kg/m^3) per m
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources
from pathlib import Path

import flowkernels as fk
import numpy as np

from .. import operators as po

CACHE = Path(
    os.environ.get(
        "DENSITY_INVERSION_CACHE", Path.home() / ".cache" / "density_inversion"
    )
)


# ----------------------------------------------------------------------------- reference model and viscosity


def reference_model(path=None):
    """The radial reference model (PREM densities on 257 nodes, CMB first; radii in m) as a flowkernels
    ShellModel with constant viscosity and flowkernels' default constants. Read from the package's
    data/reference_model.csv unless another file with the same columns (r, rho) is given.
    """
    if path is None:
        path = resources.files("density_inversion") / "data" / "reference_model.csv"
    r, rho = np.loadtxt(path, delimiter=",", unpack=True)
    return fk.ShellModel(r=r, rho=rho, eta=np.ones_like(r))


def viscosity_jump(factor=30.0, depth=670e3):
    """eta(r) relative to the upper mantle: 1 above `depth`, `factor` below. Returned as a function of radius
    (m) and surface radius R; the jump is put on the nearest model node (666 km for 670), as earth_kernels.py does.
    """

    def eta(r, R):
        r = np.asarray(r, float)
        r_jump = r[np.argmin(np.abs(r - (R - depth)))]
        return np.where(r < r_jump, factor, 1.0)

    eta.label = f"x{factor:g} below {depth / 1e3:g} km"
    return eta


def truncated(model, lab_depth):
    """The same shell, ending at a sphere lab_depth below the surface: nodes above it dropped and one node added
    on it, its density interpolated."""
    R = float(model.r[-1])
    r_top = R - float(lab_depth)
    keep = model.r < r_top - 1e-6 * R
    r = np.append(model.r[keep], r_top)
    rho = np.append(model.rho[keep], np.interp(r_top, model.r, model.rho))
    return (
        fk.ShellModel(
            r=r,
            rho=rho,
            eta=np.ones_like(r),
            compressible=model.compressible,
            constants=model.constants,
        ),
        R,
    )


# ------------------------------------------------------------------------------------------- the operator


class MantleFlow:
    """Mantle density -> load on the base of the plate, for a given radial viscosity and LAB depth."""

    def __init__(
        self,
        viscosity=None,
        lab_depth=105e3,
        lmax=40,
        *,
        model=None,
        cache=True,
    ):
        self.viscosity = viscosity or viscosity_jump()
        self.lab_depth = float(lab_depth)
        self.lmax = int(lmax)
        base = model or reference_model()
        self.shell, self.R = truncated(base, self.lab_depth)
        eta = np.asarray(self.viscosity(self.shell.r, self.R), float)
        self.shell = self.shell.with_viscosity(eta)
        self.kernels = self._kernels(cache)

    # -- kernels
    def _key(self):
        h = hashlib.sha1()
        for a in (self.shell.r, self.shell.rho, self.shell.viscosity(1)):
            h.update(np.ascontiguousarray(a, float).tobytes())
        h.update(
            json.dumps(
                [
                    self.lab_depth,
                    self.lmax,
                    self.shell.compressible,
                    self.shell.layer_quadrature,
                    fk.__version__,
                ]
            ).encode()
        )
        return h.hexdigest()[:16]

    def _kernels(self, cache):
        f = CACHE / f"mantle_{self._key()}.npz"
        if cache and f.exists():
            return dict(np.load(f))
        s, c = self.shell, self.shell.constants
        degrees = np.arange(1, self.lmax + 1)
        r, n = s.r, s.n
        K_q = np.empty((degrees.size, n))
        K_h0 = np.empty((degrees.size, n))
        for j, l in enumerate(degrees):
            u_theta, sig_top, phi_top, sig_bot, phi_bot = fk.solve_degree_traces(
                s, int(l)
            ).T
            K_q[j] = (
                -c.visc_ref / r[-1] * (sig_top + (c.rho_water / c.rho_ref) * phi_top)
            )
            K_h0[j] = K_q[j] / ((s.rho[-1] - c.rho_water) * s.gravity[-1])
        # trapezoid weights, split at the viscosity jump so each side of a density discontinuity is integrated
        # over its own layer: the node on the jump is listed twice (lower copy, then an upper copy nudged up)
        eta = s.viscosity(1)
        i_jump = (
            int(np.flatnonzero(np.diff(eta) != 0)[0]) + 1
            if np.any(np.diff(eta) != 0)
            else None
        )
        if i_jump is None:
            raise ValueError(
                "the viscosity has no jump; density_space needs one to split the ball"
            )
        lower, upper = np.arange(0, i_jump + 1), np.arange(i_jump, n)
        rr = np.concatenate([r[lower], r[upper]])
        rr[lower.size] *= 1 + 1e-12
        w = np.concatenate([_trapezoid(r[lower]), _trapezoid(r[upper])])
        idx = np.concatenate([lower, upper])
        out = dict(
            degrees=degrees,
            r=rr,
            weights=w,
            K_q=K_q[:, idx],
            K_h0=K_h0[:, idx],
            R=np.float64(self.R),
            r_cmb=np.float64(r[0]),
            r_jump=np.float64(r[i_jump]),
            r_lab=np.float64(r[-1]),
            rho_lab=np.float64(s.rho[-1]),
            g_lab=np.float64(s.gravity[-1]),
            rho_water=np.float64(c.rho_water),
            eta=eta[idx],
            notes=np.array(
                "q_lm = sum(weights * K_q[l] * drho_lm(r)) [Pa], positive up; "
                "K_h0 = K_q / ((rho_lab - rho_water) g_lab) [m per kg/m^3 per m]; "
                "see forward/mantle.py"
            ),
        )
        if cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(f, **out)
        return out

    # -- pygeoinf
    def density_space(
        self, *, order=2.0, length_scale=0.05, radial_modes=(16, 10), **options
    ):
        """drho_m on [r_cmb, r_LAB]: a layered Sobolev ball split at the viscosity jump (kg/m^3)."""
        return po.density_space(
            self.kernels,
            self.lmax,
            order=order,
            length_scale=length_scale,
            radial_modes=radial_modes,
            top_depth=self.lab_depth,
            **options,
        )

    def load_space(self, *, order=2.0, length_scale=0.05):
        """q on the sphere, a Sobolev space H^order with the metric (1 + length_scale^2 l(l+1))^order
        (radius 1, length_scale in units of R). Components are the orthonormal coefficients in Pa,
        whatever the order: the order changes the inner product (and so the adjoint), not the vectors.
        """
        from pygeoinf.symmetric_space import sphere

        return sphere.Sobolev(self.lmax, order, length_scale)

    def operator(self, X=None, Q=None):
        """The LinearOperator drho_m -> q."""
        X = X or self.density_space()
        Q = Q or self.load_space()
        return po.density_operator(X, self.kernels, "q", Q)

    def unflexed_deflection(self, q):
        """q (sphere components, Pa) -> the LAB deflection with no plate, q / ((rho_LAB - rho_w) g) (m)."""
        k = self.kernels
        return q / ((float(k["rho_lab"]) - float(k["rho_water"])) * float(k["g_lab"]))


def _trapezoid(x):
    w = np.zeros_like(x)
    d = np.diff(x)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w
