"""Sensitivity of K_N and K_q to the core density used in the CMB rows (9900 = PREM just below the CMB;
11085 = mean density implied by gm_core). Uses a separate cache directory."""
import dataclasses, os, tempfile
os.environ["DENSITY_INVERSION_CACHE"] = tempfile.mkdtemp()
import numpy as np
import flowkernels as fk
from density_inversion.forward.mantle import MantleFlow, viscosity_jump, reference_model
base = reference_model()
out = {}
for rc in (9900.0, 11085.0):
    m = dataclasses.replace(base, constants=dataclasses.replace(base.constants, rho_core=rc))
    out[rc] = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=10, model=m, cache=False).kernels
for l in (2, 3, 4, 6, 10):
    for name in ("K_N", "K_q"):
        a, b = out[9900.0][name][l-1], out[11085.0][name][l-1]
        print(f"l={l} {name}: max|diff|/max|K| = {np.max(abs(a-b))/np.max(abs(a)):.1e}")
