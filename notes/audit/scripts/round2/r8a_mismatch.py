"""Item 8a: size of the surface-mass mismatch between flowkernels' free top (mass -sigma/g, which its flow
self-gravitates) and the package's plate (Te=0: mass q/g = -sigma/g - rho_w N_fk; Te=40: flexure)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.plate import ElasticPlate
mf = MantleFlow(lmax=50); k = mf.kernels
w = k["weights"]
rel = lambda a, b: np.sqrt(np.sum(w * a**2) / np.sum(w * b**2))
p0 = ElasticPlate.from_mantle(mf, Te=0.0).filter(); p40 = ElasticPlate.from_mantle(mf, Te=40e3).filter()
rho_bar = 5.974e24 / (4/3*np.pi*float(k["R"])**3)
print(" l  |rho_w N_fk|/|m_top|   1-F40/F0   self-grav factor 3drho/((2l+1)rho_bar)  -> est. load error (land, Te=0 | flexure)")
for l in (2, 3, 4, 6, 10, 20, 30, 50):
    j = l - 1
    a = rel(float(k["rho_water"]) * k["K_N_fk"][j], k["K_m_top"][j])
    fl = 1 - p40[l] / p0[l]
    sg = 3 * (float(k["rho_lab"]) - float(k["rho_water"])) / ((2*l+1) * rho_bar)
    print(f"{l:2d}  {a:.4f}               {fl:.2e}   {sg:.3f}                                   {a*sg*1.4:.4f} | {fl*sg*1.4:.4f}")
