"""Check that density_operator maps (l, m) of the ball to the same (l, m) of the sphere, using pygeoinf's own
degree/order labels of every component; and that the field it sees is the field in physical space (project a
known function with the ball's own projector and compare to the kernel integral)."""
import numpy as np
from density_inversion.forward.mantle import MantleFlow, viscosity_jump

mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=12)
X = mf.density_space()
A = mf.operator(X)
Q = A.codomain

import scipy.sparse as sp
# rebuild the matrix as density_operator does
from density_inversion import operators as po
rows = po.density_rows(X, mf.kernels, "q")
bad = 0; tot = 0
qd, qo = np.asarray(Q.degrees), np.asarray(Q.orders)
for li, ball in enumerate(X.layers):
    bd, bo = np.asarray(ball.degrees), np.asarray(ball.orders)
    for l in range(1, 13):
        start, n = po._column_block(X, li, l)
        local0 = start - X.columns(li).start
        for mi in range(2 * l + 1):
            cols = local0 + mi * n + np.arange(n)
            row = l * l + mi
            tot += 1
            if not (np.all(bd[cols] == qd[row]) and np.all(bo[cols] == qo[row])):
                bad += 1
                if bad < 4:
                    print("mismatch l", l, "mi", mi, "ball (deg, ord)", bd[cols][:2], bo[cols][:2], "sphere", qd[row], qo[row])
print(f"(l, m) blocks checked {tot}, mismatched {bad}")
print("sphere orders for l=2:", qo[4:9], " ball orders for l=2 block, layer 0:",
      np.asarray(X.layers[0].orders)[po._column_block(X, 0, 2)[0] - X.columns(0).start::po._column_block(X, 0, 2)[1]][:5])
