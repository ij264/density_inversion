#!/bin/sh
# Robustness batch for the single-scaling rerun (written by Claude, 19 Nov 2026).
PY="$HOME/pgenv/bin/python"
export PYTHONPATH=src OMP_NUM_THREADS=4
B=scripts/single_scaling/build_operator.py
I=scripts/single_scaling/invert.py
for v in jump10 jump100 const; do
  $PY $B --visc $v --lmax 64 && $PY $I G_sl2013sv.s40rts.31.m_${v}_L64_Te40.npz
done
for t in s40rts.31.m.ab savani.31.m.ab semucb-wm1.31.m.ab tx2019.31.m.ab; do
  $PY $B --visc jump30 --tomo $t --lmax 64 && $PY $I G_${t%.ab}_jump30_L64_Te40.npz
done
echo BATCH DONE
