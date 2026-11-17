"""Shared figure style: LaTeX typesetting and the palette.

Uses real LaTeX (text.usetex) when it works on this machine (needs latex, dvipng or ghostscript,
and the cm-super fonts); otherwise matplotlib's own TeX renderer with Computer Modern fonts, which
looks the same for labels and maths. Write every label as LaTeX: r"Depth (km)", r"$\\ell$".
"""

import matplotlib
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
_BASE = {
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def _usetex_works():
    import io

    try:
        with matplotlib.rc_context({"text.usetex": True, "font.family": "serif"}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"$\ell$")
            fig.savefig(io.BytesIO(), format="png", dpi=20)
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


def apply():
    plt.rcParams.update(_BASE)
    if _usetex_works():
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        return "latex"
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }
    )
    return "mathtext (Computer Modern)"
