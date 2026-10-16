"""Shared figure style for the three agent-memory repositories (VeraKV, TracePack,
Compute-Globally-Materialize-Locally). The palette follows the Materialize paper figures.
Every README figure is drawn twice, for GitHub's light and dark themes:
assets/<name>.png and assets/<name>-dark.png, switched in the README with <picture>."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#1f77b4"        # ours / primary
GREY = "#8c8c8c"        # baselines
GREEN = "#2ca02c"       # positive
RED = "#d62728"         # negative, chance
ORANGE = "#fdae6b"      # source event, highlight
LIGHTBLUE = "#c6dbef"   # carrier, selected

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "box": "#f6f8fa"},
    "dark": {"bg": "#0d1117", "fg": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d", "box": "#151b23"},
}


def use(theme):
    """Set rcParams for one theme and return its colours."""
    t = THEMES[theme]
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.titlecolor": t["fg"],
        "axes.edgecolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["fg"], "ytick.labelcolor": t["fg"],
        "grid.color": t["grid"], "legend.frameon": False,
    })
    return t


def save(fig, outdir, name, theme):
    """Write <outdir>/<name>.png (light) or <outdir>/<name>-dark.png (dark) at 2x for a ~880 px README column."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ("" if theme == "light" else "-dark") + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", path)
