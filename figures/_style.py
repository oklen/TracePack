"""Shared figure style for the three agent-memory repositories (VeraKV, TracePack,
Compute-Globally-Materialize-Locally). The palette follows the Materialize paper figures.
Every README figure is drawn twice, for GitHub's light and dark themes:
assets/<name>.png and assets/<name>-dark.png, switched in the README with <picture>.
The Chinese README (README.md) uses Chinese copies, assets/<name>-zh.png and assets/<name>-zh-dark.png;
set_lang("zh") switches the labels (L) and the file names."""
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

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

LANG = "en"
# Chinese glyphs fall back to the first of these that is installed; Latin text stays in DejaVu Sans.
CJK = ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
       "SimHei", "WenQuanYi Zen Hei", "Arial Unicode MS"]


def _bold_cjk():
    """matplotlib reads only the first face of a .ttc, and on macOS the bold face of Hiragino Sans GB is
    the third one. Copy it out once (needs fontTools) so bold Chinese text renders bold; elsewhere bold
    Chinese falls back to the regular weight."""
    src = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    dst = os.path.join(tempfile.gettempdir(), "HiraginoSansGB-W6.ttf")
    if not os.path.exists(src):
        return
    try:
        if not os.path.exists(dst):
            from fontTools.ttLib import TTCollection
            TTCollection(src).fonts[2].save(dst)
        font_manager.fontManager.addfont(dst)
    except Exception:  # noqa: BLE001 - optional nicety
        pass


def set_lang(lang):
    """Draw labels and name files for 'en' (the English README) or 'zh' (the Chinese one)."""
    global LANG
    LANG = lang
    if lang == "zh":
        _bold_cjk()


def langs(argv=None):
    """Languages to draw: --lang en|zh on the command line, both by default."""
    argv = sys.argv[1:] if argv is None else argv
    if "--lang" in argv:
        return [argv[argv.index("--lang") + 1]]
    return ["en", "zh"]


def L(en, zh):
    """The label in the current language."""
    return zh if LANG == "zh" else en


def sans(primary="DejaVu Sans"):
    """A font family for explicit family= arguments: with Chinese fallbacks when drawing Chinese."""
    if LANG != "zh":
        return primary
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [primary] + [f for f in CJK if f in have]


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
        "font.family": sans() if LANG == "zh" else "sans-serif",
    })
    return t


def name_for(name, theme):
    """assets file stem: <name>[-zh][-dark]."""
    return name + ("-zh" if LANG == "zh" else "") + ("" if theme == "light" else "-dark")


def save(fig, outdir, name, theme):
    """Write <outdir>/<name>[-zh][-dark].png at 2x for a ~880 px README column."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name_for(name, theme) + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", path)
