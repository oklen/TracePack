"""Draw the README results figure, assets/results.png and assets/results-dark.png.

    python3 figures/make_figures.py

The numbers are typed in below, each with its source; they are not recomputed here.
- Chat memory: the pre-registered 10-08 run of `tracepack bench lme` with real Claude Code 2.1.294
  (480 LongMemEval-S questions, 32k window, DeepSeek-V4-Pro as the model in every arm). Per-type counts are
  in docs/RESEARCH.md.
- Coding sessions: `tracepack bench codememo --data <CodeMemo folder>`, 2,000-token budget
  (the Benchmark table in README.md).
- Coding agent: the pre-registered SWE-bench Verified keep study (242 checkpoints, 119 instances) in our
  research harness; docs/RESEARCH.md.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _style import BLUE, GREY, save, use

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# (label, n, correct with Claude Code, correct with Claude Code + TracePack)
LME = [("all questions", 480, 154, 364),
       ("facts you\nstated", 67, 28, 63),
       ("your\npreferences", 28, 7, 20),
       ("what Claude\ntold you", 53, 18, 31),
       ("updated\nfacts", 74, 49, 65),
       ("across\nsessions", 129, 27, 94),
       ("time and\norder", 129, 25, 91)]
LME_CODEX = 327 / 480          # Codex's compaction, same run, all questions

# answer's turn delivered within 2,000 tokens, % of 153 questions
CODEMEMO = [("newest context only", 3.9), ("grep the transcript", 13.7),
            ("BM25", 34.0), ("TracePack", 34.6)]

# SWE-bench Verified at 32k: OpenHands' own compaction vs. the same plus forgotten file views put back
SWE = [("hit the step limit", 19.2, 8.4, "−10.8 pts\n[−18.6, −3.0]"),
       ("solved", 46.7, 50.0, "+3.3 pts, tied\n[−1.4, +10.3]")]


def pct(c, n):
    return 100.0 * c / n


def draw(theme):
    t = use(theme)
    fig = plt.figure(figsize=(9.6, 7.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.78, wspace=0.42)
    gap = dict(edgecolor=t["bg"], linewidth=1.5)

    # -- A: chat memory --------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    w = 0.36
    xs = range(len(LME))
    for i, (lab, n, cc, tp) in enumerate(LME):
        a, b = pct(cc, n), pct(tp, n)
        ax.bar(i - w / 2, a, w, color=GREY, **gap, label="Claude Code" if i == 0 else None)
        ax.bar(i + w / 2, b, w, color=BLUE, **gap, label="Claude Code + TracePack" if i == 0 else None)
    a0, b0 = pct(LME[0][2], LME[0][1]), pct(LME[0][3], LME[0][1])
    ax.text(0 - w / 2, a0 + 2, "%.1f%%" % a0, ha="center", va="bottom", fontsize=9, color=t["fg"])
    ax.text(0 + w / 2, b0 + 2, "%.1f%%" % b0, ha="center", va="bottom", fontsize=9, color=t["fg"])
    ax.hlines(100 * LME_CODEX, -0.5, 0.5, colors=t["fg"], linestyles=(0, (3, 2)), linewidth=1.3,
              label="Codex's compaction, all questions: %.1f%%" % (100 * LME_CODEX))
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["%s\n%.0f → %.0f" % (lab, pct(cc, n), pct(tp, n)) for lab, n, cc, tp in LME])
    ax.set_xlim(-0.6, len(LME) - 0.4)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index("Claude Code"), ls.index("Claude Code + TracePack"), 0]
    ax.legend([hs[j] for j in order], [ls[j] for j in order], loc="upper left", bbox_to_anchor=(0.0, 1.04),
              ncol=3, handlelength=1.6, borderaxespad=0, columnspacing=1.4)
    ax.set_title("What you said · LongMemEval-S chat histories, compacted at 32k\n"
                 "real Claude Code, same model in both arms; accuracy on 480 questions",
                 loc="left", fontsize=11, pad=24)

    # -- B: coding sessions ----------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for i, (lab, v) in enumerate(CODEMEMO):
        ref = lab.startswith("BM25")
        if ref:
            ax.barh(i, v, 0.56, color="none", edgecolor=t["muted"], linewidth=1.2, hatch="///")
        else:
            ax.barh(i, v, 0.56, color=BLUE if lab == "TracePack" else GREY, **gap)
        ax.text(v + 1.0, i, "%.1f%%" % v, va="center", fontsize=9, color=t["fg"])
    ax.set_yticks(range(len(CODEMEMO)))
    ax.set_yticklabels([lab + (" (reference)" if lab == "BM25" else "") for lab, _ in CODEMEMO])
    ax.set_xlim(0, 45)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels(["0", "10", "20", "30", "40%"])
    ax.xaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Code · what the tools printed\nCodeMemo: 66 real Claude Code coding sessions;\n"
                 "answer's turn found within 2,000 tokens", loc="left", fontsize=11)

    # -- C: coding agent -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    w = 0.34
    for i, (lab, s, k, delta) in enumerate(SWE):
        ax.bar(i - w / 2, s, w, color=GREY, **gap, label="OpenHands' compaction" if i == 0 else None)
        ax.bar(i + w / 2, k, w, color=BLUE, **gap, label="+ forgotten file views put back" if i == 0 else None)
        ax.text(i - w / 2, s + 1.2, "%.1f%%" % s, ha="center", va="bottom", fontsize=9, color=t["fg"])
        ax.text(i + w / 2, k + 1.2, "%.1f%%" % k, ha="center", va="bottom", fontsize=9, color=t["fg"])
        ax.text(i, max(s, k) + 9.5, delta, ha="center", va="bottom", fontsize=9, color=t["muted"], linespacing=1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([lab for lab, _, _, _ in SWE])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 96)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_yticklabels(["0", "20", "40", "60%"])
    ax.yaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.2, borderaxespad=0, fontsize=8.5)
    ax.set_title("Code · a coding agent, end to end\nSWE-bench Verified at 32k, 242 checkpoints;\n"
                 "research harness, same idea, not the plugin", loc="left", fontsize=11)

    save(fig, OUT, "results", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme)
