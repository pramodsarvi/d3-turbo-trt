"""Plot the measured d3-turbo performance results into docs/perf/.

Numbers are copied from the real-model runs (RTX PRO 6000 Blackwell, BF16 unless
noted; 4096-token state, 16 questions, 5 requests) recorded under traces/real/:
bench-*.json for throughput and profile-compile.log for latency.

    python scripts/plot_perf.py            # writes docs/perf/performance.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "perf" / "performance.png"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BAR = "#2a78d6"        # categorical slot 1 (one series per panel)
BAR_FAIL = "#a9c9ef"   # same hue, lighter: configuration fails the 0.01 parity gate

# (label, questions/s, parity max abs diff, passes gate)
THROUGHPUT = [
    ("HF reference backend", 34.9, 0.002, True),
    ("native, first version", 40.3, 0.040, False),
    ("native, exact SDPA (current)", 38.5, 0.0058, True),
    ("native FP8", 41.1, 0.028, False),
]

# (label, cold request ms, state pass ms, warm request ms)
LATENCY = [
    ("native", 1194, 777, 416),
    ("native + compile state", 1058, 643, 412),
    ("native + compile all", 1283, 643, 637),
]


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.invert_yaxis()


def _bars(ax, labels, values, colors) -> None:
    ax.barh(labels, values, height=0.42, color=colors, edgecolor=SURFACE, linewidth=2)


def main() -> None:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.6), facecolor=SURFACE, gridspec_kw={"height_ratios": [4, 3], "hspace": 0.55}
    )

    labels = [r[0] for r in THROUGHPUT]
    values = [r[1] for r in THROUGHPUT]
    _bars(top, labels, values, [BAR if r[3] else BAR_FAIL for r in THROUGHPUT])
    base = THROUGHPUT[0][1]
    for i, (_, qps, diff, ok) in enumerate(THROUGHPUT):
        gain = f"{(qps / base - 1) * 100:+.0f}%" if i else "baseline"
        mark = "passes" if ok else "fails"
        top.text(qps + 0.6, i, f"{qps:.1f}  ({gain})   parity {diff:.3f} {mark}",
                 va="center", fontsize=9, color=TEXT if ok else MUTED)
    top.set_xlim(0, 62)
    _style(top, "Throughput: questions per second (higher is better)", "questions / s")

    labels = [r[0] for r in LATENCY]
    cold = [r[1] for r in LATENCY]
    _bars(bottom, labels, cold, [BAR] * len(LATENCY))
    for i, (_, c, state, warm) in enumerate(LATENCY):
        bottom.text(c + 15, i, f"{c} ms   (state {state} ms, warm {warm} ms)", va="center", fontsize=9, color=TEXT)
    bottom.set_xlim(0, 1900)
    _style(bottom, "Cold request latency: new 4096-token state (lower is better)", "milliseconds")

    fig.text(0.01, 0.005,
             "Real 27B model, RTX PRO 6000 Blackwell, BF16 unless noted. 16 questions per request. "
             "Parity = max |p - p_ref| vs the reference runtime; gate 0.01.\nLighter bars fail the parity gate.",
             fontsize=8, color=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
