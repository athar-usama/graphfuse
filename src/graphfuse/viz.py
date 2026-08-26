"""Charts for the benchmark and the FX-graph-rewrite diagram.

Palette, kept consistent with the rest of this portfolio: red for eager (no
compiler at all, the baseline), blue for stock `torch.compile` on Inductor
(the tool this project composes with, not replaces), emerald for
`torch.compile(backend=graphfuse_backend)` (this project). No bar charts: the
launch-count comparison below is a dot/lollipop plot instead.
"""

from __future__ import annotations

import json
from pathlib import Path

_EAGER = "#ef4444"
_INDUCTOR = "#3b82f6"
_GRAPHFUSE = "#059669"
_GRID = "#e5e7eb"
_TEXT = "#1f2937"
_MUTED = "#6b7280"
_BAND = "#f8fafc"

_LABELS = {"eager": "eager", "inductor": "torch.compile (inductor)", "graphfuse": "torch.compile (graphfuse)"}
_COLORS = {"eager": _EAGER, "inductor": _INDUCTOR, "graphfuse": _GRAPHFUSE}
_ORDER = ("eager", "inductor", "graphfuse")


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def _plot_scaling(results: dict, key: str, ylabel: str, title: str, path, *, log_y: bool = True) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND)

    for impl in _ORDER:
        rows = [r for r in results[impl] if r.get(key) is not None]
        if not rows:
            continue
        xs = [r["hidden"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, color=_COLORS[impl], linewidth=2.2, marker="o", markersize=4, label=_LABELS[impl], zorder=3)

    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("hidden size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, which="both", color=_GRID, linewidth=0.7, zorder=0)
    ax.legend(loc="upper left", frameon=False, labelcolor=_TEXT)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_memory_scaling(results: dict, path) -> None:
    _plot_scaling(results, "peak_memory_mb", "peak memory (MB, log scale)",
                  "Peak memory, forward + backward, across hidden size", path, log_y=True)


def plot_latency_scaling(results: dict, path) -> None:
    _plot_scaling(results, "latency_ms", "median latency (ms, log scale)",
                  "Latency, forward + backward, across hidden size", path, log_y=True)


def plot_kernel_launch_counts(counts: dict, path) -> None:
    """A horizontal lollipop plot, not a bar chart: one stem per compiled
    path, a dot at the measured Triton-kernel-launch count, for one
    forward+backward pass of a single block. Eager mode is deliberately not
    here: it never routes through Triton at all, so a launch count of 0
    would compare a different kind of cost, not a smaller one; its story is
    in the latency and memory charts instead.
    """
    import matplotlib.pyplot as plt

    order = ("inductor", "graphfuse")
    labels = [_LABELS[impl] for impl in order]
    values = [counts[impl] for impl in order]
    ys = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND)

    for y, impl, value in zip(ys, order, values, strict=True):
        ax.plot([0, value], [y, y], color=_COLORS[impl], linewidth=2.2, zorder=2)
        ax.plot(value, y, "o", color=_COLORS[impl], markersize=11, zorder=3)
        ax.text(value + max(values) * 0.02, y, str(value), va="center", color=_TEXT, fontsize=11, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=_TEXT)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Triton kernel launches (one block, forward + backward)")
    ax.set_title("How many Triton kernels the epilogue actually costs", fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, axis="x", color=_GRID, linewidth=0.7, zorder=0)
    ax.invert_yaxis()
    _style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


_DARK_BG = "#12141a"
_DARK_PANEL = "#1a1d27"
_DARK_ARROW = "#4b5563"
_DARK_TEXT = "#c9ced9"


def render_fx_rewrite_diagram_svg(path) -> None:
    """Three nodes collapsing into one: the exact rewrite `pattern.py`
    performs on the traced graph, before Inductor ever sees it.
    """
    width, height = 860, 230
    box_w, box_h = 220, 46
    before_x, after_x = 30, 610
    top_y = 30

    def box(x, y, label, color, width=box_w):
        return (
            f'<rect x="{x}" y="{y}" width="{width}" height="{box_h}" rx="8" '
            f'fill="{_DARK_PANEL}" stroke="{color}" stroke-width="1.8"/>'
            f'<text x="{x + width / 2}" y="{y + box_h / 2 + 4}" font-size="12.5" fill="{_DARK_TEXT}" '
            f'text-anchor="middle">{label}</text>'
        )

    before_boxes = [
        (before_x, top_y, "x + bias", "#3b82f6"),
        (before_x, top_y + 66, "gelu(_, approximate=\"tanh\")", "#3b82f6"),
        (before_x, top_y + 132, "_ + residual", "#3b82f6"),
    ]
    after_box = (after_x, top_y + 66, "torch.ops.graphfuse.\nfused_bias_gelu_residual", "#059669")

    def edge(x1, y1, x2, y2):
        return (
            f'<path d="M{x1},{y1} C{x1 + 40},{y1} {x2 - 40},{y2} {x2},{y2}" '
            f'fill="none" stroke="{_DARK_ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Consolas, Menlo, monospace">',
        f'<rect width="{width}" height="{height}" fill="{_DARK_BG}"/>',
        f'<text x="20" y="20" font-size="14" fill="{_DARK_TEXT}">Three FX nodes, one custom op</text>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{_DARK_ARROW}"/></marker></defs>',
    ]
    for bx, by, label, color in before_boxes:
        svg.append(box(bx, by, label, color))
        svg.append(edge(bx + box_w, by + box_h / 2, after_box[0], after_box[1] + box_h / 2))
    svg.append(box(after_box[0], after_box[1], after_box[2].replace("\n", " "), after_box[3], width=box_w + 30))
    svg.append("</svg>")

    Path(path).write_text("\n".join(svg), encoding="utf-8")


def write_results_json(results: dict, path) -> None:
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")
