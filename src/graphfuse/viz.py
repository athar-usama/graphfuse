"""Charts for the benchmark and the FX-graph diagrams.

Palette, kept consistent with the rest of this portfolio: red for eager (no
compiler at all, the baseline), blue for stock `torch.compile` on Inductor
(the tool this project composes with, not replaces), emerald for
`torch.compile(backend=graphfuse_backend)` (this project). No bar charts: the
launch-count comparison below is a dot/lollipop plot instead, and the
latency/memory comparisons are plotted as a relative delta from Inductor
rather than three raw curves, because the raw curves sit close enough
together that they overlap into unreadable near-identical lines; the
percentages are also exactly the numbers the README's own prose talks about.
"""

from __future__ import annotations

import html
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


def _plot_relative_delta(results: dict, key: str, baseline: str, ylabel: str, title: str, path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND)

    baseline_by_hidden = {r["hidden"]: r[key] for r in results[baseline] if r.get(key) is not None}

    # Distinct line styles and marker sizes per implementation, not just
    # distinct colors: when two implementations track each other closely
    # (eager and graphfuse do, on memory), their curves can sit almost
    # exactly on top of each other, and a solid line under another solid
    # line of the same width just disappears rather than reading as "these
    # two agree."
    style_by_impl = {"eager": ("--", 9), "inductor": ("-", 6), "graphfuse": ("-", 5)}

    series = {}
    for impl in _ORDER:
        if impl == baseline:
            continue
        xs, ys = [], []
        for row in results[impl]:
            hidden = row["hidden"]
            if row.get(key) is None or hidden not in baseline_by_hidden:
                continue
            xs.append(hidden)
            ys.append((row[key] / baseline_by_hidden[hidden] - 1) * 100)
        series[impl] = (xs, ys)

    # A filled band between the two non-baseline curves, drawn before either
    # line: even where they sit almost exactly on top of each other, a
    # colored sliver of area between them still reads as "two curves paired
    # closely," where two overlapping strokes alone would just look like one.
    non_baseline = [impl for impl in _ORDER if impl != baseline]
    if len(non_baseline) == 2 and all(series[impl][0] for impl in non_baseline):
        (impl_a, impl_b) = non_baseline
        xs_a, ys_a = series[impl_a]
        xs_b, ys_b = series[impl_b]
        if xs_a == xs_b:
            ax.fill_between(xs_a, ys_a, ys_b, color="#f59e0b", alpha=0.16, zorder=2, linewidth=0)

    for impl in non_baseline:
        xs, ys = series[impl]
        linestyle, markersize = style_by_impl[impl]
        ax.plot(xs, ys, color=_COLORS[impl], linewidth=2.4, linestyle=linestyle, marker="o",
                 markersize=markersize, markerfacecolor=_COLORS[impl], markeredgecolor="white",
                 markeredgewidth=1.2, label=_LABELS[impl], zorder=3)

    # When the two curves agree closely everywhere, the filled band between
    # them above is too thin to see at all, which is itself worth calling
    # out directly rather than leaving the reader to wonder whether the
    # second line rendered at all: a fixed corner callout says explicitly
    # that the closeness is the finding, not a glitch, without needing to
    # point at any one data point that might sit anywhere in the frame.
    if len(non_baseline) == 2:
        impl_a, impl_b = non_baseline
        xs_a, ys_a = series[impl_a]
        xs_b, ys_b = series[impl_b]
        if xs_a == xs_b and xs_a:
            gaps = [abs(a - b) for a, b in zip(ys_a, ys_b, strict=True)]
            if max(gaps) < 1.0:
                # Extra headroom above the data so the callout box has clear
                # space of its own instead of sitting on top of the curve's
                # highest point.
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.22)
                gap_desc = "identically" if max(gaps) < 0.01 else f"within {max(gaps):.2f} points"
                ax.text(0.02, 0.97, f"{_LABELS[impl_a]} and {_LABELS[impl_b]} track {gap_desc} "
                        "at every size measured", transform=ax.transAxes, ha="left", va="top",
                        fontsize=9.5, color=_TEXT, zorder=6,
                        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": _MUTED, "lw": 1.0})

    ax.axhline(0, color=_MUTED, linewidth=1.3, linestyle="--", zorder=1)
    ax.text(0.02, 0.02, f"0% = tied with {_LABELS[baseline]}", transform=ax.transAxes,
             ha="left", va="bottom", fontsize=9.5, color=_MUTED, style="italic")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("hidden size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, which="both", color=_GRID, linewidth=0.7, zorder=0)
    ax.legend(loc="upper right", frameon=False, labelcolor=_TEXT)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def plot_latency_delta(results: dict, path) -> None:
    _plot_relative_delta(results, "latency_ms", "inductor", "latency vs. inductor (%, 0 = tied)",
                          "Latency relative to stock Inductor, across hidden size", path)


def plot_memory_delta(results: dict, path) -> None:
    _plot_relative_delta(results, "peak_memory_mb", "inductor", "peak memory vs. inductor (%, 0 = tied)",
                          "Peak memory relative to stock Inductor, across hidden size", path)


def plot_kernel_launch_sweep(results: dict, path) -> None:
    """Kernel-launch count across the same hidden-size sweep as the latency
    and memory charts, not one config in isolation: a single measurement at
    one hidden size, tied 2-to-2, has nowhere to show variation at all,
    which reads as a rendering glitch rather than a real result. Sweeping it
    turns "is this tied" into "is this tied at every size," a claim the
    chart can actually support or refute; here it stays exactly tied
    everywhere measured, which is itself the finding worth showing, not an
    artifact of picking one convenient config. Eager mode is deliberately
    not here: it never routes through Triton at all, so a launch count of 0
    would compare a different kind of cost, not a smaller one; its story is
    in the latency and memory charts instead.
    """
    import matplotlib.pyplot as plt

    order = ("inductor", "graphfuse")
    style_by_impl = {"inductor": ("-", 8), "graphfuse": ("--", 9)}

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND)

    series = {}
    for impl in order:
        rows = sorted(results[impl], key=lambda r: r["hidden"])
        series[impl] = ([r["hidden"] for r in rows], [r["count"] for r in rows])

    xs_a, ys_a = series["inductor"]
    xs_b, ys_b = series["graphfuse"]
    if xs_a == xs_b:
        ax.fill_between(xs_a, ys_a, ys_b, color="#f59e0b", alpha=0.18, zorder=1, linewidth=0)

    for impl in order:
        xs, ys = series[impl]
        linestyle, markersize = style_by_impl[impl]
        ax.plot(xs, ys, color=_COLORS[impl], linewidth=2.4, linestyle=linestyle, marker="o",
                 markersize=markersize, markerfacecolor=_COLORS[impl], markeredgecolor="white",
                 markeredgewidth=1.2, label=_LABELS[impl], zorder=3)

    all_counts = [c for _, ys in series.values() for c in ys]
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, max(all_counts) + 1.5)
    ax.set_yticks(range(0, max(all_counts) + 2))
    ax.set_xlabel("hidden size")
    ax.set_ylabel("Triton kernel launches\n(one block, forward + backward)")
    ax.set_title("How many Triton kernels the epilogue costs, across hidden size",
                 fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, which="both", color=_GRID, linewidth=0.7, zorder=0)
    ax.legend(loc="center left", frameon=False, labelcolor=_TEXT)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


_DARK_BG = "#12141a"
_DARK_PANEL = "#1a1d27"
_DARK_ARROW = "#4b5563"
_DARK_TEXT = "#c9ced9"

_CHAR_PX = 7.6
_BOX_PAD_X = 28
_LINE_H = 16


def _box_width(label: str, min_width: float = 220.0) -> float:
    """Monospace font, so character count is a reliable proxy for pixel
    width. Sized per label rather than a fixed constant, after a fixed-width
    version let its longest label overflow its own box."""
    longest_line = max(len(line) for line in label.split("\n"))
    return max(min_width, longest_line * _CHAR_PX + _BOX_PAD_X * 2)


def _text_block(cx: float, cy: float, label: str, *, font_size: float = 12.5) -> str:
    lines = label.split("\n")
    start_y = cy - (len(lines) - 1) * _LINE_H / 2
    parts = []
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{cx:.1f}" y="{start_y + i * _LINE_H + 4:.1f}" font-size="{font_size}" fill="{_DARK_TEXT}" '
            f'text-anchor="middle">{html.escape(line)}</text>'
        )
    return "\n".join(parts)


def _box(x: float, y: float, w: float, h: float, label: str, color: str) -> str:
    cx, cy = x + w / 2, y + h / 2
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
        f'fill="{_DARK_PANEL}" stroke="{color}" stroke-width="1.8"/>\n{_text_block(cx, cy, label)}'
    )


def _edge(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<path d="M{x1:.1f},{y1:.1f} C{x1 + 40:.1f},{y1:.1f} {x2 - 40:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" '
        f'fill="none" stroke="{_DARK_ARROW}" stroke-width="1.5" marker-end="url(#arrow)"/>'
    )


def _svg_header(width: float, height: float, caption: str) -> list:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="Consolas, Menlo, monospace">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="{_DARK_BG}"/>',
        f'<text x="20" y="20" font-size="14" fill="{_DARK_TEXT}">{html.escape(caption)}</text>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{_DARK_ARROW}"/></marker></defs>',
    ]


def render_fx_rewrite_diagram_svg(path) -> None:
    """Three nodes collapsing into one: the exact rewrite `pattern.py`
    performs on the traced graph, before Inductor ever sees it. Every box is
    sized to its own label and the fused node's two-line label is rendered
    as two real lines, not squashed onto one: the original fixed-width,
    single-line version overflowed both boxes at once."""
    box_h = 46
    top_y = 30
    row_gap = 66
    gap = 90

    before_labels = ["x + bias", 'gelu(_, approximate="tanh")', "_ + residual"]
    after_label = "torch.ops.graphfuse.\nfused_bias_gelu_residual"

    before_w = max(_box_width(label) for label in before_labels)
    after_w = _box_width(after_label)
    after_h = box_h + _LINE_H

    before_x = 30
    after_x = before_x + before_w + gap
    after_y = top_y + row_gap - (after_h - box_h) / 2

    width = after_x + after_w + 30
    height = top_y + row_gap * 2 + box_h + 30

    svg = _svg_header(width, height, "Three FX nodes, one custom op")
    after_cy = after_y + after_h / 2
    for i, label in enumerate(before_labels):
        by = top_y + i * row_gap
        svg.append(_box(before_x, by, before_w, box_h, label, "#3b82f6"))
        svg.append(_edge(before_x + before_w, by + box_h / 2, after_x, after_cy))
    svg.append(_box(after_x, after_y, after_w, after_h, after_label, "#059669"))
    svg.append("</svg>")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def render_launch_hook_diagram_svg(path) -> None:
    """Two call paths, one counter: why the launch count needed two hooks,
    not one. A kernel invoked directly always goes through
    ``JITFunction.run``. Inductor's own generated kernels go through it only
    on the very first call; every call after that, once autotuning has
    picked a config, goes through ``CachingAutotuner.run`` instead, which the
    first version of this counter did not hook at all, and silently
    undercounted Inductor's side to zero."""
    row_top_y, row_bot_y = 40, 150
    box_h = 50
    gap = 90

    col1_labels = ["direct launch:\n_fwd_kernel / _bwd_kernel", "Inductor's generated kernel\n(2nd call onward)"]
    col2_labels = ["JITFunction.run", "CachingAutotuner.run"]
    col3_label = "count += 1"

    col1_w = max(_box_width(label) for label in col1_labels)
    col2_w = max(_box_width(label) for label in col2_labels)
    col3_w = _box_width(col3_label, min_width=160)

    col1_x = 20
    col2_x = col1_x + col1_w + gap
    col3_x = col2_x + col2_w + gap
    mid_y = (row_top_y + row_bot_y) / 2 + box_h / 2

    width = col3_x + col3_w + 30
    height = row_bot_y + box_h + 30

    svg = _svg_header(width, height, "Two call paths, one counter")
    for row_y, label1, label2, color in zip(
        (row_top_y, row_bot_y), col1_labels, col2_labels, ("#3b82f6", "#059669"), strict=True
    ):
        svg.append(_box(col1_x, row_y, col1_w, box_h, label1, color))
        svg.append(_edge(col1_x + col1_w, row_y + box_h / 2, col2_x, row_y + box_h / 2))
        svg.append(_box(col2_x, row_y, col2_w, box_h, label2, color))
        svg.append(_edge(col2_x + col2_w, row_y + box_h / 2, col3_x, mid_y))
    svg.append(_box(col3_x, mid_y - box_h / 2, col3_w, box_h, col3_label, "#b45309"))
    svg.append("</svg>")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def render_memory_boundary_diagram_svg(path) -> None:
    """Why graphfuse's peak memory tracks eager's instead of Inductor's.
    Inductor's memory planner can reuse buffers across any span of ops it
    can see through. A ``torch.library`` custom op is opaque by design, the
    same property that makes it traceable at all, so the one reusable span
    Inductor would normally plan across this epilogue splits into two
    separate ones at the custom op's edges, with no reuse across the wall."""
    box_w, box_h = 128, 44
    gap = 16
    n = 5
    top_y = 76
    row_gap = 130
    bot_y = top_y + row_gap
    start_x = 40
    labels = ["prev op", "bias-add", "gelu", "residual-add", "next op"]

    width = start_x * 2 + n * box_w + (n - 1) * gap
    height = bot_y + box_h + 40

    def row(y: float, colors: list) -> tuple:
        xs = [start_x + i * (box_w + gap) for i in range(n)]
        parts = [_box(x, y, box_w, box_h, label, color) for x, label, color in zip(xs, labels, colors, strict=True)]
        return "\n".join(parts), xs

    top_svg, top_xs = row(top_y, ["#3b82f6"] * n)
    bot_svg, bot_xs = row(bot_y, ["#3b82f6", "#b45309", "#b45309", "#b45309", "#3b82f6"])

    band_h = 14
    top_band = (
        f'<rect x="{top_xs[0] - 8:.1f}" y="{top_y - 30:.1f}" width="{top_xs[-1] + box_w + 8 - (top_xs[0] - 8):.1f}" '
        f'height="{band_h}" rx="7" fill="#3b82f6" fill-opacity="0.22"/>'
    )
    wall_left, wall_right = bot_xs[1] - gap / 2, bot_xs[3] + box_w + gap / 2
    left_band = (
        f'<rect x="{bot_xs[0] - 8:.1f}" y="{bot_y - 30:.1f}" width="{wall_left - (bot_xs[0] - 8):.1f}" '
        f'height="{band_h}" rx="7" fill="#059669" fill-opacity="0.22"/>'
    )
    right_band = (
        f'<rect x="{wall_right:.1f}" y="{bot_y - 30:.1f}" width="{bot_xs[-1] + box_w + 8 - wall_right:.1f}" '
        f'height="{band_h}" rx="7" fill="#059669" fill-opacity="0.22"/>'
    )

    svg = _svg_header(width, height, "One reusable memory span vs. two, split at the opaque boundary")
    svg.append(f'<text x="{start_x}" y="{top_y - 38:.1f}" font-size="11.5" fill="#3b82f6">'
               "Inductor sees through every op: one continuous reuse span</text>")
    svg += [top_band, top_svg]
    svg.append(f'<text x="{start_x}" y="{bot_y - 38:.1f}" font-size="11.5" fill="#059669">'
               "graphfuse&#8217;s custom op is opaque: the span splits in two</text>")
    svg += [left_band, right_band, bot_svg]
    svg.append("</svg>")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def write_results_json(results: dict, path) -> None:
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")
