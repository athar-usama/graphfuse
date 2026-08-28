import json

import matplotlib

matplotlib.use("Agg")

from graphfuse.viz import (  # noqa: E402
    plot_kernel_launch_counts,
    plot_latency_delta,
    plot_memory_delta,
    render_fx_rewrite_diagram_svg,
    render_launch_hook_diagram_svg,
    render_memory_boundary_diagram_svg,
    write_results_json,
)


def _fake_results():
    return {
        "eager": [
            {"hidden": 256, "peak_memory_mb": 40.0, "latency_ms": 2.0},
            {"hidden": 512, "peak_memory_mb": 60.0, "latency_ms": 2.5},
        ],
        "inductor": [
            {"hidden": 256, "peak_memory_mb": 35.0, "latency_ms": 1.5},
            {"hidden": 512, "peak_memory_mb": 50.0, "latency_ms": 1.8},
        ],
        "graphfuse": [
            {"hidden": 256, "peak_memory_mb": 33.0, "latency_ms": 1.3},
            {"hidden": 512, "peak_memory_mb": 47.0, "latency_ms": 1.6},
        ],
    }


def test_memory_and_latency_delta_charts_render_without_crashing(tmp_path):
    results = _fake_results()
    plot_memory_delta(results, tmp_path / "mem.png")
    plot_latency_delta(results, tmp_path / "lat.png")
    assert (tmp_path / "mem.png").exists()
    assert (tmp_path / "lat.png").exists()


def test_kernel_launch_counts_renders_without_crashing(tmp_path):
    plot_kernel_launch_counts({"inductor": 4, "graphfuse": 2}, tmp_path / "counts.png")
    assert (tmp_path / "counts.png").exists()


def test_fx_rewrite_diagram_renders_without_crashing(tmp_path):
    render_fx_rewrite_diagram_svg(tmp_path / "diagram.svg")
    assert (tmp_path / "diagram.svg").exists()


def test_launch_hook_diagram_renders_without_crashing(tmp_path):
    render_launch_hook_diagram_svg(tmp_path / "hook.svg")
    assert (tmp_path / "hook.svg").exists()


def test_memory_boundary_diagram_renders_without_crashing(tmp_path):
    render_memory_boundary_diagram_svg(tmp_path / "boundary.svg")
    assert (tmp_path / "boundary.svg").exists()


def test_results_json_round_trips(tmp_path):
    results = _fake_results()
    path = tmp_path / "results.json"
    write_results_json(results, path)
    assert json.loads(path.read_text()) == results
