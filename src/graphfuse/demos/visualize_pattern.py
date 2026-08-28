"""Renders the diagrams that need no GPU at all: the FX-graph rewrite, the
dual-hook launch-counting mechanism, and the memory-boundary explanation.
The kernel-launch-count and scaling charts live in ``benchmark.py`` because
they need a real CUDA device to measure.
"""

from __future__ import annotations

from pathlib import Path

from ..viz import (
    render_fx_rewrite_diagram_svg,
    render_launch_hook_diagram_svg,
    render_memory_boundary_diagram_svg,
)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASSETS_DIR = ROOT / "assets"


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    rewrite_path = ASSETS_DIR / "fx_rewrite_diagram.svg"
    render_fx_rewrite_diagram_svg(rewrite_path)
    print(f"wrote {rewrite_path}")

    hook_path = ASSETS_DIR / "launch_hook_diagram.svg"
    render_launch_hook_diagram_svg(hook_path)
    print(f"wrote {hook_path}")

    memory_path = ASSETS_DIR / "memory_boundary_diagram.svg"
    render_memory_boundary_diagram_svg(memory_path)
    print(f"wrote {memory_path}")


if __name__ == "__main__":
    main()
