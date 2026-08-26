"""Renders the one diagram that needs no GPU at all: the FX-graph rewrite
itself. The kernel-launch-count and scaling charts live in ``benchmark.py``
because they need a real CUDA device to measure.
"""

from __future__ import annotations

from pathlib import Path

from ..viz import render_fx_rewrite_diagram_svg

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASSETS_DIR = ROOT / "assets"


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    diagram_path = ASSETS_DIR / "fx_rewrite_diagram.svg"
    render_fx_rewrite_diagram_svg(diagram_path)
    print(f"wrote {diagram_path}")


if __name__ == "__main__":
    main()
