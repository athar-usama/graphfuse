"""Command-line entry point: ``graphfuse {diagram,benchmark}``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphfuse")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diagram", help="render the FX-graph-rewrite and kernel-launch diagrams")
    sub.add_parser("benchmark", help="run the eager vs. inductor vs. graphfuse sweep and produce the charts")
    args = parser.parse_args(argv)

    if args.command == "diagram":
        from .demos.visualize_pattern import main as run

        run()
    elif args.command == "benchmark":
        from .demos.benchmark import main as run

        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
