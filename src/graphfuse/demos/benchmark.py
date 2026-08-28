"""The actual measurements behind the README: latency and memory across
hidden size for eager vs. stock ``torch.compile`` (Inductor) vs.
``torch.compile(backend=graphfuse_backend)``, plus a Triton kernel-launch
count across that same hidden-size sweep. Every number here is real,
measured on this machine; nothing is estimated.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from ..backend import graphfuse_backend
from ..model import FusibleStack
from ..viz import plot_kernel_launch_sweep, plot_latency_delta, plot_memory_delta, write_results_json

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASSETS_DIR = ROOT / "assets"

ROWS = 4096
DEPTH = 6
HIDDEN_SIZES = [256, 512, 1024, 2048, 4096]
WARMUP = 8
ITERS = 20


def _build(impl: str, hidden: int, depth: int):
    model = FusibleStack(hidden=hidden, depth=depth).cuda()
    if impl == "eager":
        return model
    if impl == "inductor":
        return torch.compile(model)
    if impl == "graphfuse":
        return torch.compile(model, backend=graphfuse_backend, fullgraph=True)
    raise ValueError(impl)


def _measure(model, x) -> tuple[float, float]:
    def step():
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.sum().backward()

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    latency_ms = statistics.median(times)
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return latency_ms, peak_memory_mb


def run_latency_memory_sweep() -> dict:
    results = {"eager": [], "inductor": [], "graphfuse": []}
    for hidden in HIDDEN_SIZES:
        x = torch.randn(ROWS, hidden, device="cuda")
        for impl in ("eager", "inductor", "graphfuse"):
            torch._dynamo.reset()
            model = _build(impl, hidden, DEPTH)
            latency_ms, peak_memory_mb = _measure(model, x)
            results[impl].append({"hidden": hidden, "latency_ms": latency_ms, "peak_memory_mb": peak_memory_mb})
            print(f"{impl:>10} hidden={hidden:<6} latency={latency_ms:8.3f}ms peak_mem={peak_memory_mb:8.1f}MB")
    return results


def _count_triton_launches(model, x) -> int:
    """Counts real Triton kernel launches by hooking two call sites, not
    one, and both of them directly rather than through GPU-side profiling:
    ``torch.profiler``'s CUDA activity tracing needs CUPTI, and CUPTI fails
    outright under WSL2 with ``CUPTI_ERROR_INVALID_DEVICE``, confirmed
    directly while writing this script, not assumed.

    ``triton.runtime.jit.JITFunction.run`` is the call a directly-invoked
    kernel goes through, cached or not, which covers this project's own
    ``_fwd_kernel``/``_bwd_kernel``. It is not, it turns out, what Inductor's
    *generated* kernels go through after the first call: Inductor wraps its
    own kernels in ``CachingAutotuner``, which after autotuning picks a best
    config and calls its compiled launcher directly, bypassing
    ``JITFunction.run`` entirely, confirmed directly here too, the first
    version of this function silently undercounted Inductor's side to zero.
    ``CachingAutotuner.run`` is the entry point that survives the bypass.

    Only Triton kernels are covered by either hook, so eager mode, which
    never touches Triton, is intentionally left out of this comparison; its
    story is in the latency and memory charts instead.
    """
    import triton.runtime.jit as jit_mod
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    def step():
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.sum().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    count = 0
    original_jit_run = jit_mod.JITFunction.run
    original_autotuner_run = CachingAutotuner.run

    def counting_jit_run(self, *args, **kwargs):
        nonlocal count
        count += 1
        return original_jit_run(self, *args, **kwargs)

    def counting_autotuner_run(self, *args, **kwargs):
        nonlocal count
        count += 1
        return original_autotuner_run(self, *args, **kwargs)

    jit_mod.JITFunction.run = counting_jit_run
    CachingAutotuner.run = counting_autotuner_run
    try:
        step()
        torch.cuda.synchronize()
    finally:
        jit_mod.JITFunction.run = original_jit_run
        CachingAutotuner.run = original_autotuner_run
    return count


def run_kernel_launch_sweep() -> dict:
    """The same hidden-size sweep as latency and memory, not one isolated
    config: a single measurement can't show whether a tie holds generally
    or just happened to land at one convenient size, and depth is fixed at
    1 here (rather than the 6 latency/memory use) so the count reflects one
    block's epilogue directly, not six of them summed."""
    results = {"inductor": [], "graphfuse": []}
    for hidden in HIDDEN_SIZES:
        x = torch.randn(ROWS, hidden, device="cuda")
        for impl in ("inductor", "graphfuse"):
            torch._dynamo.reset()
            model = _build(impl, hidden, depth=1)
            count = _count_triton_launches(model, x)
            results[impl].append({"hidden": hidden, "count": count})
            print(f"{impl:>10} hidden={hidden:<6} {count} Triton kernel launches (one block, forward + backward)")
    return results


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    results = run_latency_memory_sweep()
    write_results_json(results, ASSETS_DIR / "results.json")
    plot_latency_delta(results, ASSETS_DIR / "latency_delta.png")
    plot_memory_delta(results, ASSETS_DIR / "memory_delta.png")

    launch_counts = run_kernel_launch_sweep()
    write_results_json(launch_counts, ASSETS_DIR / "kernel_launch_counts.json")
    plot_kernel_launch_sweep(launch_counts, ASSETS_DIR / "kernel_launch_counts.png")


if __name__ == "__main__":
    main()
