#!/usr/bin/env python3
"""Mobile performance profiler for the Hermes model (pre-conversion).

Measures the PyTorch reference model's throughput and memory at batch size 1
(the mobile serving target) so you can compare presets and sequence lengths
*before* spending time on LiteRT conversion. Numbers here are an upper bound on
device behaviour — the INT4 ``.litertlm`` graph will differ — but the relative
ordering between presets/lengths is a useful proxy.

Reported per sequence length:

* **Prefill tokens/sec** — throughput of the single forward pass over the prompt.
* **Decode tokens/sec** — throughput of incremental single-token generation
  (KV-cache reuse), the metric users feel during streaming.
* **Time-to-first-token (ms)** — prefill latency for the prompt.
* **Peak RSS (MB)** — process resident memory via ``psutil``.
* **Peak VRAM (MB)** — ``torch.cuda.max_memory_allocated`` when on GPU.

Example::

    python scripts/benchmark.py --preset hermes-270m \
        --seq-lens 64 128 256 512 1024 --runs 5

Prints a markdown table and writes ``benchmark_results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from hermes.config import HermesConfig, get_config  # noqa: E402
from hermes.inference import HermesInference  # noqa: E402
from hermes.model import build_model  # noqa: E402

try:
    import psutil  # noqa: E402

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is a declared dependency
    _HAS_PSUTIL = False


def _peak_rss_mb() -> Optional[float]:
    """Current process resident set size in MB, or None if psutil is absent."""
    if not _HAS_PSUTIL:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _reset_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _peak_vram_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return None


@torch.no_grad()
def benchmark_seq_len(
    engine: HermesInference,
    seq_len: int,
    runs: int,
    decode_tokens: int,
    device: torch.device,
) -> Dict[str, float]:
    """Time prefill + decode for one sequence length, averaged over ``runs``."""
    model = engine.model
    vocab = engine.config.vocab_size
    _reset_vram(device)

    prefill_times: List[float] = []
    decode_times: List[float] = []

    for _ in range(runs):
        prompt_ids = torch.randint(0, vocab, (1, seq_len), device=device)

        # --- Prefill: single forward over the whole prompt. ---
        caches = [None] * len(model.layers)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits, caches = engine._forward_with_cache(prompt_ids, caches, start_pos=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_times.append(time.perf_counter() - t0)

        # --- Decode: incremental single-token steps reusing the KV cache. ---
        pos = seq_len
        next_id = logits.argmax(dim=-1, keepdim=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(decode_tokens):
            step_logits, caches = engine._forward_with_cache(next_id, caches, start_pos=pos)
            next_id = step_logits.argmax(dim=-1, keepdim=True)
            pos += 1
            if pos >= engine.config.max_seq_len:
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_times.append(time.perf_counter() - t0)

    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    decoded = min(decode_tokens, max(1, engine.config.max_seq_len - seq_len))

    return {
        "seq_len": seq_len,
        "ttft_ms": avg_prefill * 1000.0,
        "prefill_tok_per_s": seq_len / avg_prefill if avg_prefill > 0 else 0.0,
        "decode_tok_per_s": decoded / avg_decode if avg_decode > 0 else 0.0,
        "peak_rss_mb": _peak_rss_mb() or 0.0,
        "peak_vram_mb": _peak_vram_mb(device) or 0.0,
    }


class _NullTokenizer:
    """Placeholder tokenizer — benchmark only needs the model, not real text."""

    def encode(self, text: str) -> List[int]:
        return [1]

    def decode(self, ids: List[int]) -> str:
        return ""


def render_table(rows: List[Dict[str, float]]) -> str:
    """Format benchmark rows as a markdown table."""
    header = (
        "| seq_len | TTFT (ms) | prefill tok/s | decode tok/s | "
        "peak RSS (MB) | peak VRAM (MB) |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {int(r['seq_len'])} | {r['ttft_ms']:.1f} | "
            f"{r['prefill_tok_per_s']:.1f} | {r['decode_tok_per_s']:.1f} | "
            f"{r['peak_rss_mb']:.1f} | {r['peak_vram_mb']:.1f} |"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    config: HermesConfig = get_config(args.preset)
    model = build_model(config)
    engine = HermesInference(model, _NullTokenizer(), device=device, preset_name=args.preset)
    print(engine)

    rows: List[Dict[str, float]] = []
    for seq_len in args.seq_lens:
        if seq_len >= config.max_seq_len:
            print(f"Skipping seq_len={seq_len} (>= max_seq_len={config.max_seq_len}).")
            continue
        print(f"Benchmarking seq_len={seq_len} ...")
        rows.append(
            benchmark_seq_len(engine, seq_len, args.runs, args.decode_tokens, device)
        )

    table = render_table(rows)
    print("\n" + table + "\n")

    results = {
        "preset": args.preset,
        "device": args.device,
        "runs": args.runs,
        "decode_tokens": args.decode_tokens,
        "param_count": sum(p.numel() for p in model.parameters()),
        "psutil_available": _HAS_PSUTIL,
        "rows": rows,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.output}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Hermes model speed/memory.")
    p.add_argument("--preset", default="hermes-270m", choices=["hermes-1b", "hermes-500m", "hermes-270m"])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    p.add_argument("--runs", type=int, default=5, help="Repeats per sequence length.")
    p.add_argument("--decode-tokens", type=int, default=32, help="Tokens to time during decode.")
    p.add_argument("--device", default="cpu", help="Torch device (cpu, cuda, cuda:0, ...).")
    p.add_argument("--output", default="benchmark_results.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
