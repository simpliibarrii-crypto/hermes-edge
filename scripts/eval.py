#!/usr/bin/env python3
"""Lightweight evaluation harness for the Hermes model.

Runs two cheap, CI-friendly evals against tiny bundled datasets and prints a
summary table:

* **Perplexity** — token-level cross-entropy perplexity over ``data/eval.jsonl``
  (10 diverse chat conversations rendered through the Hermes ChatML template).
* **Tool-call accuracy** — over ``data/tool_eval.jsonl`` (10 prompts whose
  expected reply is a ``<tool_call>``), the fraction for which the model emits a
  parseable tool call whose ``name`` matches the expected tool.

The harness runs with **randomly initialized weights** when ``--checkpoint`` is
omitted (perplexity will be ~vocab-size and tool accuracy ~0), which keeps it
usable as a smoke test in CI. With a trained checkpoint + SentencePiece
tokenizer the numbers become meaningful.

Example::

    python scripts/eval.py --preset hermes-270m \
        --checkpoint checkpoints/hermes-270m.pt --tokenizer tokenizer/hermes.model

Writes ``eval_results.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from hermes.chat_template import Message, build_prompt, parse_tool_call  # noqa: E402
from hermes.config import get_config  # noqa: E402
from hermes.inference import HermesInference  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ByteTokenizer:
    """Deterministic byte-level tokenizer fallback (no external deps).

    Used when no SentencePiece model is supplied so the harness runs in CI.
    Maps each UTF-8 byte to an id offset past the reserved special tokens.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.offset = 5  # leave room for pad/bos/eos/tool sentinels

    def encode(self, text: str) -> List[int]:
        ids = [(b + self.offset) % self.vocab_size for b in text.encode("utf-8")]
        return ids or [1]

    def decode(self, ids: List[int]) -> str:
        out = bytes((i - self.offset) % 256 for i in ids if i >= self.offset)
        return out.decode("utf-8", errors="replace")


def load_tokenizer(path: Optional[str], vocab_size: int):
    """Load a SentencePiece tokenizer if available, else the byte fallback."""
    if path and os.path.exists(path):
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor(model_file=path)

            class _SP:
                def encode(self, text: str) -> List[int]:
                    return sp.encode(text, out_type=int)

                def decode(self, ids: List[int]) -> str:
                    return sp.decode(ids)

            return _SP()
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            print(f"[warn] could not load SentencePiece tokenizer ({exc}); using bytes.")
    return ByteTokenizer(vocab_size)


def _messages_from(obj: Dict[str, Any]) -> List[Message]:
    return [Message(m["role"], m["content"]) for m in obj["messages"]]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def eval_perplexity(engine: HermesInference, path: str) -> float:
    """Mean perplexity over rendered conversations in ``path``."""
    examples = _read_jsonl(path)
    total_loss = 0.0
    count = 0
    for ex in examples:
        prompt = build_prompt(_messages_from(ex), add_generation_prompt=False)
        ids = engine.tokenizer.encode(prompt)[: engine.config.max_seq_len]
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=engine.device)
        out = engine.model(input_ids, labels=input_ids)
        loss = out["loss"]
        if loss is not None and math.isfinite(float(loss)):
            total_loss += float(loss)
            count += 1
    if count == 0:
        return float("nan")
    mean_loss = total_loss / count
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


def eval_tool_calls(engine: HermesInference, path: str, max_new_tokens: int) -> Dict[str, float]:
    """Fraction of prompts where the model emits the expected tool call name."""
    examples = _read_jsonl(path)
    correct = 0
    parseable = 0
    latencies: List[float] = []
    for ex in examples:
        msgs = _messages_from(ex)
        tools = ex.get("tools")
        t0 = time.perf_counter()
        reply = engine.chat(msgs, tools=tools, max_new_tokens=max_new_tokens, temperature=0.0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        call = parse_tool_call(reply)
        if call is not None:
            parseable += 1
            if call.get("name") == ex.get("expected", {}).get("name"):
                correct += 1
    n = max(len(examples), 1)
    return {
        "tool_call_accuracy": correct / n,
        "parseable_rate": parseable / n,
        "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
        "num_examples": len(examples),
    }


def run(args: argparse.Namespace) -> int:
    config = get_config(args.preset)
    tokenizer = load_tokenizer(args.tokenizer, config.vocab_size)
    engine = HermesInference.from_checkpoint(
        config, args.checkpoint, tokenizer, device=args.device, preset_name=args.preset
    )
    print(engine)
    if args.checkpoint is None:
        print("[info] No checkpoint supplied — evaluating randomly initialized weights (CI mode).")

    eval_path = args.eval_data or os.path.join(_REPO_ROOT, "data", "eval.jsonl")
    tool_path = args.tool_data or os.path.join(_REPO_ROOT, "data", "tool_eval.jsonl")

    perplexity = eval_perplexity(engine, eval_path)
    tool_metrics = eval_tool_calls(engine, tool_path, args.max_new_tokens)

    results = {
        "preset": args.preset,
        "checkpoint": args.checkpoint,
        "perplexity": perplexity,
        **tool_metrics,
    }

    print("\n| metric | value |")
    print("|---|---|")
    print(f"| perplexity | {perplexity:.2f} |")
    print(f"| tool_call_accuracy | {tool_metrics['tool_call_accuracy']:.2%} |")
    print(f"| parseable_rate | {tool_metrics['parseable_rate']:.2%} |")
    print(f"| avg_latency_ms | {tool_metrics['avg_latency_ms']:.1f} |")
    print()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.output}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Hermes perplexity + tool-call accuracy.")
    p.add_argument("--preset", default="hermes-270m", choices=["hermes-1b", "hermes-500m", "hermes-270m"])
    p.add_argument("--checkpoint", default=None, help="Optional .pt checkpoint (random init if omitted).")
    p.add_argument("--tokenizer", default=None, help="Optional SentencePiece model (byte fallback if omitted).")
    p.add_argument("--eval-data", default=None, help="Override path to perplexity JSONL.")
    p.add_argument("--tool-data", default=None, help="Override path to tool-call JSONL.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="eval_results.json")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
