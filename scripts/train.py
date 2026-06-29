#!/usr/bin/env python3
"""Lightweight training / fine-tuning for the Hermes mobile transformer.

Designed for the mobile model sizes (270M / 1B), this script does supervised
fine-tuning on agentic chat data formatted with the Hermes tool-calling
template (see :mod:`hermes.chat_template`). It is deliberately framework-light
(plain PyTorch + an optional JSONL dataset) so it runs on a single GPU or even
CPU for smoke tests, and produces a checkpoint that
``scripts/convert_to_litertlm.py`` can consume directly.

Dataset format (JSONL), one conversation per line::

    {"messages": [
        {"role": "user", "content": "What is 12*9?"},
        {"role": "assistant", "content": "<tool_call>{\"name\":\"calculator\",\"arguments\":{\"expression\":\"12*9\"}}</tool_call>"},
        {"role": "tool", "content": "108"},
        {"role": "assistant", "content": "12 * 9 = 108."}
     ],
     "tools": [{"name": "calculator", "description": "...", "parameters": {...}}]}

Example::

    python scripts/train.py \
        --preset hermes-1b \
        --data data/agentic_sft.jsonl \
        --tokenizer tokenizer/hermes.model \
        --output checkpoints/hermes-1b.pt \
        --epochs 1 --batch-size 4 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.chat_template import Message, build_prompt  # noqa: E402
from hermes.config import get_config  # noqa: E402
from hermes.model import build_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hermes.train")


def load_tokenizer(path: str):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError(
            "sentencepiece is required for training. `pip install sentencepiece`."
        ) from exc
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. Train one with "
            "scripts/train_tokenizer.py first."
        )
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def encode_example(example: Dict[str, Any], sp, max_len: int) -> List[int]:
    messages = [Message(m["role"], m["content"]) for m in example["messages"]]
    tools = example.get("tools")
    prompt = build_prompt(messages, tools=tools, add_generation_prompt=False)
    ids = sp.encode(prompt, out_type=int)
    return ids[:max_len]


class JsonlDataset:
    """Tiny map-style dataset over a JSONL agentic-chat file."""

    def __init__(self, path: str, sp, max_len: int) -> None:
        self.examples: List[List[int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = encode_example(json.loads(line), sp, max_len)
                if len(ids) >= 2:
                    self.examples.append(ids)
        logger.info("Loaded %d training examples from %s", len(self.examples), path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.examples[idx]


def collate(batch: List[List[int]], pad_id: int):
    import torch

    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return input_ids


def train(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader

    config = get_config(args.preset)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info("Training %s on %s (~%.0fM params)", args.preset, device,
                config.estimated_parameters() / 1e6)

    sp = load_tokenizer(args.tokenizer)
    if sp.get_piece_size() != config.vocab_size:
        logger.warning(
            "Tokenizer vocab (%d) != config vocab (%d); using tokenizer size.",
            sp.get_piece_size(), config.vocab_size,
        )
        config.vocab_size = sp.get_piece_size()

    dataset = JsonlDataset(args.data, sp, config.max_seq_len)
    if len(dataset) == 0:
        logger.error("No usable training examples; aborting.")
        return 1
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, config.pad_token_id),
    )

    model = build_model(config).to(device)
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        logger.info("Initialized from %s", args.init_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()

    step = 0
    for epoch in range(args.epochs):
        for input_ids in loader:
            input_ids = input_ids.to(device)
            out = model(input_ids, labels=input_ids)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % args.log_every == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch, step, loss.item())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config.__dict__},
        args.output,
    )
    logger.info("Saved checkpoint to %s", args.output)
    logger.info(
        "Next: python scripts/convert_to_litertlm.py --checkpoint %s "
        "--tokenizer %s --preset %s",
        args.output, args.tokenizer, args.preset,
    )
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/fine-tune Hermes mobile model")
    p.add_argument("--preset", default="hermes-1b", choices=["hermes-1b", "hermes-270m"])
    p.add_argument("--data", required=True, help="Path to agentic-chat JSONL dataset")
    p.add_argument("--tokenizer", required=True, help="SentencePiece .model path")
    p.add_argument("--output", default="checkpoints/hermes-1b.pt")
    p.add_argument("--init-checkpoint", default=None, help="Optional warm-start checkpoint")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(train(parse_args()))
