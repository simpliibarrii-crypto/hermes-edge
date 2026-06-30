#!/usr/bin/env python3
"""Knowledge distillation from Gemma 3 1B teacher to Hermes student.

Applies DeepSeek-R1 style distillation: the teacher's logits are softened
by temperature T and the student minimizes a weighted sum of:
  - Cross-entropy against ground-truth labels (hard loss)
  - KL divergence against teacher logits (soft loss)

Usage:
    python scripts/distill_from_gemma.py \
        --teacher google/gemma-3-1b \
        --student-preset hermes-distilled-1b \
        --data data/agentic_sft.jsonl \
        --output checkpoints/hermes-distilled-1b.pt \
        --temperature 3.0 --alpha 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hermes.config import HermesConfig, get_config
from hermes.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hermes.distill")


class ChatDataset(Dataset):
    """JSONL dataset of conversations for distillation."""

    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = " ".join(m.get("content", "") for m in obj.get("messages", []))
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tokenizer.encode(text, out_type=int)[:self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long)


def distill_step(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    input_ids: torch.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Single distillation step: KL(student || teacher) + CE(student, labels).

    Args:
        temperature: Softmax temperature for teacher logits (higher = softer).
        alpha: Weight for KL divergence (1-alpha for CE).
    """
    with torch.no_grad():
        teacher_out = teacher(input_ids)
        teacher_logits = teacher_out["logits"]
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_out = student(input_ids)
    student_logits = student_out["logits"]
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_log_probs.view(-1, student_log_probs.size(-1)),
        teacher_probs.view(-1, teacher_probs.size(-1)),
        reduction="batchmean",
        log_target=False,
    ) * (temperature ** 2)

    labels = input_ids[:, 1:].contiguous()
    shift_logits = student_logits[:, :-1, :].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        ignore_index=0,
    )

    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def run(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load teacher (Gemma 3 1B from HF)
    logger.info("Loading teacher: %s", args.teacher)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.float16, device_map="auto"
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
        teacher.eval()
    except ImportError:
        logger.error(
            "transformers is required for teacher loading. "
            "Install: pip install transformers torch"
        )
        return 1

    # Build student model
    config = get_config(args.student_preset)
    student = build_model(config)
    logger.info("Student: %s (%.0fM params)", args.student_preset, config.estimated_parameters() / 1e6)

    # Load data
    dataset = ChatDataset(args.data, teacher_tokenizer, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info("Data: %d samples", len(dataset))

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Training loop
    student.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            loss = distill_step(
                student, teacher, input_ids,
                temperature=args.temperature, alpha=args.alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                logger.info(
                    "Epoch %d | Step %d | Loss: %.4f | LR: %.2e",
                    epoch + 1, global_step, loss.item(), args.lr,
                )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model": student.state_dict(), "config": config}, args.output)
    logger.info("Distilled model saved: %s", args.output)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill Gemma 3 1B → Hermes")
    p.add_argument("--teacher", default="google/gemma-3-1b", help="Teacher model ID")
    p.add_argument("--student-preset", default="hermes-distilled-1b", help="Student preset")
    p.add_argument("--data", required=True, help="JSONL training data")
    p.add_argument("--output", default="checkpoints/hermes-distilled-1b.pt")
    p.add_argument("--temperature", type=float, default=3.0, help="Softmax temperature")
    p.add_argument("--alpha", type=float, default=0.7, help="KL weight (0-1)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
