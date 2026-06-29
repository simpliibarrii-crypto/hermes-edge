#!/usr/bin/env python3
"""Train the SentencePiece tokenizer bundled inside the ``.litertlm``.

Reserves the Hermes special tokens (BOS/EOS/PAD and the ChatML + tool-call
sentinels) so ids stay aligned with :class:`hermes.config.HermesConfig` and the
prompt format in :mod:`hermes.chat_template`.

Example::

    python scripts/train_tokenizer.py \
        --input corpus.txt --vocab-size 32000 --output tokenizer/hermes.model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hermes.tokenizer")

# Kept in sync with hermes.chat_template sentinels.
USER_DEFINED_SYMBOLS = [
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train Hermes SentencePiece tokenizer")
    p.add_argument("--input", required=True, help="Plain-text training corpus")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--output", default="tokenizer/hermes.model")
    p.add_argument("--character-coverage", type=float, default=0.9995)
    args = p.parse_args(argv)

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("`pip install sentencepiece` to train a tokenizer.") from exc

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.splitext(args.output)[0]

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=args.character_coverage,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
    )
    logger.info("Tokenizer written to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
