"""Architecture configuration for the Hermes mobile transformer.

The configuration intentionally mirrors the knobs exposed by
``ai_edge_torch.generative.layers.model_config`` so that the same numbers can
drive both the reference PyTorch implementation (used for training) and the
LiteRT conversion path. Keeping a single source of truth avoids the classic
"the converted graph does not match the trained weights" failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HermesConfig:
    """Hyper-parameters for a decoder-only, grouped-query-attention model.

    Attributes:
        vocab_size: SentencePiece vocabulary size (must match the tokenizer).
        hidden_size: Model / embedding dimension.
        intermediate_size: Feed-forward (MLP) inner dimension.
        num_layers: Number of transformer decoder blocks.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads (GQA). Must divide num_heads.
        head_dim: Dimension per attention head.
        max_seq_len: Maximum context window (KV-cache length) in tokens.
        rope_theta: RoPE base frequency.
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        tie_embeddings: Share input embedding and output projection weights.
        pad_token_id / bos_token_id / eos_token_id: Special token ids.
    """

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int = 4
    head_dim: int = 64
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    # Tool-call sentinel tokens reserved in the tokenizer for constrained
    # decoding of function calls (see scripts/train.py chat template).
    tool_call_start_id: Optional[int] = 3
    tool_call_end_id: Optional[int] = 4

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention."
            )
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_heads * head_dim ({self.num_heads * self.head_dim})."
            )

    @property
    def num_query_groups(self) -> int:
        """Heads per KV group (the GQA sharing factor)."""
        return self.num_heads // self.num_kv_heads

    def estimated_parameters(self) -> int:
        """Rough parameter count (weights only, ignoring norms/biases)."""
        emb = self.vocab_size * self.hidden_size
        q = self.hidden_size * self.num_heads * self.head_dim
        kv = 2 * self.hidden_size * self.num_kv_heads * self.head_dim
        o = self.num_heads * self.head_dim * self.hidden_size
        attn = q + kv + o
        mlp = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        per_layer = attn + mlp
        total = emb + self.num_layers * per_layer
        if not self.tie_embeddings:
            total += emb  # separate lm_head
        return total


def hermes_1b_config() -> HermesConfig:
    """~1B parameter variant — the default mobile target (~600 MB at INT4)."""
    return HermesConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=22,
        num_heads=32,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=4096,
    )


def hermes_500m_config() -> HermesConfig:
    """~500M parameter variant — quality/speed sweet spot (~280 MB at INT4)."""
    return HermesConfig(
        vocab_size=32000,
        hidden_size=1536,
        intermediate_size=4096,
        num_layers=24,
        num_heads=24,
        num_kv_heads=6,
        head_dim=64,
        max_seq_len=4096,
    )


def hermes_270m_config() -> HermesConfig:
    """~270M parameter variant — smallest, FunctionGemma-class footprint."""
    return HermesConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_layers=21,
        num_heads=16,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=4096,
    )


# ── Gemma-inspired presets (optimized for iPhone 16 A18 Pro / ANE) ──────────

def gemma_3_1b_config() -> HermesConfig:
    """Gemma 3 1B — Google's latest small model architecture.

    Optimized for on-device inference with Apple Neural Engine:
    - 32k vocab, 26 layers, 2048 hidden dim
    - 16 heads, 8 KV heads (GQA ratio 2:1 — efficient for ANE)
    - 8192 context window for longer conversations
    - Ideal for iPhone 16 A18 Pro at INT4 (~250 MB on disk)
    """
    return HermesConfig(
        vocab_size=32768,
        hidden_size=2048,
        intermediate_size=8192,
        num_layers=26,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=8192,
        rope_theta=10000.0,
    )


def gemma_2_2b_config() -> HermesConfig:
    """Gemma 2 2B — higher quality with shared KV-heads.

    Uses deeper GQA (2 KV heads shared across 16 query heads) for
    memory-efficient inference on iPhone 16 Pro / Pro Max.
    ~1.1 GB at INT4.
    """
    return HermesConfig(
        vocab_size=32768,
        hidden_size=2560,
        intermediate_size=9216,
        num_layers=26,
        num_heads=16,
        num_kv_heads=2,
        head_dim=160,
        max_seq_len=8192,
        rope_theta=10000.0,
    )


# ── DeepSeek-inspired distilled presets ─────────────────────────────────────

def hermes_distilled_1b_config() -> HermesConfig:
    """Distilled 1B model using DeepSeek-R1 reasoning principles.

    Knowledge distilled from Gemma 3 1B teacher. Maintains the same
    architecture as hermes-1b but with extended context and tuned
    for step-by-step reasoning before tool calls.
    """
    return HermesConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_layers=22,
        num_heads=32,
        num_kv_heads=4,
        head_dim=64,
        max_seq_len=8192,
        rope_theta=10000.0,
    )


PRESETS = {
    "hermes-1b": hermes_1b_config,
    "hermes-500m": hermes_500m_config,
    "hermes-270m": hermes_270m_config,
    "gemma-3-1b": gemma_3_1b_config,
    "gemma-2-2b": gemma_2_2b_config,
    "hermes-distilled-1b": hermes_distilled_1b_config,
}


def get_config(name: str) -> HermesConfig:
    """Look up a preset config by name."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]()
