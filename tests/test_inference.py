"""Tests for the streaming inference engine (no LiteRT stack, random weights)."""

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from hermes.chat_template import Message  # noqa: E402
from hermes.config import HermesConfig  # noqa: E402
from hermes.inference import HermesInference  # noqa: E402
from hermes.model import build_model  # noqa: E402


class _StubTokenizer:
    """Tiny tokenizer whose decode is always non-empty for non-empty input."""

    def encode(self, text: str) -> List[int]:
        return [(ord(c) % 50) + 5 for c in text] or [5]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(33 + (i % 90)) for i in ids)


def _engine():
    # eos_token_id=-1 is unreachable, so generation always runs to max_new_tokens.
    cfg = HermesConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=1024, eos_token_id=-1,
    )
    return HermesInference(build_model(cfg), _StubTokenizer(), preset_name="test")


def test_streaming_tokens():
    engine = _engine()
    stream = engine.generate("hello", max_new_tokens=6, temperature=0.7, stream=True)
    chunks = list(stream)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_chat_returns_string():
    engine = _engine()
    reply = engine.chat([Message("user", "hi there")], max_new_tokens=6, temperature=0.0)
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_tool_call_loop_terminates():
    engine = _engine()
    calls = {"n": 0}

    def fake_tool(**kwargs):
        calls["n"] += 1
        return "42"

    convo = engine.tool_call_loop(
        [Message("user", "what is 6 times 7?")],
        tools=[{"name": "calculator", "description": "math"}],
        tool_functions={"calculator": fake_tool},
        max_rounds=3,
        max_new_tokens=6,
        temperature=0.0,
    )
    # Random weights won't emit a valid tool call, so the loop ends quickly and
    # never exceeds the round cap (each round appends at most one assistant +
    # one tool message).
    assistant_turns = [m for m in convo if m.role == "assistant"]
    assert 1 <= len(assistant_turns) <= 3
    assert convo[-1].role in ("assistant", "tool")


def test_repetition_penalty_applied():
    logits = torch.tensor([[2.0, 4.0, -3.0, 1.0]])
    seen = [1, 2]  # token 1 (positive) divided, token 2 (negative) multiplied
    out = HermesInference._apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
    assert out[0, 1].item() == pytest.approx(2.0)   # 4.0 / 2.0
    assert out[0, 2].item() == pytest.approx(-6.0)  # -3.0 * 2.0
    assert out[0, 0].item() == pytest.approx(2.0)   # unseen, unchanged
    assert out[0, 3].item() == pytest.approx(1.0)   # unseen, unchanged
