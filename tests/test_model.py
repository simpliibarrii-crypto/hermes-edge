"""Smoke tests for the Hermes mobile model and tooling.

These tests avoid the heavy LiteRT stack so they run anywhere torch is present.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.chat_template import (  # noqa: E402
    Message,
    build_prompt,
    format_tool_call,
    parse_tool_call,
)
from hermes.config import (  # noqa: E402
    HermesConfig,
    get_config,
    hermes_1b_config,
    hermes_270m_config,
    hermes_500m_config,
)

torch = pytest.importorskip("torch")

from hermes.model import build_model  # noqa: E402


def test_config_invariants():
    cfg = hermes_1b_config()
    assert cfg.hidden_size == cfg.num_heads * cfg.head_dim
    assert cfg.num_heads % cfg.num_kv_heads == 0
    # ~1B target, within a sane band.
    params = cfg.estimated_parameters()
    assert 0.8e9 < params < 1.3e9


def test_config_rejects_bad_gqa():
    with pytest.raises(ValueError):
        HermesConfig(hidden_size=2048, num_heads=32, num_kv_heads=5, head_dim=64)


def test_get_config_presets():
    assert get_config("hermes-270m").num_layers == hermes_270m_config().num_layers
    assert get_config("hermes-500m").num_layers == hermes_500m_config().num_layers
    with pytest.raises(KeyError):
        get_config("nope")


def test_hermes_500m_config():
    cfg = hermes_500m_config()
    assert cfg.hidden_size == 1536
    assert cfg.num_heads == 24 and cfg.num_kv_heads == 6
    assert cfg.num_layers == 24
    assert cfg.hidden_size == cfg.num_heads * cfg.head_dim
    assert "hermes-500m" in get_config.__globals__["PRESETS"]


def test_forward_and_loss():
    cfg = HermesConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=16, max_seq_len=32,
    )
    model = build_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    assert out["loss"].item() > 0


def test_generate_runs():
    cfg = HermesConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=16, max_seq_len=32,
    )
    model = build_model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=5, temperature=0.0)
    assert out.shape[1] == 9


def test_tool_call_roundtrip():
    text = "sure " + format_tool_call("calculator", {"expression": "2+2"})
    call = parse_tool_call(text)
    assert call["name"] == "calculator"
    assert call["arguments"]["expression"] == "2+2"


def test_parse_tool_call_absent():
    assert parse_tool_call("just a normal answer") is None


def test_build_prompt_includes_tools():
    msgs = [Message("user", "hi")]
    tools = [{"name": "calculator", "description": "math"}]
    prompt = build_prompt(msgs, tools=tools)
    assert "<|im_start|>system" in prompt
    assert "calculator" in prompt
    assert prompt.rstrip().endswith("<|im_start|>assistant")
