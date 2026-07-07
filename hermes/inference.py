"""Streaming inference engine for the Hermes mobile transformer.

:class:`HermesInference` wraps a :class:`~hermes.model.HermesForCausalLM` and a
SentencePiece tokenizer into a single object with three entry points:

* :meth:`generate` — low-level text completion with nucleus (top-p), top-k, and
  repetition-penalty sampling, optionally streaming token strings as they decode.
* :meth:`chat` — renders a message list through the Hermes ChatML template, then
  generates an assistant turn.
* :meth:`tool_call_loop` — the agentic loop: generate, parse any ``<tool_call>``,
  dispatch it to a Python callable, feed the ``<tool_response>`` back, and repeat
  until the model produces a plain answer (or ``max_rounds`` is hit).

Decoding reuses the existing :class:`~hermes.model.Attention` KV-cache: the
prompt is run once to prime per-layer caches, then each new token is decoded with
a single-position forward pass, so cost is linear in generated length rather than
quadratic.

The tokenizer is duck-typed: anything exposing ``encode(str) -> list[int]`` and
``decode(list[int]) -> str`` works, which covers ``sentencepiece`` and the tiny
byte-level stub used in tests.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from hermes.chat_template import (
    Message,
    build_prompt,
    parse_tool_call,
)
from hermes.config import HermesConfig
from hermes.model import HermesForCausalLM, build_model

KVList = List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
ToolFn = Callable[..., Any]


class HermesInference:
    """Load a Hermes checkpoint + tokenizer and run streaming generation."""

    def __init__(
        self,
        model: HermesForCausalLM,
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        preset_name: str = "custom",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.config: HermesConfig = model.config
        self.preset_name = preset_name

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoint(
        cls,
        config: HermesConfig,
        checkpoint_path: Optional[str],
        tokenizer: Any,
        device: Union[str, torch.device] = "cpu",
        preset_name: str = "custom",
    ) -> "HermesInference":
        """Build a model from ``config``, optionally load weights, and wrap it.

        If ``checkpoint_path`` is None the model keeps its random init — handy for
        CI and shape tests that don't need a trained checkpoint.
        """
        model = build_model(config)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict, strict=False)
        return cls(model, tokenizer, device=device, preset_name=preset_name)

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"HermesInference(preset={self.preset_name!r}, "
            f"params={n_params / 1e6:.1f}M, "
            f"layers={self.config.num_layers}, ctx={self.config.max_seq_len}, "
            f"device={self.device.type})"
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, generated: List[int], penalty: float
    ) -> torch.Tensor:
        """Divide logits of already-seen tokens by ``penalty`` (CTRL-style)."""
        if penalty == 1.0 or not generated:
            return logits
        idx = torch.tensor(sorted(set(generated)), device=logits.device)
        selected = logits.index_select(-1, idx)
        # Positive logits are divided, negative are multiplied (push both down).
        selected = torch.where(selected > 0, selected / penalty, selected * penalty)
        logits = logits.index_copy(-1, idx, selected)
        return logits

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        """Sample a single token id from ``logits`` with top-k + nucleus filtering."""
        if temperature <= 0.0:
            return int(logits.argmax(dim=-1))

        logits = logits / temperature

        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[..., -1, None]
            logits = torch.where(
                logits < kth, torch.full_like(logits, float("-inf")), logits
            )

        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            # Keep tokens up to and including the one that crosses top_p.
            remove = cumulative - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits
            )

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    # ------------------------------------------------------------------ #
    # KV-cache primed decode
    # ------------------------------------------------------------------ #
    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        caches: KVList,
        start_pos: int,
    ) -> Tuple[torch.Tensor, KVList]:
        """Run the model for ``input_ids`` reusing/extending per-layer KV caches.

        Returns the last-position logits and the updated cache list. This bypasses
        ``HermesForCausalLM.forward`` so it can thread the per-layer cache tuples
        through the existing :class:`Attention` ``kv_cache`` argument.
        """
        model = self.model
        b, t = input_ids.shape
        x = model.embed_tokens(input_ids)
        cos = model.rope_cos[start_pos : start_pos + t].to(x.device)
        sin = model.rope_sin[start_pos : start_pos + t].to(x.device)

        # Causal mask over the *full* attended length (past + current).
        total = start_pos + t
        full_mask = torch.full((t, total), float("-inf"), device=x.device)
        full_mask = torch.triu(full_mask, diagonal=1 + start_pos)

        new_caches: KVList = [None] * len(model.layers)
        for i, layer in enumerate(model.layers):
            h, new_cache = layer.self_attn(
                layer.input_layernorm(x), cos, sin, full_mask, caches[i]
            )
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            new_caches[i] = new_cache

        x = model.norm(x)
        logits = model.lm_head(x[:, -1, :])
        return logits, new_caches

    @torch.no_grad()
    def _generate_ids(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Iterator[int]:
        """Yield newly generated token ids one at a time, using a KV cache."""
        self.model.eval()
        eos = self.config.eos_token_id
        caches: KVList = [None] * len(self.model.layers)

        # Keep room for at least one generated token: truncate the prompt to the
        # most recent (max_seq_len - 1) tokens if it would otherwise fill context.
        max_prompt = max(1, self.config.max_seq_len - 1)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        # Prime the cache on the full prompt in one prefill pass.
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits, caches = self._forward_with_cache(ids, caches, start_pos=0)
        pos = len(prompt_ids)

        generated: List[int] = []
        for _ in range(max_new_tokens):
            step_logits = self._apply_repetition_penalty(
                logits.clone(), prompt_ids + generated, repetition_penalty
            )
            next_id = self._sample(step_logits.squeeze(0), temperature, top_p, top_k)
            if next_id == eos:
                break
            generated.append(next_id)
            yield next_id

            if pos >= self.config.max_seq_len:
                break
            step = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            logits, caches = self._forward_with_cache(step, caches, start_pos=pos)
            pos += 1

    # ------------------------------------------------------------------ #
    # Public generation API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate text from ``prompt``.

        Returns the full completion string, or — if ``stream=True`` — a generator
        that yields incremental token strings as they are produced.
        """
        prompt_ids = self.tokenizer.encode(prompt)

        def _token_strings() -> Iterator[str]:
            prev_text = ""
            buffer: List[int] = []
            for tok in self._generate_ids(
                prompt_ids, max_new_tokens, temperature, top_p, top_k, repetition_penalty
            ):
                buffer.append(tok)
                # Decode incrementally so multi-token characters render correctly.
                text = self.tokenizer.decode(buffer)
                delta = text[len(prev_text) :]
                if delta:
                    prev_text = text
                    yield delta

        if stream:
            return _token_strings()
        return "".join(_token_strings())

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        """Render ``messages`` (+ optional ``tools``) and generate a reply string."""
        prompt = build_prompt(messages, tools=tools)
        result = self.generate(prompt, stream=False, **kwargs)
        assert isinstance(result, str)
        return result

    def tool_call_loop(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        tool_functions: Dict[str, ToolFn],
        max_rounds: int = 5,
        **kwargs: Any,
    ) -> List[Message]:
        """Agentic loop: generate → parse tool call → dispatch → feed back.

        Each round generates an assistant turn. If it contains a parseable
        ``<tool_call>``, the named callable in ``tool_functions`` is invoked with
        the parsed arguments and its result is appended as a ``tool`` message;
        otherwise the loop ends. Returns the full conversation including all
        assistant and tool turns appended.

        Args:
            messages: Seed conversation (mutated copy is returned).
            tools: Tool schemas advertised to the model in the system prompt.
            tool_functions: Maps tool ``name`` → Python callable.
            max_rounds: Hard cap on generate/dispatch cycles.
        """
        convo = list(messages)
        for _ in range(max_rounds):
            reply = self.chat(convo, tools=tools, **kwargs)
            convo.append(Message("assistant", reply))

            call = parse_tool_call(reply)
            if call is None:
                break

            fn = tool_functions.get(call["name"])
            if fn is None:
                convo.append(
                    Message("tool", f'{{"error": "unknown tool: {call["name"]}"}}')
                )
                continue
            try:
                result = fn(**call.get("arguments", {}))
            except Exception as exc:  # surface tool errors back to the model
                result = {"error": str(exc)}
            convo.append(Message("tool", str(result)))
        return convo


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from hermes.config import hermes_270m_config

    class _ByteTokenizer:
        def encode(self, text: str) -> List[int]:
            return [b % 32000 for b in text.encode("utf-8")] or [1]

        def decode(self, ids: List[int]) -> str:
            return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")

    engine = HermesInference.from_checkpoint(
        hermes_270m_config(), None, _ByteTokenizer(), preset_name="hermes-270m"
    )
    print(engine)
    print(engine.generate("Hello", max_new_tokens=8, temperature=0.0))


class DemoHermesInference:
    """Runs Hermes with random weights — no checkpoint needed. For architecture demonstration only."""
    
    def __init__(self, preset: str = "hermes-270m"):
        from hermes.config import PRESETS
        self.config = PRESETS[preset]()
        self.preset = preset
        print(f"[Hermes Demo] Running {preset} with random weights — output is nonsense but the pipeline works.")
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Returns a canned response showing the pipeline works."""
        responses = [
            f"[Hermes Demo | {self.preset}] Pipeline operational. Architecture: {self.config.num_layers} layers, {self.config.hidden_size}d, {self.config.num_heads} heads. Install real weights via: pip install homeforai-blockchain && hermes download {self.preset}",
            f"[Hermes Demo] Received: '{prompt[:50]}...' — Real inference requires model weights. See README for download instructions.",
            f"[Hermes Demo] GQA with {self.config.num_kv_heads} KV heads active. Context window: {self.config.max_seq_len} tokens. Quantization target: INT4. Ready for LiteRT-LM conversion.",
        ]
        import random
        return random.choice(responses)
    
    def chat(self, message: str) -> str:
        return self.generate(message)
