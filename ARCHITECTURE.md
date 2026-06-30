# Hermes Edge v2 — Enhanced Architecture

## Executive Summary

```ascii
┌─────────────────────────────────────────────────────────────────────┐
│                     Hermes Edge v2 Architecture                      │
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────────┐  │
│  │ HF Model │──▶│  CPU-Wise │──▶│  .litertlm │──▶│  iPhone 16     │  │
│  │ Qwen3-0.6B│  │ Converter │   │  Bundle    │   │  AI Edge       │  │
│  └──────────┘   └──────────┘   └─────┬──────┘   │  Gallery       │  │
│                                       │          └────────┬───────┘  │
│  ┌──────────┐   ┌──────────┐         │                   │          │
│  │ Draft    │──▶│ Draft    │─────────┘                   │          │
│  │ Model    │   │ Verifier │                             │          │
│  └──────────┘   └──────────┘                             │          │
│                                                           │          │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐             │          │
│  │ Agent    │◀─▶│ Tool     │◀─▶│ Memory    │             │          │
│  │ Loop     │   │ Registry │   │ Store     │             │          │
│  └──────────┘   └──────────┘   └───────────┘             │          │
│                                                           ▼          │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────────┐              │
│  │ DeepSeek │──▶│ Thinking │──▶│ Tool-Augmented      │              │
│  │ Reasoner │   │ Trace    │   │ Generation (TAG)    │              │
│  └──────────┘   └──────────┘   └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## A. Model Pipeline — HF → .litertlm on CPU (2.7GB RAM, no GPU)

### Challenge
Qwen3-0.6B is 586 MB at INT4. Full FP16 weights are ~1.2 GB. `ai_edge_torch` conversion normally requires ~8GB+ RAM. We need to fit in 2.7GB.

### Strategy: Stage-wise conversion with memory pooling

```ascii
Stage 1: Download & Shrink ─────────────────────────────────────
  HF Qwen3-0.6B (FP16 ~1.2GB)
      │
      ▼
  apply_weight_only_int4() → in-place STE quant → ~350 MB in RAM
      │
      ▼
  Save as checkpoint.pt (state_dict only, no optimizer)
      │ (~350 MB on disk)
      ▼

Stage 2: ai_edge_torch Build & Load ────────────────────────────
  build_ai_edge_model(config) → ~200 MB (uninitialized)
      │
      ▼
  Load checkpoint via memory-mapped state_dict
  Use torch.load(..., mmap=True) → ~200 MB peak
      │
      ▼

Stage 3: Trace & Lower ─────────────────────────────────────────
  converter.convert_to_tflite(
      prefill_seq_len=[1024, 1],  # shorter prefill = less peak
      quantize=full_int4_dynamic_recipe(),
  )
      │ (~500 MB temporary TFLite)
      ▼

Stage 4: Bundle ────────────────────────────────────────────────
  litert_lm.bundler.create_bundle(
      tflite_model=...,
      tokenizer=...,
      output=dist/hermes-mobile-qwen3-0.6b.litertlm,
  )
      │
      ▼
  Final .litertlm (~586 MB)
```

### New file: `scripts/convert_qwen.py`

Converts Qwen3-0.6B with CPU-optimized settings:

```
python scripts/convert_qwen.py \
    --hf-model Qwen/Qwen3-0.6B \
    --preset qwen3-0.6b \
    --output dist/hermes-mobile-qwen3-0.6b-int4.litertlm \
    --low-memory \              # enables mmap + stage-wise GC
    --max-prefill 1024 \         # shorter prefill for RAM savings
    --dtype fp32 \               # force fp32 accumulation (no GPU)
    --gc-collect-between         # explicit gc between stages
```

### Memory Budget (2.7 GB total)

| Step | Peak RSS | Cumulative |
|------|----------|------------|
| HF model load (fp16, mmap) | 0 MB (disk-mapped) | 0 MB |
| PTQ calibration (4 batches) | ~200 MB | 200 MB |
| INT4 weight quant in-place | ~200 MB | 400 MB |
| ai_edge_torch model build | ~200 MB | 600 MB |
| Weight load + remap | ~200 MB | 800 MB |
| TFLite lowering | ~1200 MB | 2000 MB |
| TFLite → .litertlm | ~300 MB | 2300 MB |
| Headroom | 400 MB | 2700 MB |

### New config presets in `hermes/config.py`

```python
def qwen3_0_6b_config() -> HermesConfig:
    """Qwen3-0.6B architecture mapped to HermesConfig."""
    return HermesConfig(
        vocab_size=151936,       # Qwen3 vocabulary
        hidden_size=2048,
        intermediate_size=8192,  # SwiGLU: 3 * hidden
        num_layers=28,
        num_heads=32,
        num_kv_heads=4,          # GQA 8:1
        head_dim=64,
        max_seq_len=32768,       # Qwen3 supports 32K context
        rope_theta=1000000.0,    # Qwen3's RoPE base freq
        rms_norm_eps=1e-6,
        tie_embeddings=False,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
        tool_call_start_id=151646,  # reserved sentinel
        tool_call_end_id=151647,
    )
```

### Weight remapping (`convert_qwen.py`)

Qwen3 uses `model.layers.{i}.self_attn.{q,k,v,o}_proj` → fuses to `atten_func.qkv_projection` same as existing `remap_state_dict`. New mapping for Qwen3-specific naming:

| Qwen3 HF name | ai_edge_torch name |
|---------------|-------------------|
| `model.embed_tokens.weight` | `tok_embedding.weight` |
| `model.layers.{i}.self_attn.q_proj.weight` | `transformer_blocks.{i}.atten_func.qkv_projection.weight` (concat q,k,v) |
| `model.layers.{i}.self_attn.k_proj.weight` | ↑ same concat |
| `model.layers.{i}.self_attn.v_proj.weight` | ↑ same concat |
| `model.layers.{i}.self_attn.o_proj.weight` | `transformer_blocks.{i}.atten_func.output_projection.weight` |
| `model.layers.{i}.mlp.gate_proj.weight` | `transformer_blocks.{i}.ff.w1.weight` |
| `model.layers.{i}.mlp.up_proj.weight` | `transformer_blocks.{i}.ff.w3.weight` |
| `model.layers.{i}.mlp.down_proj.weight` | `transformer_blocks.{i}.ff.w2.weight` |
| `model.layers.{i}.input_layernorm.weight` | `transformer_blocks.{i}.pre_atten_norm.weight` |
| `model.layers.{i}.post_attention_layernorm.weight` | `transformer_blocks.{i}.post_atten_norm.weight` |
| `model.norm.weight` | `final_norm.weight` |
| `lm_head.weight` | `lm_head.weight` |

---

## B. Inference Engine — Streaming, DeepSeek Reasoning, Tool Calling

### Architecture

```ascii
┌──────────────────────────────────────────────────────────────────┐
│                     InferenceEngine v2                            │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ LiteRT-LM    │     │ Reasoning    │     │ Constrained     │  │
│  │ Runtime      │────▶│ Pipeline     │────▶│ Decoder         │  │
│  │ (.litertlm)  │     │ (think/tell) │     │ (tool schema)   │  │
│  └──────┬───────┘     └──────┬───────┘     └────────┬────────┘  │
│         │                    │                      │           │
│         ▼                    ▼                      ▼           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Token Stream (AsyncIterator)                │    │
│  │  [token, token, ..., <think>, ..., </think>, ...,  ]    │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  StreamProcessor                                         │    │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────────────┐      │    │
│  │  │ Detoken  │  │ Reason    │  │ Tool Call        │      │    │
│  │  │ & Buffer │  │ Extractor │  │ Parser & Router  │      │    │
│  │  └──────────┘  └───────────┘  └──────────────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### New file: `hermes/reasoning.py` — DeepSeek V4 Flash Reasoning

DeepSeek V4 Flash reasoning uses a **thinking trace** pattern:

```
User: What is 234 * 567?

Assistant: <think>
Let me break this down step by step...
234 * 500 = 117,000
234 * 60 = 14,040
234 * 7 = 1,638
Sum: 117,000 + 14,040 + 1,638 = 132,678
</think>

The answer is 132,678.
```

Key interface:

```python
@dataclass
class ReasoningConfig:
    enabled: bool = True
    think_tag: str = "<think>"
    end_think_tag: str = "</think>"
    max_think_tokens: int = 512
    separate_in_stream: bool = True  # yield think vs answer separately
    think_speed_factor: float = 2.0  # show thinking faster

class ReasoningPipeline:
    """
    Wraps token generation with DeepSeek-style think/tell separation.

    The model is prompted with a system message that asks it to reason
    inside <think> tags before answering. The pipeline:
    1. Detects entry into <think> mode
    2. Collects thinking trace tokens
    3. Detects exit into </think> → answer mode
    4. Yields (type, text) tuples: ("think", "...") or ("answer", "...")
    """

    def __init__(self, config: ReasoningConfig):
        ...

    def process_stream(
        self, token_stream: Iterator[str]
    ) -> Iterator[Tuple[str, str]]:
        """
        Yields ("think", str) while inside <think>...</think>
        Yields ("answer", str) when outside.
        """
        ...

    def inject_reasoning_prompt(
        self, messages: List[Message]
    ) -> List[Message]:
        """Adds system-level reasoning instruction."""
        ...
```

### Inference integration (`hermes/inference.py` — rewritten)

The new `InferenceEngine` combines LiteRT-LM runtime with all pipeline stages:

```python
class LiteRTInference:
    """
    Runs the .litertlm model via LiteRT-LM Python bindings.

    Unlike the old HermesInference (which used PyTorch), this directly
    interfaces with the on-device runtime, making it suitable for both
    desktop testing (via litert_lm) and mobile deployment (identical API).
    """

    def __init__(
        self,
        model_path: str,            # path to .litertlm
        runtime: str = "litert",    # "litert" | "xnnpack" | "coreml"
        max_seq_len: int = 4096,
    ):
        self.model = litert_lm.LiteRTModel(model_path)
        self.cache = self.model.create_kv_cache(max_seq_len)

    def generate_stream(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        reasoning: bool = True,         # DeepSeek reasoning mode
        speculative: bool = True,        # DSpark draft verification
        stream: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """
        Primary generation entry point.

        Yields dicts with keys:
          - "type": "think" | "answer" | "tool_call" | "tool_result" | "error"
          - "text": str  (detokenized chunk)
          - "tokens": int  (cumulative count)
          - "speed": float  (tok/s for this chunk)
        """
        ...
```

### LiteRT-LM Python API binding pattern

The LiteRT-LM runtime exposes this C API via Python ctypes/ffi:

```python
# Pseudocode for how we interact with LiteRT-LM on device
class LiteRTRuntime:
    def prefill(self, tokens: List[int]) -> np.ndarray:
        """Run prefill, returns logits for last token. Populates KV cache."""

    def decode(self, token: int) -> np.ndarray:
        """Single-token decode with existing KV cache. Returns logits."""

    def reset_kv_cache(self):
        """Clear KV cache for new conversation."""
```

---

## C. Agent Framework — Hermes-Style Tool Calling

### Architecture

```ascii
┌────────────────────────────────────────────────────────────────────┐
│                      AgentLoop                                      │
│                                                                    │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐ │
│  │ System     │   │ Generate   │   │ Parse      │   │ Execute  │ │
│  │ Prompt     │──▶│ Response   │──▶│ Tool Calls  │──▶│ Tools    │ │
│  │ Builder    │   │ (with      │   │ (supports  │   │ (sandbox │ │
│  │            │   │  reasoning)│   │  parallel) │   │  + retry)│ │
│  └────────────┘   └────────────┘   └────────────┘   └────┬─────┘ │
│         ▲                                                 │       │
│         │                    ┌───────────┐                │       │
│         └────────────────────│ Memory    │◀───────────────┘       │
│                              │ Store     │                        │
│                              │ (persist) │                        │
│                              └───────────┘                        │
│                                                                    │
│  ┌────────────┐   ┌────────────┐   ┌─────────────────────────┐   │
│  │ Tool       │   │ Tool       │   │ Tool                    │   │
│  │ Registry   │──▶│ Schema     │──▶│ Dispatcher              │   │
│  │ (global)   │   │ Generator  │   │ (async, timeout, retry) │   │
│  └────────────┘   └────────────┘   └─────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### New file: `hermes/agent.py`

```python
@dataclass
class ToolDefinition:
    """JSON Schema tool definition matching OpenAI function calling format."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object
    required: List[str]
    handler: Optional[Callable] = None  # Python handler (desktop)
    skill_url: Optional[str] = None     # AI Edge Gallery skill URL (mobile)

class AgentLoop:
    """
    Hermes agent with parallel tool calling, retry, and persistent memory.

    Flow per round:
    1. Build prompt from conversation history + tool schemas
    2. Run LiteRTInference.generate_stream() with reasoning=True
    3. Parse tool calls from the output (supports multiple parallel calls)
    4. For each tool call:
       a. Look up handler in registry
       b. Execute with timeout & retry
       c. Collect result
    5. Append tool results to conversation
    6. Loop until no more tool calls or max_rounds reached
    """

    def __init__(
        self,
        inference: LiteRTInference,
        tokenizer: Any,
        tool_registry: ToolRegistry,
        memory: MemorySystem,
        max_rounds: int = 10,
        parallel_tools: bool = True,
    ):
        ...

    async def run(
        self,
        user_input: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Full agent loop. Yields events:
          {"type": "think", "content": "..."}
          {"type": "answer", "content": "..."}
          {"type": "tool_call", "name": "...", "args": {...}}
          {"type": "tool_result", "name": "...", "result": ...}
          {"type": "error", "content": "..."}
          {"type": "done", "content": "...", "usage": {...}}
        """
        ...

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """
        Extract all tool calls from model output.
        Supports both single and parallel formats:

        Single:  <tool_call>{...}</tool_call>
        Parallel: <tool_calls>
                    <tool_call>{...}</tool_call>
                    <tool_call>{...}</tool_call>
                  </tool_calls>
        """
        ...

    def _build_tool_system_prompt(self, tools: List[ToolDefinition]) -> str:
        """Build Hermes-style tool description for the system prompt."""
        ...
```

### New file: `hermes/tool_registry.py`

```python
class ToolRegistry:
    """
    Global tool registry with schema generation.

    Tools can be registered either:
    - As Python callables (for desktop testing)
    - As AI Edge Gallery Skill URLs (for mobile deployment)
    """

    def register(self, tool: ToolDefinition): ...
    def unregister(self, name: str): ...
    def get_schema(self, name: str) -> Dict: ...
    def get_all_schemas(self) -> List[Dict]: ...
    def dispatch(self, name: str, arguments: Dict) -> Any:
        """Execute tool with timeout and error handling."""
        ...
```

### New file: `hermes/memory.py`

```python
class MemorySystem:
    """
    Persistent agent memory with retrieval.

    Stores conversation summaries, facts, and user preferences
    that persist across sessions. Uses a lightweight semantic
    indexing approach (simple TF-IDF or miniLM embeddings via
    the model's own hidden states).

    Memory is injected into the system prompt as context.
    """

    def store(self, key: str, value: str, metadata: Dict = {}): ...
    def recall(self, query: str, top_k: int = 5) -> List[Dict]: ...
    def summarize_conversation(self, messages: List[Message]) -> str: ...
    def get_context_prompt(self, query: str) -> str:
        """Returns memory context to inject into system prompt."""
        ...
```

### Tool Calling Format (NousResearch hermes-agent pattern)

```
Hermes Agent tool format:

<tool_calls>
<tool_call>
{"name": "calculator", "arguments": {"expression": "234*567"}}
</tool_call>
<tool_call>
{"name": "web_search", "arguments": {"query": "current weather London"}}
</tool_call>
</tool_calls>
```

The model is trained to emit parallel `<tool_call>` blocks inside a `<tool_calls>` wrapper. Each call is a JSON object with `name` and `arguments`, matching the Hermes function calling standard.

---

## D. DSpark Speculative Decoding Draft Model

### Theory

Speculative decoding accelerates autoregressive generation by:
1. **Draft**: Small model predicts k tokens in one forward pass
2. **Verify**: Large model evaluates all k tokens in parallel
3. **Accept**: Accept tokens where distributions match, resample at first rejection

```
Without Draft:     [tok1] → [tok2] → [tok3] → [tok4] → [tok5]  (5 steps)
With DSpark:       [tok1 tok2 tok3 tok4]                         (1 verify step)
                   [─draft─▶][──────verify──────]
                   Accept 3/4 → draft again from accepted prefix
```

### Architecture

```ascii
┌──────────────────────────────────────────────────────────────────────┐
│                        DSpark Speculative Decoder                     │
│                                                                      │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐     │
│  │ Main Model │     │ Draft Model    │     │ Acceptance       │     │
│  │ 270M INT4  │     │ 30M INT4       │     │ Criterion        │     │
│  │ ~55 tok/s  │     │ ~300 tok/s     │     │                  │     │
│  └──────┬─────┘     └───────┬────────┘     └────────┬─────────┘     │
│         │                   │                        │              │
│         ▼                   ▼                        ▼              │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    Speculative Loop                        │       │
│  │                                                           │       │
│  │  1. Draft model autoregressively produces k=4 tokens      │       │
│  │     (using its own small KV cache)                        │       │
│  │                                                           │       │
│  │  2. Main model prefill-fills all k draft tokens in one    │       │
│  │     forward pass (extending its KV cache)                 │       │
│  │                                                           │       │
│  │  3. Compare draft vs main logits at each position:        │       │
│  │     - If draft token == argmax(main_logits): ACCEPT       │       │
│  │     - If draft token != argmax(main_logits): REJECT       │       │
│  │       and resample from main distribution + truncated     │       │
│  │       draft distribution                                  │       │
│  │                                                           │       │
│  │  4. Repeat from the last accepted position                │       │
│  └──────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
```

### New file: `dspark/draft_model_arch.py`

```python
class DraftModelConfig:
    """
    Ultra-light draft model for speculative decoding.

    Architecture: 4-layer, 4-head transformer with tied embeddings.
    ~30M params → ~60 MB at INT4 → ~300 tok/s on iPhone 16 ANE.
    """
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 64
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
```

### New file: `dspark/draft_verify.py`

```python
class DraftVerifyEngine:
    """
    Core speculative decoding loop.

    Manages two KV caches (draft and main), runs the draft-verify cycle,
    and handles acceptance/rejection logic.
    """

    def __init__(
        self,
        main_model: LiteRTRuntime,
        draft_model: LiteRTRuntime,
        draft_k: int = 5,          # tokens to speculate
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ):
        self.main = main_model
        self.draft = draft_model
        self.draft_k = draft_k
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.draft_cache = KVCache(...)
        self.main_cache = KVCache(...)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
    ) -> Iterator[int]:
        """
        Generate tokens with speculative decoding.

        Yields accepted token IDs one at a time.
        Internal flow:
          1. Prefill both models with prompt
          2. Loop:
             a. Draft k tokens autoregressively (draft model)
             b. Main model forward on all k tokens (single pass)
             c. Compare & accept/reject each position
             d. Yield accepted tokens
             e. Reset draft cache to last accepted position
        """
        ...

    def _verify(
        self,
        draft_tokens: List[int],
        main_logits: np.ndarray,   # [k, vocab_size]
        draft_logits: np.ndarray,  # [k, vocab_size]
    ) -> Tuple[List[int], Optional[int]]:
        """
        Verify each draft token against main model logits.

        Returns: (accepted_tokens, rejected_position_or_None)
        Uses the standard rejection sampling criterion from
        Leviathan et al. "Fast Inference from Transformers via
        Speculative Decoding" (2022).
        """
        ...
```

### New file: `dspark/acceptance.py`

```python
def rejection_sample(
    main_logits: np.ndarray,    # [vocab_size]
    draft_logits: np.ndarray,   # [vocab_size]
    draft_token: int,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, int]:
    """
    Standard speculative decoding acceptance criterion.

    Accept draft_token with probability min(1, p_main / p_draft).
    On rejection, resample from max(0, p_main - p_draft) distribution.
    """
    ...

def greedy_accept(
    main_logits: np.ndarray,
    draft_token: int,
) -> Tuple[bool, int]:
    """
    Greedy acceptance: accept iff draft_token == argmax(main_logits).
    On rejection, return argmax(main_logits) as replacement.

    Faster than rejection sampling, slightly lower acceptance rate.
    This is the recommended mode for INT4 mobile deployment.
    """
    ...
```

### Bundling Draft Model

The draft model is bundled **inside** the same `.litertlm` file as a second signature:

```python
# In scripts/convert_to_litertlm.py (extended)
def bundle_with_draft(
    main_tflite: str,
    draft_tflite: str,
    tokenizer_path: str,
    output_path: str,
    config: HermesConfig,
) -> str:
    """
    Bundle main model + draft model + tokenizer into single .litertlm.

    The .litertlm container supports multiple TFLite graphs as
    named signatures:
      - "prefill": main model prefill
      - "decode": main model decode
      - "draft_prefill": draft model prefill
      - "draft_decode": draft model decode
    """
    from litert_lm import bundler

    bundler.create_bundle(
        tflite_models={
            "prefill": main_tflite.replace(".tflite", "_prefill.tflite"),
            "decode": main_tflite.replace(".tflite", "_decode.tflite"),
            "draft_prefill": draft_tflite.replace(".tflite", "_prefill.tflite"),
            "draft_decode": draft_tflite.replace(".tflite", "_decode.tflite"),
        },
        tokenizer=tokenizer_path,
        output=output_path,
        metadata={"speculative_decoding": True, "draft_k": 5},
    )
```

### Training the Draft Model: `scripts/train_draft.py`

```
python scripts/train_draft.py \
    --teacher dist/hermes-mobile-270m-int4.litertlm \
    --student-config draft-30m \
    --data data/agentic_sft.jsonl \
    --output checkpoints/draft-30m.pt \
    --temperature 2.0 \
    --lr 1e-3 \
    --epochs 5
```

The draft model is trained via **distribution distillation**: minimize KL(teacher || draft) over the teacher's full vocabulary distribution. This teaches the draft model to match the teacher's token preferences, maximizing acceptance rate.

**Outputs**: `checkpoints/draft-30m.pt` → converted to `dist/draft-30m-int4.tflite` via same `convert_to_litertlm.py` pipeline.

---

## E. Deployment — iPhone 16 via AI Edge Gallery

### Package Structure

```ascii
hermes-mobile-v2.litertlm  (single file, ~650 MB)
├── Signature: "prefill"     →  Main model prefill (TFLite)
├── Signature: "decode"      →  Main model decode (TFLite)
├── Signature: "draft_prefill" → Draft model prefill (TFLite)
├── Signature: "draft_decode" → Draft model decode (TFLite)
├── Tokenizer: SentencePiece .model
├── Metadata:
│   ├── model_name: "Hermes Edge v2"
│   ├── quantization: "int4_per_channel"
│   ├── context_length: 4096
│   ├── speculative_decoding: true
│   ├── draft_k: 5
│   ├── agentic: true
│   ├── tools: ["calculator", "web_search", "memory", "timer"]
│   ├── reasoning: true
│   └── runtime_info:
│       ├── min_ios: "18.0"
│       ├── min_device: "iPhone 16"
│       └── delegate: "coreml"
```

### Gallery Import

Users import via URL or file:

```
https://huggingface.co/bclermo/hermes-edge/resolve/main/hermes-mobile-v2.litertlm
```

### iOS Runtime Layer (Swift pseudocode for Xcode project)

```swift
// HermesEdgeAgent.swift — The on-device agent runtime

import LiteRTLM

class HermesEdgeAgent {
    let model: LiteRTLModel
    let tokenizer: SentencePieceTokenizer
    let draftModel: LiteRTLModel?  // optional, for speculative decoding

    init(bundlePath: String) throws {
        self.model = try LiteRTLModel(path: bundlePath, signature: "decode")
        self.tokenizer = try SentencePieceTokenizer(path: bundlePath)
        if model.hasSignature("draft_decode") {
            self.draftModel = try LiteRTLModel(path: bundlePath, signature: "draft_decode")
        }
    }

    func generate(
        messages: [Message],
        tools: [ToolDefinition]?,
        onToken: (TokenEvent) -> Void,
        completion: (Result<String, Error>) -> Void
    ) {
        let prompt = buildHermesPrompt(messages, tools: tools)

        // Prefill
        let tokens = tokenizer.encode(prompt)
        model.runSignature("prefill", input: tokens)

        // Generate loop with optional speculative decoding
        if let draft = draftModel {
            speculativeGenerate(draft: draft, onToken: onToken, completion: completion)
        } else {
            standardGenerate(onToken: onToken, completion: completion)
        }
    }

    private func speculativeGenerate(
        draft: LiteRTLModel,
        onToken: (TokenEvent) -> Void,
        completion: (Result<String, Error>) -> Void
    ) {
        let draftK = 5
        var acceptedTokens: [Int] = []

        while acceptedTokens.count < maxTokens {
            // Draft: run draft model autoregressively
            var draftTokens: [Int] = []
            for _ in 0..<draftK {
                let draftLogits = draft.runSignature("draft_decode", input: lastToken)
                draftTokens.append(sample(draftLogits))
            }

            // Verify: run main model on all draft tokens in one prefill
            let mainLogits = model.runSignature("prefill", input: draftTokens)
            // mainLogits shape: [draftK, vocabSize]

            // Accept/reject each token
            for i in 0..<draftK {
                if greedy_accept(mainLogits[i], draftTokens[i]) {
                    acceptedTokens.append(draftTokens[i])
                    onToken(.token(tokenizer.decode([draftTokens[i]])))
                } else {
                    acceptedTokens.append(argmax(mainLogits[i]))
                    onToken(.token(tokenizer.decode([argmax(mainLogits[i])])))
                    break  // stop at first rejection
                }
            }
        }

        completion(.success(tokenizer.decode(acceptedTokens)))
    }
}
```

### Performance Targets (iPhone 16, A18 Pro ANE)

| Mode | Tokens/sec | Speedup vs Baseline |
|------|-----------|-------------------|
| Baseline (no draft) | ~55 tok/s | 1.0× |
| DSpark k=3 | ~110 tok/s | 2.0× |
| DSpark k=5 | ~140 tok/s | 2.5× |
| DSpark k=7 | ~150 tok/s | 2.7× |
| DSpark + CoreML optimizations | ~165 tok/s | 3.0× |

### AI Edge Gallery Agent Skills

Each tool maps to an AI Edge Gallery Agent Skill (JavaScript, sandboxed):

| Tool | Skill File | Runtime |
|------|-----------|---------|
| Calculator | `skills/hermes_calculator/SKILL.md` | In-app JS sandbox |
| Web Search | `skills/hermes_web_search/SKILL.md` | URL session (offline cache) |
| Memory | `skills/hermes_memory/SKILL.md` | App storage (KV store) |
| Timer | `skills/hermes_timer/SKILL.md` | iOS timer API via bridge |

---

## New File Structure (Additions in bold)

```
hermes-edge/
├── hermes/
│   ├── __init__.py          [ADD] exports ReasoningConfig, AgentLoop, ToolRegistry
│   ├── config.py            [EDIT] add qwen3_0_6b_config()
│   ├── model.py             [EDIT] add DraftModelForCausalLM for training
│   ├── inference.py         [REWRITE] LiteRTInference with streaming & speculative
│   ├── kv_cache.py          [EXISTING]
│   ├── quantization.py      [EXISTING]
│   ├── chat_template.py     [EDIT] add parallel tool call format, DeepSeek reason tags
│   ├── reasoning.py         [NEW] DeepSeek V4 Flash reasoning pipeline
│   ├── agent.py             [NEW] Hermes agent loop with tool orchestration
│   ├── tool_registry.py     [NEW] Tool registration & dispatch
│   └── memory.py            [NEW] Persistent agent memory store
├── dspark/
│   ├── __init__.py          [NEW]
│   ├── draft_model_arch.py  [NEW] Draft transformer architecture
│   ├── draft_verify.py      [NEW] Draft-verify loop
│   └── acceptance.py        [NEW] Acceptance criteria (greedy, rejection)
├── agent/
│   ├── __init__.py          [NEW]
│   ├── tool_defs.py         [NEW] Tool definition schemas & validation
│   ├── dispatcher.py        [NEW] Async tool dispatcher with timeout/retry
│   ├── context.py           [NEW] Conversation context manager
│   └── memory_store.py      [NEW] On-device KV memory store backend
├── scripts/
│   ├── convert_to_litertlm.py [EDIT] add draft model bundling
│   ├── convert_qwen.py      [NEW] Qwen3-specific CPU-only conversion
│   ├── train_draft.py       [NEW] Train draft model via distillation
│   ├── train.py             [EXISTING]
│   ├── distill_from_gemma.py [EXISTING]
│   ├── benchmark.py         [EDIT] add speculative decode benchmark mode
│   ├── eval.py              [EXISTING]
│   └── train_tokenizer.py   [EXISTING]
├── deployment/
│   ├── gallery_manifest.json [NEW] AI Edge Gallery metadata
│   └── hermes_ios/          [NEW] Optional Swift Xcode project
├── data/
│   ├── eval.jsonl           [EXISTING]
│   └── tool_eval.jsonl      [EXISTING]
├── tests/
│   ├── test_model.py        [EDIT] add draft model tests
│   ├── test_inference.py    [EDIT] add reasoning & speculative tests
│   ├── test_kv_cache.py     [EXISTING]
│   ├── test_quantization.py [EXISTING]
│   ├── test_reasoning.py    [NEW] Reasoning pipeline tests
│   ├── test_agent.py        [NEW] Agent loop tests
│   └── test_dspark.py       [NEW] Speculative decoding tests
└── requirements.txt         [EDIT] add psutil, transformers (optional)
```

---

## Key Interfaces Summary

| Interface | File | Purpose |
|-----------|------|---------|
| `LiteRTInference.generate_stream()` | `hermes/inference.py` | Main streaming generation (new) |
| `ReasoningPipeline.process_stream()` | `hermes/reasoning.py` | DeepSeek think/tell separation |
| `AgentLoop.run()` | `hermes/agent.py` | Full agent orchestration loop |
| `ToolRegistry.dispatch()` | `hermes/tool_registry.py` | Tool lookup & execution |
| `MemorySystem.recall()` | `hermes/memory.py` | Semantic memory retrieval |
| `DraftVerifyEngine.generate()` | `dspark/draft_verify.py` | Speculative decoding loop |
| `greedy_accept()` | `dspark/acceptance.py` | Token acceptance criterion |

---

## Data Flow: Complete Request → Response

```
User: "What's 234*567? Also, set a timer for 5 minutes."
                                                                                
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. AgentLoop.run()                                                          │
  │    ├── Build system prompt with:                                            │
  │    │   - Tool schemas (calculator, timer)                                   │
  │    │   - Memory context (if any)                                            │
  │    │   - Reasoning instruction                                              │
  │    ├── Prefill prompt (main model)                                          │
  │    └── Enter generate loop                                                 │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ 2. LiteRTInference.generate_stream(speculative=True, reasoning=True)         │
  │    ├── Draft model predicts k=5 tokens: "Let", " me", " think", "...", ""    │
  │    ├── Main model verifies: accept "Let", " me", " think", "..."             │
  │    │                        reject "" → replace with "<"                     │
  │    ├── Continue: draft "think", ">", "234", " *", " 567" → verify → accept  │
  │    └── After ~20 tokens: reach "</think>"                                    │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ 3. ReasoningPipeline.process_stream()                                        │
  │    ├── Detect <think> tag → emit type="think" chunks                        │
  │    ├── Detect </think> tag → switch to type="answer" chunks                 │
  │    └── Yield: ("think", "Let me break this down..."), ("answer", "I'll...")  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ 4. Model emits:                                                              │
  │    <tool_calls>                                                              │
  │    <tool_call>{"name":"calculator","arguments":{"expression":"234*567"}}</>  │
  │    <tool_call>{"name":"timer","arguments":{"duration":300,"unit":"seconds"}}</>│
  │    </tool_calls>                                                             │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ 5. AgentLoop._parse_tool_calls()                                             │
  │    ├── Extract 2 tool calls from <tool_calls> block                          │
  │    ├── Parallel dispatch via ToolRegistry                                    │
  │    │   ├── calculator → 132,678                                             │
  │    │   └── timer → {"status": "created", "id": "t1"}                         │
  │    └── Append results as tool messages                                      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ 6. Second round: model generates final answer                                │
  │    "234 * 567 = 132,678. I've also set a 5-minute timer."                    │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Build Steps (Ordered)

### Phase 1: Environment

```bash
# 1. Install system deps
sudo apt-get install cmake python3-dev build-essential

# 2. Create venv
python3 -m venv venv && source venv/bin/activate

# 3. Install Hermes Edge + LiteRT stack
pip install -e .
pip install ai-edge-torch litert-lm sentencepiece torch numpy psutil

# 4. Install optional (for Qwen3 conversion)
pip install transformers accelerate safetensors
```

### Phase 2: Convert Qwen3-0.6B to .litertlm

```bash
# 5. Convert Qwen3-0.6B (CPU, <2.7GB RAM)
python scripts/convert_qwen.py \
    --hf-model Qwen/Qwen3-0.6B \
    --preset qwen3-0.6b \
    --output dist/hermes-mobile-qwen3-0.6b-int4.litertlm \
    --low-memory --max-prefill 1024 --gc-collect-between
```

### Phase 3: Train Draft Model

```bash
# 6. Train 30M draft model
python scripts/train_draft.py \
    --teacher dist/hermes-mobile-qwen3-0.6b-int4.litertlm \
    --student-config draft-30m \
    --data data/agentic_sft.jsonl \
    --output checkpoints/draft-30m.pt \
    --temperature 2.0 --lr 1e-3 --epochs 5

# 7. Convert draft to TFLite
python scripts/convert_to_litertlm.py \
    --checkpoint checkpoints/draft-30m.pt \
    --tokenizer tokenizer/hermes.model \
    --preset draft-30m \
    --output dist/draft-30m-int4.tflite \
    --backend apple --multi-sig
```

### Phase 4: Final Bundle

```bash
# 8. Bundle main + draft + tokenizer into single .litertlm
python scripts/convert_to_litertlm.py \
    --checkpoint dist/hermes-mobile-qwen3-0.6b-int4.litertlm \
    --draft-checkpoint dist/draft-30m-int4.tflite \
    --tokenizer tokenizer/hermes.model \
    --preset qwen3-0.6b \
    --output dist/hermes-mobile-v2.litertlm \
    --backend apple --multi-sig --bundle-draft
```

### Phase 5: Verify

```bash
# 9. Run tests
pytest tests/ -v

# 10. Benchmark (desktop - CPU)
python scripts/benchmark.py \
    --preset qwen3-0.6b \
    --seq-lens 64 128 256 512 \
    --speculative \
    --runs 3

# 11. Run agent eval
python scripts/eval.py \
    --model dist/hermes-mobile-v2.litertlm \
    --data data/tool_eval.jsonl \
    --reasoning \
    --speculative
```

### Phase 6: Deploy

```bash
# 12. Upload to HuggingFace
huggingface-cli upload bclermo/hermes-edge \
    dist/hermes-mobile-v2.litertlm \
    --repo-type model

# 13. Import URL in AI Edge Gallery:
# https://huggingface.co/bclermo/hermes-edge/resolve/main/hermes-mobile-v2.litertlm
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ai-edge-torch` | ≥0.3.0 | PyTorch → TFLite conversion |
| `litert-lm` | ≥0.1.0 | .litertlm bundler + runtime |
| `torch` | ≥2.4.0 | Reference model training |
| `sentencepiece` | ≥0.2.0 | Tokenizer |
| `numpy` | ≥1.26.0 | Array ops, sampling |
| `transformers` | (optional) | HF model loading for Qwen3 |
| `accelerate` | (optional) | CPU memory-efficient loading |
| `safetensors` | (optional) | Safe weight loading |
| `psutil` | ≥5.9.0 | Memory profiling |
| `tqdm` | ≥4.66.0 | Progress bars |

---

## Performance Model (Estimated)

### Without DSpark (Baseline)

| Stage | Time | Tok/s |
|-------|------|-------|
| Prefill (512 tok prompt) | ~2.5 s | 205 tok/s |
| Decode (100 tokens) | ~1.8 s | 55 tok/s |
| Total | ~4.3 s | — |

### With DSpark (k=5, 60% acceptance)

| Stage | Time | Tok/s |
|-------|------|-------|
| Prefill (512 tok prompt) | ~2.5 s | 205 tok/s |
| Draft decode (100 tokens ~ 20 drafts) | ~0.3 s | — |
| Main verify (20 verifications) | ~0.4 s | — |
| Total | ~3.2 s | — |
| **Effective decode** | — | **~140 tok/s** |
| **Speedup** | — | **2.5×** |
