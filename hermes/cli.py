"""
Hermes Edge CLI — interactive agent with intent routing, web search, and calculator.

ChatGPT-5.5-level smoothness features:
  --effort low|medium|high  adaptive reasoning effort (like GPT-5.5 reasoning_effort)
  --stream                  progressive token-by-token streaming
  Response cache             instant re-query for repeated questions
  Intent tag                 shows [chat]/[reasoning]/[tools] immediately (~5μs)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from hermes.rag import RAGEngine
from hermes.memory import AgentMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hermes")

INTENT_TAGS = {
    "chat": "\033[36m[chat]\033[0m ",
    "reasoning": "\033[33m[reasoning]\033[0m ",
    "tools": "\033[32m[tools]\033[0m ",
}
CLEAR_LINE = "\033[K"


def main():
    parser = argparse.ArgumentParser(description="Hermes Edge — On-Device AI Agent")
    parser.add_argument(
        "--model", "-m",
        default="dist/hermes-mobile-270m-int4.litertlm",
        help="Path to .litertlm model file",
    )
    parser.add_argument(
        "--reasoning-model",
        help="Path to reasoning model (DeepSeek-R1-Distill)",
    )
    parser.add_argument(
        "--tools-model",
        help="Path to tools model (Gemma-4-E2B)",
    )
    parser.add_argument(
        "--backend", "-b",
        default="auto",
        choices=["auto", "cpu", "gpu", "ane", "metal", "vulkan"],
        help="Compute backend",
    )
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="Disable intent routing",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as HTTP API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--rag",
        default="",
        help="Path to RAG database (enables knowledge retrieval)",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort (like GPT-5.5 reasoning_effort): low=fast, medium=balanced, high=deep",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens progressively (like ChatGPT)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response cache",
    )
    args = parser.parse_args()

    from hermes.litert_model import LiteRTModel
    from hermes.agent import (
        HermesAgent, AgentConfig, ModelManager,
        REASONING_EFFORT_LOW, REASONING_EFFORT_MEDIUM, REASONING_EFFORT_HIGH,
    )
    from hermes.router import INTENT_CHAT, INTENT_REASONING, INTENT_TOOLS

    config = AgentConfig(
        enable_routing=not args.no_routing,
        reasoning_effort=args.effort,
        enable_streaming=args.stream,
        enable_response_cache=not args.no_cache,
    )

    model_path = Path(args.model)
    if not model_path.exists():
        log.warning("Model not found: %s (using simulated responses)", model_path)
        model = LiteRTModel(str(model_path), backend=args.backend)
    else:
        model = LiteRTModel(str(model_path), backend=args.backend)
        model.load()

    model_manager = ModelManager(backend=args.backend)

    if args.reasoning_model:
        p = Path(args.reasoning_model)
        if p.exists():
            model_manager.register(INTENT_REASONING, str(p))
            log.info("Reasoning model registered: %s", p.name)

    if args.tools_model:
        p = Path(args.tools_model)
        if p.exists():
            model_manager.register(INTENT_TOOLS, str(p))
            log.info("Tools model registered: %s", p.name)

    if model._loaded and not model_manager._models:
        model_manager.load_hot(str(model_path))

    agent = HermesAgent(
        model=model if not model_manager._models else None,
        model_manager=model_manager if model_manager._models else None,
        config=config,
    )
    agent.register_default_tools()

    # Initialize RAG
    rag = None
    if args.rag:
        try:
            rag = RAGEngine(db_path=args.rag)
            log.info("RAG engine ready: %s", args.rag)
        except Exception as e:
            log.warning("RAG init failed: %s", e)

    # Initialize memory
    agent_memory = AgentMemory()

    if args.server:
        run_openai_server(args, agent, agent_memory, rag)
        return
    else:
        _run_interactive(agent)


def _run_interactive(agent):
    mode = "STREAM" if agent.config.enable_streaming else "BATCH"
    cache_status = "ON" if agent.config.enable_response_cache else "OFF"
    print("Hermes Edge — interactive mode (Ctrl+D to exit)")
    print(f"  Routing: {'ON' if agent.config.enable_routing else 'OFF'}")
    print(f"  Effort:  {agent.config.reasoning_effort}  |  Mode: {mode}  |  Cache: {cache_status}")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/q"):
            break

        try:
            if agent.config.enable_streaming:
                _run_streaming(agent, user_input)
            else:
                _run_batch(agent, user_input)
        except Exception as exc:
            log.error("Error: %s", exc)
            print(f"Error: {exc}")

        print()

    summary = agent.get_conversation_summary()
    log.info("Session summary: %s", summary)


def _run_batch(agent, user_input: str) -> None:
    """Non-streaming batch response with timing."""
    start = time.perf_counter()
    response = agent.run(user_input)
    elapsed = (time.perf_counter() - start) * 1000

    if response:
        print(response)
    if elapsed > 100:
        print(f"\033[90m({elapsed:.0f}ms)\033[0m")


def _run_streaming(agent, user_input: str) -> None:
    """Streaming response: shows intent tag immediately, then tiles tokens."""
    first = True
    for part in agent.run_stream(user_input):
        if first:
            intent_tag = INTENT_TAGS.get(part.strip("[] "), "")
            if part in ("[chat] ", "[reasoning] ", "[tools] "):
                print(f"{intent_tag}", end="", flush=True)
            else:
                print(part, end="", flush=True)
            first = False
        else:
            print(part, end="", flush=True)
    print()


def run_openai_server(args, agent, agent_memory, rag):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json, time

    class OpenAIHandler(BaseHTTPRequestHandler):
        server_agent = None  # set from outer scope
        server_memory = None
        server_rag = None

        def do_POST(self):
            path = self.path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if path == "/v1/chat/completions":
                messages = body.get("messages", [])
                stream = body.get("stream", False)
                model_name = args.model.split("/")[-1].replace(".litertlm", "")
                user_input = messages[-1]["content"] if messages else ""

                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    for chunk_text in self.server_agent.run_stream(user_input):
                        delta = chunk_text
                        chunk = {
                            "id": "chatcmpl-hermes",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()

                    final = {
                        "id": "chatcmpl-hermes",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    response_text = self.server_agent.run(user_input)
                    response = {
                        "id": "chatcmpl-hermes",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": response_text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

            elif path == "/v1/models":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "object": "list",
                    "data": [{"id": "hermes-edge", "object": "model", "created": int(time.time()), "owned_by": "simpliibarrii-crypto"}],
                }).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            log.info("HTTP: %s", format % args)

    OpenAIHandler.server_agent = agent
    OpenAIHandler.server_memory = agent_memory
    OpenAIHandler.server_rag = rag

    server = HTTPServer(("0.0.0.0", args.port), OpenAIHandler)
    log.info("OpenAI-compatible API server on http://0.0.0.0:%d/v1", args.port)
    log.info("  GET  /v1/models")
    log.info("  POST /v1/chat/completions  (stream=True supported)")
    log.info("  Effort: %s | Cache: %s", args.effort, "ON" if not args.no_cache else "OFF")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()


def demo_command(args):
    """Run the Hermes demo with random weights — no checkpoint needed."""
    from hermes.inference import DemoHermesInference
    from hermes.config import PRESETS

    preset = args.preset
    if preset not in PRESETS:
        print(f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}")
        return

    print(f"\n=== Hermes Edge Demo — {preset} ===")
    model = DemoHermesInference(preset)
    cfg = model.config
    print(f"Architecture: {cfg.num_layers} layers, {cfg.hidden_size}d hidden, {cfg.num_heads} heads, {cfg.num_kv_heads} KV heads")
    print(f"Context window: {cfg.max_seq_len} tokens | Parameters: ~{cfg.estimated_parameters() / 1e6:.0f}M")
    print(f"Quantization target: INT4 | Runtime: LiteRT-LM\n")

    sample_prompts = [
        "Hello, what can you do?",
        "What is the capital of Canada?",
        "Explain quantum computing in simple terms.",
    ]
    for prompt in sample_prompts:
        print(f"User: {prompt}")
        response = model.chat(prompt)
        print(f"Hermes: {response}\n")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Hermes Edge CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo subcommand
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run architecture demo with random weights (no checkpoint needed)",
    )
    demo_parser.add_argument(
        "--preset",
        default="hermes-270m",
        choices=["hermes-270m", "hermes-500m", "hermes-1b", "gemma-3-1b", "gemma-2-2b", "hermes-distilled-1b"],
        help="Model preset to demo",
    )
    return parser


def main_with_demo():
    """Extended main that supports the 'demo' subcommand."""
    parser = _build_parser()
    # Check if first arg is 'demo'
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "demo":
        args = parser.parse_args()
        demo_command(args)
    else:
        main()
