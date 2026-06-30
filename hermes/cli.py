"""
Hermes Edge CLI — interactive agent with intent routing, web search, and calculator.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hermes")


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
    args = parser.parse_args()

    from hermes.litert_model import LiteRTModel
    from hermes.agent import HermesAgent, AgentConfig, ModelManager
    from hermes.router import INTENT_CHAT, INTENT_REASONING, INTENT_TOOLS

    config = AgentConfig(enable_routing=not args.no_routing)

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

    if args.server:
        _run_server(agent, args.port)
    else:
        _run_interactive(agent)


def _run_interactive(agent):
    print("Hermes Edge — interactive mode (Ctrl+D to exit)")
    print(f"  Routing: {'ON' if agent.config.enable_routing else 'OFF'}")
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
            response = agent.run(user_input)
            print(response)
        except Exception as exc:
            log.error("Error: %s", exc)
            print(f"Error: {exc}")

        print()

    summary = agent.get_conversation_summary()
    log.info("Session summary: %s", summary)


def _run_server(agent, port: int):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        log.error("Server mode requires fastapi and uvicorn: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="Hermes Edge", version="0.2.0")

    class ChatRequest(BaseModel):
        message: str
        context: str | None = None

    class ChatResponse(BaseModel):
        response: str
        intent: str = ""
        model: str = ""
        latency_ms: float = 0.0

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        start = __import__("time").time()
        resp = agent.run(req.message, context=req.context)
        elapsed = (__import__("time").time() - start) * 1000
        turn = agent.conversation.turns[-1] if agent.conversation.turns else None
        return ChatResponse(
            response=resp,
            intent=turn.intent if turn else "",
            model=turn.model_used if turn else "",
            latency_ms=elapsed,
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "routing": agent.config.enable_routing}

    log.info("Starting server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
