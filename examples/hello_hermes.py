"""Hello Hermes Edge — quick start example."""

from hermes.litert_model import LiteRTModel
from hermes.agent import HermesAgent, AgentConfig

MODEL_PATH = "dist/hermes-mobile-270m-int4.litertlm"


def main():
    model = LiteRTModel(MODEL_PATH)
    if not model.load():
        print(f"Model not found at {MODEL_PATH}")
        print("Download from: https://huggingface.co/bclermo/hermes-edge")
        return

    agent = HermesAgent(
        model=model,
        config=AgentConfig(use_reasoning=True, enable_routing=True),
    )
    agent.register_default_tools()

    print("Hermes Edge ready. Ask me anything!")
    response = agent.run("What is 15% of 80?")
    print(response)


if __name__ == "__main__":
    main()
