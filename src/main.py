from model_wrapper import LiteRTModelWrapper
from router import IntentRouter
from specialists import SpecialistRegistry

def run_agent():
    llm = LiteRTModelWrapper(model_path="models/hermes_edge.litert")
    router = IntentRouter(llm)
    registry = SpecialistRegistry()

    print("🚀 Hermes Edge: Intent-Routed System Online")
    print("Type 'exit' to shutdown.\n")

    while True:
        user_input = input("User > ")
        if user_input.lower() in ['exit', 'quit']: break

        decision = router.route(user_input)
        intent = decision.intent
        conf = decision.confidence
        
        spec = registry.get_specialist_config(intent)
        system_prompt = spec['prompt']
        tier = spec['model_tier']

        print(f"🛠️  [Router] Intent: {intent} | Confidence: {conf:.2f} | Tier: {tier}")

        final_prompt = f"System: {system_prompt}\nUser: {user_input}\nAssistant:"
        response = llm.generate(final_prompt, tier=tier)
        
        print(f"Hermes > {response}\n")

if __name__ == "__main__":
    run_agent()
