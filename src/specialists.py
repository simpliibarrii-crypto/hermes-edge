import yaml
from typing import Dict, Any

class SpecialistRegistry:
    def __init__(self, config_path: str = "config/routing.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['specialists']

    def get_specialist_config(self, intent: str) -> Dict[str, Any]:
        return self.config.get(intent, self.config["fallback"])
