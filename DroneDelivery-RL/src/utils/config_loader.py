"""
Configuration file loading
""",
import yaml

class ConfigLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_config(self):
        return self.config
