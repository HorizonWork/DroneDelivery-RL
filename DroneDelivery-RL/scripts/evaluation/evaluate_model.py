#!/usr/bin/env python3
"""
Evaluate trained model performance
""",
from src.rl.evaluation.evaluator import Evaluator
from src.environment.airsim_env import AirSimEnv

def main():
    # Load trained model and environment
    env = AirSimEnv()
    # Load agent from checkpoint
    # agent = load_agent(model_path)
    
    evaluator = Evaluator(agent, env)
    results = evaluator.evaluate()
    
    print(f"Performance results: {results}")
    
if __name__ == "__main__":
    main()
