#!/usr/bin/env python3
"""
Complete 5M timestep curriculum training
""",
from src.rl.training.trainer import Trainer
from src.rl.agents.ppo_agent import PPOAgent
from src.environment.airsim_env import AirSimEnv

def main():
    # Initialize environment and agent
    env = AirSimEnv()
    agent = PPOAgent(env.observation_space, env.action_space)
    trainer = Trainer(agent, env)
    
    # Train with full curriculum (5M timesteps)
    trainer.train(total_timesteps=50000)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
