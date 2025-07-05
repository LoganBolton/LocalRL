import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os
import wandb
from callback import WandbCallback
from test_model import TestModel

wandb.init(
    project="frozenlake-ppo",
    name="frozenlake-experiment",
    config={
        "algorithm": "PPO",
        "environment": "FrozenLake-v1",
        "policy": "MlpPolicy"
    }
)
    
env = gym.make("FrozenLake-v1")

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

wandb_callback = WandbCallback()
model.learn(total_timesteps=10_000, callback=wandb_callback)
model.save("ppo_frozenlake")
wandb.save("ppo_frozenlake.zip")

TestModel()
