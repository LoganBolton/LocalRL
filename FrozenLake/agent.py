from stable_baselines3 import PPO
import wandb
from callback import WandbCallback
from test_model import TestModel
from custom_rewards import make_frozenlake_with_custom_rewards

wandb.init(
    project="frozenlake-ppo",
    name="frozenlake-experiment",
    config={
        "algorithm": "PPO",
        "environment": "FrozenLake-v1",
        "policy": "MlpPolicy"
    }
)
    
# Create environment with custom rewards
env = make_frozenlake_with_custom_rewards()

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

wandb_callback = WandbCallback()
model.learn(total_timesteps=200_000, callback=wandb_callback)
model.save("ppo_frozenlake")
wandb.save("ppo_frozenlake.zip")

TestModel()
