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
    
# Fixed 5×5 map layout (S = start, F = frozen, H = hole, G = goal)
FIXED_MAP_5x5 = [
    "SFFFF",
    "FHFHF",
    "FFFHF",
    "HFHFF",
    "FFFHG",
]

# Create environment with custom rewards on a fixed map
env = make_frozenlake_with_custom_rewards(desc=FIXED_MAP_5x5)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

wandb_callback = WandbCallback()
model.learn(total_timesteps=300_000, callback=wandb_callback)
model.save("ppo_frozenlake")
wandb.save("ppo_frozenlake.zip")

TestModel()
