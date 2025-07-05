import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os
import wandb
from callback import WandbCallback


class TestModel:
    def __init__(self):
        self.test_model()
        
    def test_model(self):
        
        model = PPO.load("ppo_frozenlake")

        test_env = gym.make("FrozenLake-v1", render_mode="rgb_array")
        test_env = RecordVideo(test_env, video_folder="./videos/", episode_trigger=lambda x: True, name_prefix="frozenlake_training_test")

        obs, _ = test_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # Convert numpy array to integer
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            step_count += 1

        print(f"Total reward: {total_reward}")
        print(f"Total steps: {step_count}")

        # Log test results to wandb
        wandb.log({
            "test_total_reward": total_reward,
            "test_total_steps": step_count,
            "test_success": total_reward > 0
        })

        test_env.close()

        # Finish wandb run
        wandb.finish()