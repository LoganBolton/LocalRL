import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import wandb
from custom_rewards import CustomRewardWrapper
from custom_rewards import make_frozenlake_with_custom_rewards

# Same fixed map used during training
FIXED_MAP_5x5 = [
    "SFFFF",
    "FHFHF",
    "FFFHF",
    "HFHFF",
    "FFFHG",
]

class TestModel:
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb
        self.test_model()
        
    def test_model(self):
        # Initialize wandb only if requested
        if self.use_wandb:
            wandb.init(project="frozenlake-test", name="model-testing")
        
        model = PPO.load("ppo_frozenlake")
        
        # Create base environment with render mode on the fixed map
        test_env = make_frozenlake_with_custom_rewards(render_mode="rgb_array", desc=FIXED_MAP_5x5)
        test_env = RecordVideo(test_env, video_folder="./videos/", episode_trigger=lambda x: True, name_prefix="frozenlake_training_test")

        obs, _ = test_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            step_count += 1

        print(f"Total reward: {total_reward}")
        print(f"Total steps: {step_count}")

        # Log test results to wandb only if requested
        if self.use_wandb:
            wandb.log({
                "test_total_reward": total_reward,
                "test_total_steps": step_count,
                "test_success": total_reward > 0
            })

        test_env.close()

        # Finish wandb run only if it was initialized
        if self.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    # Run without wandb when executed directly
    TestModel(use_wandb=False)