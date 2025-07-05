from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log step-level metrics
        if len(self.locals.get('rewards', [])) > 0:
            current_reward = self.locals['rewards'][0]
            wandb.log({
                "step_reward": current_reward,
                "step": self.step_count
            })
        
        # Log episode-level metrics when episode ends
        if self.locals.get('dones', [False])[0]:
            episode_reward = sum(self.locals.get('rewards', [0]))
            episode_length = len(self.locals.get('rewards', []))
            episode_success = episode_reward > 0  # Success if reward > 0
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_successes.append(episode_success)
            
            # Calculate running averages
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
            success_rate = np.mean(self.episode_successes[-100:]) if len(self.episode_successes) >= 100 else np.mean(self.episode_successes)
            
            try:
                wandb.log({
                    "episode": len(self.episode_rewards),
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_success": episode_success,
                    "avg_reward_100": avg_reward,
                    "avg_length_100": avg_length,
                    "success_rate_100": success_rate,
                })
            except:
                pass  # Continue if wandb logging fails
            
        return True