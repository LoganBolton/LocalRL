import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    """
    Simple wrapper to customize FrozenLake rewards.
    """
    
    def __init__(self, env, reward_config=None):
        super().__init__(env)
        self.reward_config = reward_config or self._default_config()
        # Track the number of steps taken in the current episode so that
        # reward shaping can depend on how quickly the agent reaches the goal.
        self.step_count: int = 0


    def reset(self, **kwargs):
        """Reset the underlying env **and** the episode step counter."""
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def _default_config(self):
        return {
            'goal': 20.0,        # Reward for reaching goal
            'hole': -1.0,       # Penalty for falling in hole
            'step': -0.01,      # Small penalty per step
            # 'ice': -0.001,      # Small penalty for ice
        }
    
    def step(self, action):
        # Increment episode step counter *before* delegating to env.step so it
        # reflects the step we are about to take.
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get environment state
        env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        current_pos = getattr(env, 's', 0) // getattr(env, 'ncol', 4), getattr(env, 's', 0) % getattr(env, 'ncol', 4)
        map_desc = getattr(env, 'desc', [])
        
        # Calculate custom reward
        custom_reward = self._get_custom_reward(reward, terminated, current_pos, map_desc)
        
        return obs, custom_reward, terminated, truncated, info
    
    def _get_custom_reward(self, original_reward, terminated, current_pos, map_desc):
        """Calculate custom reward based on current state"""
        reward = 0.0
        
        # Goal reward
        if terminated and original_reward > 0:
            reward += self.reward_config['goal']
            if self.step_count < 20:
                reward += 0.5 *self.reward_config['goal']
        
        # Hole penalty
        elif terminated and original_reward == 0:
            reward += self.reward_config['hole']
        
        # Step penalty (always applied)
        reward += self.reward_config['step']
        
        # Ice penalty
        # row, col = current_pos
        # if map_desc[row][col] == b'I':
        #     reward += self.reward_config['ice']
        
        return reward

class GlobalStateWrapper(gym.Wrapper):
    """
    Wrapper that provides global board state information to the agent.
    Modifies observation to include the full map layout.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Get map size from the underlying (unwrapped) env
        self._map_size: int = len(self.env.unwrapped.desc)

        # Observation = [row_norm, col_norm] + flattened_map
        new_obs_size = 2 + self._map_size * self._map_size

        # Build low/high arrays that match the expected value ranges
        low = np.zeros(new_obs_size, dtype=np.float32)
        high = np.ones(new_obs_size, dtype=np.float32)  # everything is normalised to [0, 1]

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_global_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_global_observation(obs), reward, terminated, truncated, info

    def _get_global_observation(self, original_obs):
        """Convert original observation to include global board state"""
        env = self.env.unwrapped
        map_desc = env.desc

        # Convert map to numerical representation
        map_encoding = {
            b'S': 0,  # Start
            b'F': 1,  # Frozen
            b'H': 2,  # Hole
            b'G': 3   # Goal
        }

        # Flatten and normalise the map encoding to [0, 1]
        flattened_map = np.array([
            map_encoding[cell] for row in map_desc for cell in row
        ], dtype=np.float32) / 3.0

        # Decode original observation (0..nS-1) -> (row, col)
        row = original_obs // self._map_size
        col = original_obs % self._map_size
        row_norm = row / (self._map_size - 1)
        col_norm = col / (self._map_size - 1)

        # Build the final observation vector
        global_obs = np.concatenate((
            np.array([row_norm, col_norm], dtype=np.float32),
            flattened_map
        ))

        return global_obs.astype(np.float32)

def make_frozenlake_with_custom_rewards(
    reward_config: dict | None = None,
    render_mode: str | None = None,
    *,
    desc: list[str] | None = None,
    map_size: int = 5,
    is_slippery: bool = True,
    seed: int | None = None,
):

    # Determine map description
    if desc is None:
        desc = generate_random_map(size=map_size, seed=seed)

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )

    # Apply wrappers in order: GlobalStateWrapper first, then CustomRewardWrapper
    env = GlobalStateWrapper(env)
    env = CustomRewardWrapper(env, reward_config)

    return env