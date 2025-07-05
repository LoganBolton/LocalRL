import gymnasium as gym

class CustomRewardWrapper(gym.Wrapper):
    """
    Simple wrapper to customize FrozenLake rewards.
    """
    
    def __init__(self, env, reward_config=None):
        super().__init__(env)
        self.reward_config = reward_config or self._default_config()
        
    def _default_config(self):
        return {
            'goal': 2.0,        # Reward for reaching goal
            'hole': -1.0,       # Penalty for falling in hole
            'step': -0.01,      # Small penalty per step
            'ice': -0.001,      # Small penalty for ice
        }
    
    def step(self, action):
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
        
        # Hole penalty
        elif terminated and original_reward == 0:
            reward += self.reward_config['hole']
        
        # Step penalty (always applied)
        reward += self.reward_config['step']
        
        # Ice penalty
        row, col = current_pos
        if map_desc[row][col] == b'I':
            reward += self.reward_config['ice']
        
        return reward

def make_frozenlake_with_custom_rewards(reward_config=None):
    env = gym.make("FrozenLake-v1")
    return CustomRewardWrapper(env, reward_config)