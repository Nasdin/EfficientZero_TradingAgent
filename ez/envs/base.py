import gymnasium as gym
import numpy as np


class BaseWrapper(gym.Wrapper):
    def __init__(self, env, clip_reward):
        """Base environment wrapper for reward clipping and done flag handling
        Parameters
        ----------
        clip_reward: bool. Whether to clip rewards to [-1, 1]
        """
        super().__init__(env)
        self.clip_reward = clip_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        # Combine terminated and truncated into a single done flag for backward compatibility
        done = terminated or truncated
        info['terminated'] = terminated
        info['truncated'] = truncated

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def close(self):
        return self.env.close()
