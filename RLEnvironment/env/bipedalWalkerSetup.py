import gymnasium as gym
import numpy as np
from .baseEnvironmentWrapper import CustomEnvWrapper

class DiscretizedBipedalWalkerEnv(gym.ActionWrapper):
    """Wrapper that discretizes BipedalWalker's continuous action space"""
    
    def __init__(self, env, bins=3):
        super().__init__(env)
        # For each of the 4 joints, discretize into bins (e.g., -1, 0, 1)
        self.bins = bins
        self.values = np.linspace(-1, 1, bins)
        
        # Original action space is Box(-1, 1, (4,))
        # New action space is Discrete(bins**4)
        self.action_space = gym.spaces.Discrete(bins**4)
        
    def action(self, action):
        """Convert discrete action to continuous action"""
        # Convert integer to base-n representation
        indices = []
        temp = action
        for i in range(4):
            indices.insert(0, temp % self.bins)
            temp = temp // self.bins
            
        # Map indices to continuous values
        continuous_action = np.array([self.values[idx] for idx in indices])
        return continuous_action

class CustomBipedalWalkerEnv(CustomEnvWrapper):
    """BipedalWalker-specific environment wrapper with discretized actions"""
    def __init__(self, env, numComponents=2, discretize_bins=3):
        # First discretize the action space
        discretized_env = DiscretizedBipedalWalkerEnv(env, bins=discretize_bins)
        super().__init__(discretized_env, numComponents)
        self.env_type = "bipedal"
    
    def step(self, action):
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute reward using components or single function
        reward = self.computeReward(observation, action, terminated, truncated, info)
        
        # Track termination reason
        if terminated:
            hull_angle = observation[0] if len(observation) > 0 else 0
            if abs(hull_angle) > 0.7:
                info['termination_reason'] = 'angle_limit'
            elif hasattr(self.env.unwrapped, 'game_over') and self.env.unwrapped.game_over:
                info['termination_reason'] = 'game_over'
            else:
                info['termination_reason'] = 'other'
        
        return observation, reward, terminated, truncated, info
    
    def setEnvironmentParameters(self, leg_length=40, terrain_roughness=1.0, gravity=9.8):
        """Update BipedalWalker physical parameters"""
        modified = False
        
        # For leg length we must recreate the environment
        if hasattr(self.env.unwrapped, 'LEG_H') and abs(self.env.unwrapped.LEG_H - leg_length/2) > 0.001:
            self.env.unwrapped.LEG_H = leg_length / 2  # Total leg length is 2 * LEG_H
            modified = True
            
        # For terrain and gravity, we can modify directly
        if hasattr(self.env.unwrapped, 'TERRAIN_ROUGHNESS'):
            self.env.unwrapped.TERRAIN_ROUGHNESS = terrain_roughness
            modified = True
            
        # Modify gravity (this is approximate as it affects all forces)
        if hasattr(self.env.unwrapped, 'GRAVITY'):
            self.env.unwrapped.GRAVITY = gravity
            modified = True
        
        if modified:
            print(f"BipedalWalker parameters updated: leg_length={leg_length}, "
                  f"terrain_roughness={terrain_roughness}, gravity={gravity}")
        return modified