import gymnasium as gym
import numpy as np
from collections import deque

class CustomEnvWrapper(gym.Wrapper):
    """Base wrapper class for reinforcement learning environments with adaptive rewards"""
    def __init__(self, env, numComponents=2):
        super().__init__(env)
        self.env = env
        self.rewardFunction = self.defaultReward
        
        # Component handling
        self.usingComponents = True
        self.rewardComponents = {}
        self.componentWeights = {}
        
        # Initialize components with default weights
        for i in range(1, numComponents+1):
            self.rewardComponents[f'rewardFunction{i}'] = None
            self.componentWeights[f'rewardFunction{i}'] = 1.0 / numComponents
        
        # Track scales for reward normalization
        self.reward_scales = {}
        self.reward_history = {name: [] for name in self.rewardComponents.keys()}
        
        # For weight history tracking
        self.weight_history = {name: deque(maxlen=10) for name in self.rewardComponents.keys()}
        for name, weight in self.componentWeights.items():
            self.weight_history[name].append(weight)
    
    def setComponentReward(self, componentNumber: int, rewardFunction):
        """Set a specific reward component function"""
        if componentNumber < 1 or componentNumber > len(self.rewardComponents):
            raise ValueError(f"Component number must be between 1 and {len(self.rewardComponents)}")
            
        funcName = f'rewardFunction{componentNumber}'
        if funcName in self.rewardComponents:
            self.rewardComponents[funcName] = rewardFunction
            if componentNumber == 1:  # Primary component
                self.rewardFunction = rewardFunction
            return True
        return False
    
    def updateComponentWeight(self, componentNumber: int, weight: float, smooth_factor=0.05):
        """Update component weights with smoother transitions"""
        if componentNumber < 1 or componentNumber > len(self.rewardComponents):
            raise ValueError(f"Component number must be between 1 and {len(self.rewardComponents)}")
            
        funcName = f'rewardFunction{componentNumber}'
        
        # Check for oscillation patterns
        if len(self.weight_history[funcName]) >= 3:
            recent_weights = list(self.weight_history[funcName])
            if (recent_weights[-1] > recent_weights[-2] and weight < recent_weights[-1]) or \
               (recent_weights[-1] < recent_weights[-2] and weight > recent_weights[-1]):
                smooth_factor *= 0.5
                print(f"Detected oscillation pattern. Reducing weight change rate.")
        
        # Set bounds based on component
        if componentNumber == 1:  # Stability
            weight = max(0.3, min(0.8, weight))
        else:  # Efficiency or other
            weight = max(0.2, min(0.7, weight))
        
        # Smooth transition
        old_weight = self.componentWeights[funcName]
        new_weight = (1 - smooth_factor) * old_weight + smooth_factor * weight
        
        # Track weight history
        self.weight_history[funcName].append(new_weight)
        
        # Apply new weight
        self.componentWeights[funcName] = new_weight
        
        # Update other components to maintain sum of 1.0
        remaining_weight = 1.0 - new_weight
        remaining_components = len(self.componentWeights) - 1
        
        if remaining_components > 0:
            # Distribute remaining weight among other components
            for name in self.componentWeights:
                if name != funcName:
                    self.componentWeights[name] = remaining_weight / remaining_components
        
        # Print weight summary
        weights_str = ", ".join([f"{name.replace('rewardFunction', 'Component ')}: {w:.3f}" 
                               for name, w in self.componentWeights.items()])
        print(f"Updated weights - {weights_str}")
        
        return True
    
    def defaultReward(self, observation, action):
        """Default environment reward function"""
        return 1.0
    
    def setRewardFunction(self, rewardFunction):
        """Set a single reward function (used mainly for baseline comparison)"""
        self.rewardFunction = rewardFunction
        
    def getCurrentWeights(self):
        """Get current component weights for monitoring"""
        return {
            'stability': self.componentWeights['rewardFunction1'],
            'efficiency': self.componentWeights['rewardFunction2']
        }
    
    def computeReward(self, observation, action, terminated, truncated, info):
        """Compute reward based on components or single reward function"""
        if self.usingComponents and any(self.rewardComponents.values()):
            info['componentRewards'] = {}
            rewards = []
            raw_rewards = {}
            
            for name, func in self.rewardComponents.items():
                if func and callable(func):
                    try:
                        componentReward = func(observation, action)
                        raw_rewards[name] = componentReward
                        
                        # Add to history for smoothing
                        self.reward_history[name].append(componentReward)
                        if len(self.reward_history[name]) > 50:
                            self.reward_history[name].pop(0)
                        
                        # Update running scale estimate
                        if name not in self.reward_scales:
                            self.reward_scales[name] = abs(componentReward) if componentReward != 0 else 1.0
                        else:
                            recent_rewards = self.reward_history[name][-10:] if len(self.reward_history[name]) >= 10 else self.reward_history[name]
                            if recent_rewards:
                                median_reward = sorted(map(abs, recent_rewards))[len(recent_rewards)//2]
                                self.reward_scales[name] = 0.98 * self.reward_scales[name] + 0.02 * median_reward
                        
                        # Normalize reward
                        scale = max(self.reward_scales[name], 1e-4)
                        normalized_reward = componentReward / scale
                        
                        # Apply sigmoid normalization to contain extreme values
                        normalized_reward = 2.0 / (1.0 + np.exp(-normalized_reward)) - 1.0
                        
                        # Apply weight
                        weightedReward = normalized_reward * self.componentWeights[name]
                        
                        rewards.append(weightedReward)
                        info['componentRewards'][name] = componentReward
                        
                    except Exception as e:
                        print(f"Error in reward component {name}: {e}")
                        rewards.append(0)
            
            # Add raw reward values to info for analysis
            info['raw_rewards'] = raw_rewards
            info['scales'] = {name: scale for name, scale in self.reward_scales.items()}
            
            return sum(rewards) if rewards else self.defaultReward(observation, action)
        else:
            # Use single reward function
            if self.rewardFunction is None or not callable(self.rewardFunction):
                print("Warning: rewardFunction is None or not callable.")
                self.rewardFunction = self.defaultReward
                
            return self.rewardFunction(observation, action)