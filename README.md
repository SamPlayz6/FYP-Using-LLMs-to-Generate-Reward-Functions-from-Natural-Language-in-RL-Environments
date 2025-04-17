# Adaptive Reward Function Learning using LLMs  
**Dynamically generates reward functions for RL agents** using LLMs through the Anthropic API. Automatically adapts to environment changes for more robust performance than static reward systems.  

## Quick Setup  
1. Install requirements:  
```bash  
pip install -r requirements.txt  
```

2. Add API key to `AdaptiveRewardFunctionLearning/Prompts/prompts.py`:
```python
apiKey = "your_anthropic_api_key_here"  # Get from Anthropic  
```

## Run Basic Training (CartPole)
```python
from RLEnvironment.env import CustomCartPoleEnv  
from RLEnvironment.training.agent import DQLearningAgent  
from RLEnvironment.training.training import trainDQLearning  
import gymnasium as gym  

env = CustomCartPoleEnv(gym.make('CartPole-v1'), numComponents=2)  
agent = DQLearningAgent(env=env, stateSize=4, actionSize=2)  
trainDQLearning(agent, env, numEpisodes=1000)  
```

## Run Analysis Notebooks
```bash
#Navigate to the AdaptiveRewardFunctionLearning folder

# Explainability analysis  
jupyter notebook Experiments/1.1_explainability.ipynb  

# CartPole robustness  
jupyter notebook Experiments/2.1_Robustness_Cart_Pole.ipynb

# Bi-Pedal Walker Robustness
jupyter notebook Experiments/2.2_Robustness_Bi-Pedal_Walker.ipynb

# CartPole Performance
jupyter notebook Experiments/3.1_performance.ipynb


# Bi-Pedal Walker performance  
jupyter notebook Experiments/3.2_performance_Bi-Pedal_Walker.ipynb  
```

## Key Features
* Automatic reward adaptation during environment changes
* LLM-generated reward components with dynamic weights
* Built-in visualization tools for training analysis
* Pre-configured experiments for reproducibility

**Note**: All paths are relative to project root. Ensure you have Jupyter installed for notebook analysis.