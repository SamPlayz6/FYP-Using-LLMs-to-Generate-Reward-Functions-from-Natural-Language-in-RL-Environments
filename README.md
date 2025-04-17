# Adaptive Reward Function Learning using LLMs

## What Is This Project?

This project demonstrates an innovative approach to reinforcement learning by using Large Language Models (LLMs) to dynamically generate and adapt reward functions. Key aspects include:

- **Dynamic Reward Generation**: The system uses Anthropic's Claude to create reward functions that evolve during training
- **Environmental Adaptation**: Automatically detects and adapts to changes in the environment parameters
- **Component-Based Rewards**: Uses separate components for stability and efficiency with dynamic weighting
- **Enhanced Robustness**: Provides significantly better performance than static reward functions when environments change
- **Explainable AI**: Generates natural language explanations for reward adjustments

## Project Structure

The repository is organized into two main modules:

### RLEnvironment
- **env/**: Contains environment wrappers for CartPole and BipedalWalker
  - `cartPoleSetup.py`: CartPole environment wrapper with component rewards
  - `bipedalWalkerSetup.py`: BipedalWalker environment wrapper
  - `wrappers.py`: General reward function wrapper classes
- **training/**: Contains reinforcement learning training functionality
  - `agent.py`: Implementation of the DQN learning agent
  - `training.py`: Main training loop and utilities

### AdaptiveRewardFunctionLearning
- **Prompts/**: Contains LLM prompting infrastructure
  - `prompts.py`: API key configuration and model settings
  - `criticPrompts.py`: Templates for reward function evaluation
- **RewardGeneration/**: Core reward function management
  - `rewardCritic.py`: System for evaluating and updating reward functions
  - `rewardCodeGeneration.py`: Initial reward function components
  - `cartpole_energy_reward.py`: Energy-based reward function implementation
- **Experiments/**: Analysis notebooks for running experiments
  - `1.1_explainability.ipynb`: Analysis of LLM's ability to explain reward functions
  - `2.1_Robustness_Cart_Pole.ipynb`: Tests adaptation to CartPole environment changes
  - `2.2_Robustness_Bi-Pedal_Walker.ipynb`: Tests adaptation in complex BipedalWalker environment
  - `3.1_performance.ipynb`: Performance comparison of adaptive vs static rewards in CartPole
  - `3.2_performance_Bi-Pedal_Walker.ipynb`: Performance analysis in BipedalWalker environment

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- Access to Anthropic's Claude API (API key required)
- CUDA-compatible GPU (required for all experiments)
- Sufficient computational resources (see Hardware Requirements)

### Installation Steps
1. **Clone the repository**:
   - Use Git to clone the repository to your local machine

2. **Install dependencies**:
   - Run `pip install -r requirements.txt`
   - The requirements file includes:
     - PyTorch (1.10+) with CUDA support
     - Gymnasium (0.26+)
     - Numpy, Pandas, Matplotlib
     - Anthropic Python SDK
     - Jupyter Notebook/Lab
     - Seaborn and other visualization tools

3. **Configure API access**:
   - Open `AdaptiveRewardFunctionLearning/Prompts/prompts.py`
   - Replace the placeholder with your Anthropic API key:
     ```
     apiKey = "your_anthropic_api_key_here"
     ```
   - The default model is set to Claude 3.5 Sonnet, but can be changed to other Claude models

## Hardware Requirements

All experiments require a GPU as they use Deep Q-learning algorithms. Specific requirements:

- **GPU Requirement (MANDATORY)**:
  - CUDA-compatible GPU with at least 4GB VRAM
  - Latest NVIDIA drivers and CUDA toolkit
  - PyTorch with CUDA support properly installed and verified

- **CartPole experiments**:
  - Minimum: 4 CPU cores, 8GB RAM, 4GB VRAM GPU
  - Recommended: 6+ CPU cores, 16GB RAM, 6GB+ VRAM GPU

- **BipedalWalker experiments**:
  - Minimum: 6 CPU cores, 16GB RAM, 6GB VRAM GPU
  - Recommended: 8+ CPU cores, 32GB RAM, 8GB+ VRAM GPU
  - Expected runtime: 8-24 hours depending on hardware

- **API Usage**: The experimental notebooks make approximately 10-100 API calls to Claude (depending on configuration), so ensure your Anthropic API quota is sufficient

## Running Experiments

### Setting Up Jupyter Notebook

1. **Start Jupyter**:
   ```
   jupyter notebook
   ```
   or
   ```
   jupyter lab
   ```

2. **Navigate to the experiment directory**:
   - Browse to `AdaptiveRewardFunctionLearning/Experiments/`
   - Open the desired notebook

3. **Configure notebook environment**:
   - Each notebook has a configuration cell at the top
   - Adjust parameters like episode count or update frequency if needed
   - Lower episode counts will run faster but may not show full adaptation effects
   - Verify GPU is properly detected with `torch.cuda.is_available()`

### Recommended Experiment Sequence

For best understanding of the project, run notebooks in this order:

1. `1.1_explainability.ipynb` - Understand how LLMs explain reward functions
2. `3.1_performance.ipynb` - See basic performance in CartPole
3. `3.2_performance_Bi-Pedal_Walker.ipynb` - See performance in Bi-Pedal Walker
4. `2.1_Robustness_Cart_Pole.ipynb` - Observe adaptation to environment changes
5. `2.2_Robustness_Bi-Pedal_Walker.ipynb` - See adaptation in complex environments

### Experiment Parameters

Each notebook contains configurable parameters:
- **Episodes**: Number of training iterations (higher = better results but longer runtime)
- **Change Interval**: When environment parameters change
- **Seeds**: Random seeds for reproducibility
- **Update Settings**: How frequently to request LLM updates
- **Batch Size**: May need adjustment based on available GPU memory

## Interpreting Results

The experiments produce various visualizations:

- **Reward Plots**: Show performance over time
- **Component Weight Evolution**: Track how stability vs efficiency balance changes
- **Adaptation Speed**: Measure recovery time after environment changes
- **Comparison Metrics**: Quantify advantages over static reward approaches

## Extending the Project

To customize for your own environments:
1. Create a new environment wrapper in `RLEnvironment/env/`
2. Define appropriate reward components
3. Configure the adaptation system in a new notebook
4. Ensure GPU compatibility with your environment

## Troubleshooting

Common issues:
- **CUDA Errors**: Verify that PyTorch can access your GPU with `torch.cuda.is_available()`
- **Out of Memory**: Reduce batch size or model complexity
- **API Key Errors**: Ensure the Anthropic API key is correctly configured
- **Memory Errors**: Reduce episode counts or batch sizes
- **Import Errors**: Verify all dependencies are installed with correct versions
- **Runtime Errors**: Check logs for specific component failures
