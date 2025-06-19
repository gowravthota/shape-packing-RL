# space_RL

A 2D space optimization and packing environment using Reinforcement Learning (RL).

## Overview
This project simulates a container-packing problem, where the goal is to optimally place shapes inside a container using RL agents. The environment is built with OpenAI Gym and PyTorch, and supports multiple RL algorithms.

## Features
- Custom 2D container-packing environment (`ContainerEnv`)
- Discrete action space: select grid cells for shape placement
- Reward structure: +1 for valid placement, -1 for invalid
- Modular RL agent support (DQN, PPO, A2C)
- Visualization utilities (see `visualization.py`)

## Directory Structure
- `main.py` — Main training loop and environment definition
- `constraints.py` — Shape and container logic
- `models/` — Reference RL agent implementations (DQN, PPO, A2C)
- `visualization.py` — Visualization utilities (optional)
- `requirements.txt` — Python dependencies

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd space_RL
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Additional dependencies (if not in requirements.txt):
   - torch
   - gym
   - numpy
   - pygame (for visualization)

## How to Run
By default, `main.py` runs a DQN agent on the container-packing environment:
```bash
python main.py
```

- The script will train the agent and periodically print progress.
- After training, the model is saved as `dqn_container.pt`.
- You can modify `main.py` to use other agents from `models/` (see below).

## Using Other RL Models
Reference RL agents are provided in the `models/` directory:
- `dqn.py` — Deep Q-Network (DQN)
- `ppo.py` — Proximal Policy Optimization (PPO)
- `a2c.py` — Advantage Actor-Critic (A2C)

To use a different agent, import and instantiate it in `main.py`:
```python
from models.dqn import create_dqn_agent, dqn_agent_step
from models.ppo import create_ppo_agent, ppo_update
from models.a2c import create_a2c_agent, a2c_update
```

## How It Works
- The environment is a 10x10 grid container.
- Each action selects a cell to place a shape (square by default).
- The agent receives +1 for valid placements, -1 for invalid.
- The episode ends when the grid is full or after a set number of steps.
- The RL agent learns to maximize the number of valid placements.

## Optimizations & Tips
- **Model Architecture:** Try deeper or wider networks for more complex packing strategies.
- **Reward Shaping:** Experiment with different reward structures (e.g., bonus for filling rows/columns).
- **Action Space:** Extend to continuous placement or allow for different shape types.
- **Curriculum Learning:** Start with smaller grids and increase difficulty.
- **Visualization:** Use `visualization.py` to debug and visualize agent behavior.
- **Batch Size & Learning Rate:** Tune these hyperparameters for better convergence.

## Implemented Optimizations
- **Action Masking:** The agent is prevented from selecting already-filled grid cells, improving learning efficiency and stability.
- **Reward Normalization:** Rewards are normalized using a running mean and standard deviation, helping stabilize training and improve convergence.

## Creative Optimization Ideas (Future Work)
- **Curriculum Learning:** Start with smaller/easier grids and gradually increase difficulty as the agent improves.
- **Shape Prioritization:** Place larger or more awkward shapes first to maximize space utilization.
- **Learning Rate Scheduling:** Dynamically adjust the learning rate during training for better convergence.
- **Dynamic Grid Resolution:** Begin with a coarse grid and refine as the agent learns, or use multi-scale approaches.
- **Hybrid RL + Heuristics:** Combine RL with classic packing heuristics (e.g., bottom-left, best-fit) for improved performance.
- **Edge/Corner Biasing:** Encourage the agent to fill edges and corners first, reducing isolated empty spaces.
- **Action Space Extensions:** Allow for shape rotation, flipping, or variable shape types and sizes.
- **Penalize Isolated Cells:** Add penalties for leaving single empty cells surrounded by filled cells.
- **Dueling/Double DQN:** Use advanced DQN variants for more robust value estimation.
- **Prioritized Experience Replay:** Sample important experiences more frequently during training.
- **Convolutional Networks:** Use CNNs to better process grid-based state representations.
- **Visual Attention Mechanisms:** Focus the agent's learning on the most relevant parts of the grid.
- **Self-Play/Adversarial Training:** Use competing agents or adversarial environments to improve robustness.
- **Transfer Learning:** Pretrain on similar tasks or use pretrained models to accelerate learning.

Feel free to experiment with or contribute any of these ideas to further improve the project!

## License
MIT License. See `LICENSE` for details.