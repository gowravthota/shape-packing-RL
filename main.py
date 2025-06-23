import math
import random
from collections import deque
from typing import Deque, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

from constraints import Shape, Container
from models.dqn import create_dqn_agent, dqn_agent_step
from models.ppo import create_ppo_agent, ppo_update
from models.a2c import create_a2c_agent, a2c_update
# RL models available in models/ directory

def _patched_contains(self, shape: Shape) -> bool:
    """Patch for Container.contains method to fix bounding box logic."""
    left, top, right, bottom = shape.bounding_box()
    return (
        left >= self.x and right <= self.x + self.width and 
        top >= self.y and bottom <= self.y + self.height
    )

Container.contains = _patched_contains

# ────────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ────────────────────────────────────────────────────────────────────────────────

GRID_CELLS = 10  # resolution of the placement grid (10×10 → 100 possible positions)
CONTAINER_W = 100
CONTAINER_H = 100
CELL_SIZE = CONTAINER_W / GRID_CELLS  # square cells → CELL_SIZE×CELL_SIZE pixels

# Example irregular polygon (pentagon) and obstacles (triangles)
IRREGULAR_POLYGON = [
    (10, 10), (90, 10), (95, 50), (50, 90), (10, 60)
]
OBSTACLES = [
    [(30, 30), (40, 30), (35, 40)],
    [(60, 60), (70, 60), (65, 70)]
]

# ────────────────────────────────────────────────────────────────────────────────
# Gym Environment
# ────────────────────────────────────────────────────────────────────────────────

class ContainerEnv(gym.Env):
    """Simplified container‑packing environment with a discrete action space.

    Each action chooses a grid cell in which we *attempt* to drop a square shape
    of side length *CELL_SIZE*. The environment rewards successful placements
    (+1) and penalises invalid placements (‑1). The episode terminates once the
    grid is full or after *max_steps* actions, whichever comes first.
    """

    metadata = {"render.modes": []}

    def __init__(self, max_steps: Optional[int] = None, use_irregular: bool = True):
        super().__init__()
        self.max_steps = max_steps or GRID_CELLS * GRID_CELLS
        self.action_space = spaces.Discrete(GRID_CELLS * GRID_CELLS)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(GRID_CELLS, GRID_CELLS),
            dtype=np.int8,
        )
        self._container: Optional[Container] = None
        self._grid: Optional[np.ndarray] = None
        self._steps = 0
        self.use_irregular = use_irregular

    def _idx_to_xy(self, idx: int) -> Tuple[float, float]:
        """Helper converting a discrete action index → (x, y) pixel coordinates."""
        row, col = divmod(idx, GRID_CELLS)
        x = col * CELL_SIZE + CELL_SIZE / 2.0
        y = row * CELL_SIZE + CELL_SIZE / 2.0
        return x, y

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if self.use_irregular:
            self._container = Container(
                0, 0, CONTAINER_W, CONTAINER_H, 
                polygon=IRREGULAR_POLYGON, obstacles=OBSTACLES
            )
        else:
            self._container = Container(0, 0, CONTAINER_W, CONTAINER_H)
        self._grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.int8)
        self._steps = 0
        return self._grid.copy(), {}

    def step(self, action: int):
        assert self._grid is not None and self._container is not None, "Call reset() first!"
        self._steps += 1
        row, col = divmod(action, GRID_CELLS)

        reward = 0.0
        done = False
        info = {}

        if self._grid[row, col] == 1:
            reward = -1.0  # Already occupied
        else:
            x, y = self._idx_to_xy(action)
            shape = Shape(x, y, size=CELL_SIZE)
            if self._container.add_shape(shape):
                self._grid[row, col] = 1
                reward = 1.0
            else:
                # Penalize for out-of-bounds or obstacle placement
                reward = -1.0

        filled = int(self._grid.sum())
        if filled == GRID_CELLS * GRID_CELLS or self._steps >= self.max_steps:
            done = True
            info["filled_cells"] = filled

        return self._grid.copy(), reward, done, False, info

    def get_valid_actions(self) -> list[int]:
        """Get list of valid (unoccupied) action indices."""
        if self._grid is None:
            return list(range(self.action_space.n))
        return [i for i in range(self.action_space.n) if self._grid.flatten()[i] == 0]

# ────────────────────────────────────────────────────────────────────────────────
# Training Configuration and Main Function
# ────────────────────────────────────────────────────────────────────────────────

def train_dqn_agent(episodes: int = 1000, verbose: bool = True):
    """Train a DQN agent on the container environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    env = ContainerEnv()
    state_size = GRID_CELLS * GRID_CELLS
    action_size = GRID_CELLS * GRID_CELLS
    
    agent = create_dqn_agent(state_size, action_size)
    
    scores = deque(maxlen=100)
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = state.flatten()
        total_reward = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:  # Fallback for no valid actions
                break
                
            # Epsilon-greedy action selection with masking
            if random.random() < max(0.01, 1.0 - episode / episodes):
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = agent['qnetwork_local'](torch.FloatTensor(state))
                    # Mask invalid actions
                    masked_q = torch.full_like(q_values, float('-inf'))
                    masked_q[valid_actions] = q_values[valid_actions]
                    action = masked_q.argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.flatten()
            
            dqn_agent_step(agent, state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode + 1:>4}/{episodes} | Average Score: {avg_score:.2f}")
    
    return agent

def main():
    """Main function to demonstrate the environment and train agents."""
    print("Space RL Container Packing Environment")
    print("=" * 50)
    
    # Train DQN agent
    try:
        trained_agent = train_dqn_agent(episodes=1000)
        print("DQN training completed successfully!")
        
        # Save the trained model
        torch.save(trained_agent['qnetwork_local'].state_dict(), "dqn_container.pt")
        print("Model saved to dqn_container.pt")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Quick evaluation
    env = ContainerEnv()
    state, _ = env.reset(seed=42)
    done = False
    steps = 0
    
    while not done and steps < 100:  # Prevent infinite loops
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
            
        with torch.no_grad():
            q_values = trained_agent['qnetwork_local'](torch.FloatTensor(state.flatten()))
            masked_q = torch.full_like(q_values, float('-inf'))
            masked_q[valid_actions] = q_values[valid_actions]
            action = masked_q.argmax().item()
        
        state, _, done, _, info = env.step(action)
        steps += 1
    
    filled_cells = info.get('filled_cells', 0)
    print(f"Evaluation: Filled {filled_cells} of {GRID_CELLS * GRID_CELLS} cells in {steps} steps")

if __name__ == "__main__":
    main()
