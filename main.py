import math
import random
import json
import csv
import time
from collections import deque
from typing import Deque, Tuple, Optional, Dict, List
from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
import matplotlib.pyplot as plt

from constraints import Shape, Container
from models.dqn import create_dqn_agent, dqn_agent_step
from models.ppo import create_ppo_agent, ppo_update
from models.a2c import create_a2c_agent, a2c_update
# RL models available in models/ directory

class MetricsCollector:
    """Collects and manages training metrics for analysis and visualization."""
    
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.episode_metrics = []
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'filled_cells': [],
            'success_rate': [],
            'average_reward': [],
            'loss_values': [],
            'epsilon_values': [],
            'q_values': [],
            'total_steps': [],
            'timestamp': []
        }
        
        # Session info
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
    def log_episode(self, episode: int, reward: float, steps: int, filled_cells: int, 
                   loss: float = None, epsilon: float = None, avg_q_value: float = None):
        """Log metrics for a single episode."""
        timestamp = datetime.now()
        
        episode_data = {
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'filled_cells': filled_cells,
            'loss': loss,
            'epsilon': epsilon,
            'avg_q_value': avg_q_value,
            'timestamp': timestamp.isoformat()
        }
        
        self.episode_metrics.append(episode_data)
        
        # Update running metrics
        self.training_metrics['episode_rewards'].append(reward)
        self.training_metrics['episode_lengths'].append(steps)
        self.training_metrics['filled_cells'].append(filled_cells)
        self.training_metrics['total_steps'].append(sum(self.training_metrics['episode_lengths']))
        self.training_metrics['timestamp'].append(timestamp.isoformat())
        
        if loss is not None:
            self.training_metrics['loss_values'].append(loss)
        if epsilon is not None:
            self.training_metrics['epsilon_values'].append(epsilon)
        if avg_q_value is not None:
            self.training_metrics['q_values'].append(avg_q_value)
            
        # Calculate rolling averages
        window_size = min(100, len(self.training_metrics['episode_rewards']))
        recent_rewards = self.training_metrics['episode_rewards'][-window_size:]
        recent_filled = self.training_metrics['filled_cells'][-window_size:]
        
        self.training_metrics['average_reward'].append(np.mean(recent_rewards))
        self.training_metrics['success_rate'].append(np.mean([1 if r > 0 else 0 for r in recent_rewards]))
        
    def save_metrics(self, filename_prefix: str = None):
        """Save metrics to JSON and CSV files."""
        if filename_prefix is None:
            filename_prefix = f"training_metrics_{self.session_id}"
            
        # Save detailed episode data as JSON
        json_path = os.path.join(self.save_dir, f"{filename_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'session_info': {
                    'session_id': self.session_id,
                    'start_time': self.session_start.isoformat(),
                    'total_episodes': len(self.episode_metrics)
                },
                'episode_data': self.episode_metrics,
                'summary_metrics': self.training_metrics
            }, f, indent=2)
        
        # Save summary metrics as CSV for easy plotting
        csv_path = os.path.join(self.save_dir, f"{filename_prefix}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = ['episode', 'reward', 'avg_reward', 'filled_cells', 'success_rate', 'epsilon', 'loss', 'avg_q_value']
            writer.writerow(headers)
            
            # Write data
            for i, episode_data in enumerate(self.episode_metrics):
                row = [
                    episode_data['episode'],
                    episode_data['reward'],
                    self.training_metrics['average_reward'][i] if i < len(self.training_metrics['average_reward']) else '',
                    episode_data['filled_cells'],
                    self.training_metrics['success_rate'][i] if i < len(self.training_metrics['success_rate']) else '',
                    episode_data.get('epsilon', ''),
                    episode_data.get('loss', ''),
                    episode_data.get('avg_q_value', '')
                ]
                writer.writerow(row)
        
        print(f"Metrics saved to: {json_path} and {csv_path}")
        return json_path, csv_path
        
    def plot_metrics(self, save_plots: bool = True):
        """Generate and optionally save training plots."""
        if len(self.episode_metrics) < 2:
            print("Not enough data to plot.")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Metrics - Session {self.session_id}', fontsize=16)
        
        episodes = [d['episode'] for d in self.episode_metrics]
        
        # Plot 1: Episode Rewards and Moving Average
        axes[0, 0].plot(episodes, self.training_metrics['episode_rewards'], alpha=0.6, label='Episode Reward')
        axes[0, 0].plot(episodes, self.training_metrics['average_reward'], label='Moving Average (100 ep)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Filled Cells
        axes[0, 1].plot(episodes, self.training_metrics['filled_cells'], color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Filled Cells')
        axes[0, 1].set_title('Filled Cells per Episode')
        axes[0, 1].grid(True)
        
        # Plot 3: Success Rate
        axes[0, 2].plot(episodes, self.training_metrics['success_rate'], color='orange')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Success Rate')
        axes[0, 2].set_title('Success Rate (Moving Average)')
        axes[0, 2].grid(True)
        
        # Plot 4: Epsilon (if available)
        if self.training_metrics['epsilon_values']:
            axes[1, 0].plot(episodes[:len(self.training_metrics['epsilon_values'])], 
                           self.training_metrics['epsilon_values'], color='red')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_title('Exploration Rate (Epsilon)')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Epsilon Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Exploration Rate (Epsilon)')
        
        # Plot 5: Loss (if available)
        if self.training_metrics['loss_values']:
            axes[1, 1].plot(episodes[:len(self.training_metrics['loss_values'])], 
                           self.training_metrics['loss_values'], color='purple')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Loss')
        
        # Plot 6: Episode Length
        axes[1, 2].plot(episodes, self.training_metrics['episode_lengths'], color='brown')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps')
        axes[1, 2].set_title('Episode Length')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.save_dir, f"training_plots_{self.session_id}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to: {plot_path}")
        
        plt.show()
        return fig

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

def train_dqn_agent(episodes: int = 1000, verbose: bool = True, metrics_collector: MetricsCollector = None):
    """Train a DQN agent on the container environment with comprehensive metrics collection."""
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
        steps = 0
        episode_loss = 0
        loss_count = 0
        
        # Calculate current epsilon
        epsilon = max(0.01, 1.0 - episode / episodes)
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:  # Fallback for no valid actions
                break
                
            # Epsilon-greedy action selection with masking
            if random.random() < epsilon:
                action = random.choice(valid_actions)
                avg_q_value = None
            else:
                with torch.no_grad():
                    q_values = agent['qnetwork_local'](torch.FloatTensor(state))
                    # Mask invalid actions
                    masked_q = torch.full_like(q_values, float('-inf'))
                    masked_q[valid_actions] = q_values[valid_actions]
                    action = masked_q.argmax().item()
                    avg_q_value = q_values[valid_actions].mean().item()
            
            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.flatten()
            
            # Get loss from DQN step
            loss = dqn_agent_step(agent, state, action, reward, next_state, done)
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        filled_cells = info.get('filled_cells', 0)
        
        # Calculate average loss for episode
        avg_loss = episode_loss / loss_count if loss_count > 0 else None
        avg_q_val = avg_q_value if 'avg_q_value' in locals() else None
        
        # Log metrics
        if metrics_collector:
            metrics_collector.log_episode(
                episode=episode + 1,
                reward=total_reward,
                steps=steps,
                filled_cells=filled_cells,
                loss=avg_loss,
                epsilon=epsilon,
                avg_q_value=avg_q_val
            )
        
        if verbose and (episode + 1) % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode + 1:>4}/{episodes} | Average Score: {avg_score:.2f} | Filled Cells: {filled_cells} | Epsilon: {epsilon:.3f}")
    
    return agent

def main():
    """Main function to demonstrate the environment and train agents with metrics collection."""
    print("Space RL Container Packing Environment")
    print("=" * 50)
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # Train DQN agent
    try:
        print("Starting DQN training with metrics collection...")
        trained_agent = train_dqn_agent(episodes=500, metrics_collector=metrics_collector)  # Reduced episodes for demo
        print("DQN training completed successfully!")
        
        # Save the trained model
        model_path = "dqn_container.pt"
        torch.save(trained_agent['qnetwork_local'].state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save metrics
        json_path, csv_path = metrics_collector.save_metrics()
        
        # Generate and save plots
        print("Generating training plots...")
        metrics_collector.plot_metrics(save_plots=True)
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        # Print summary statistics
        if metrics_collector.episode_metrics:
            rewards = [ep['reward'] for ep in metrics_collector.episode_metrics]
            filled_cells = [ep['filled_cells'] for ep in metrics_collector.episode_metrics]
            
            print(f"Total Episodes: {len(metrics_collector.episode_metrics)}")
            print(f"Average Reward: {np.mean(rewards):.2f}")
            print(f"Best Reward: {np.max(rewards):.2f}")
            print(f"Average Filled Cells: {np.mean(filled_cells):.2f}")
            print(f"Best Filled Cells: {np.max(filled_cells)}")
            print(f"Success Rate (last 100 episodes): {metrics_collector.training_metrics['success_rate'][-1]:.2%}")
        
        print(f"\nMetrics saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        # Still save metrics if any were collected
        if metrics_collector.episode_metrics:
            metrics_collector.save_metrics("failed_training")
        return
    
    # Quick evaluation
    print("\nRunning final evaluation...")
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
