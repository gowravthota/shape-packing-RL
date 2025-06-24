# Shape Fitting Reinforcement Learning

A reinforcement learning project where an agent learns to fit groups of shapes into containers using deep reinforcement learning (PPO) with discrete rotation control.

## 🎯 Project Overview

This project trains an AI agent to:
- **Fit groups of predefined shapes** into a rectangular container
- **Maximize the number of successfully fitted shapes** (primary reward)
- **Use discrete 20-degree rotation intervals** (0°, 20°, 40°, ..., 340°)
- **Handle multiple shape types**: rectangles, circles, triangles, L-shapes, and irregular polygons
- **Progress through difficulty levels** with increasingly complex shape combinations

## 🏗️ Architecture

### Environment (`env.py`)
- **`ShapeFittingEnv`**: Main RL environment
- **Action Space**: [shape_id, x_position, y_position, rotation_index]
  - `shape_id`: Which shape from the group to place (discrete: 0 to num_shapes-1)
  - `x, y`: Position coordinates (continuous: 0 to container_width/height)
  - `rotation_index`: Discrete rotation (0-17, representing 0° to 340° in 20° steps)
- **Observation Space**: Container state + occupancy grid + shape information
- **Reward System**:
  - **+100** for each successful shape fit
  - **-10** for collision attempts
  - **-5** for trying to place already fitted shapes
  - Small efficiency bonus for compact placement

### Agent (`agent.py`)
- **`ShapeFittingActorCritic`**: Neural network with mixed discrete/continuous actions
  - Shape selection: Categorical distribution
  - Position: Normal distributions for x, y coordinates
  - Rotation: Categorical distribution over 18 discrete angles
- **`PPOTrainer`**: Proximal Policy Optimization implementation
  - Collects trajectories and updates policy
  - Includes GAE (Generalized Advantage Estimation)
  - Gradient clipping and learning rate scheduling

### Shapes (`shapes.py`)
- **Multiple shape types**: Rectangle, Circle, Triangle, L-Shape, Irregular
- **`is_fitted` tracking**: Each shape knows if it's been successfully placed
- **Difficulty-based generation**: Progressive complexity levels
- **Realistic collision detection** using Shapely geometry

### Training (`train.py`)
- **Progressive difficulty**: Automatic advancement through 5 difficulty levels
- **Comprehensive evaluation**: Multiple test scenarios
- **Metrics tracking**: Rewards, success rates, shapes fitted
- **Model checkpointing**: Regular saves during training

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from env import ShapeFittingEnv
from agent import PPOTrainer

# Create environment
env = ShapeFittingEnv(
    container_width=100,
    container_height=100,
    num_shapes_to_fit=10,
    difficulty_level=1,
    max_steps=50
)

# Create and train agent
trainer = PPOTrainer(
    env=env,
    obs_size=env.observation_space.shape[0],
    num_shapes=env.num_shapes_to_fit,
    num_rotations=len(env.rotation_angles)
)

# Train for 1M timesteps
trainer.train(total_steps=1000000)
```

### Full Training Pipeline

```bash
# Run complete training with progressive difficulty
python train.py
```

This will:
- Train for 2M timesteps with progressive difficulty
- Evaluate on multiple scenarios every 100k steps
- Save models and metrics regularly
- Generate training plots

## 📊 Evaluation Scenarios

The project includes 5 evaluation scenarios:

1. **Basic Shapes**: Simple rectangles and circles
2. **Mixed Shapes**: Rectangles, circles, and triangles  
3. **Complex Shapes**: All shape types including L-shapes
4. **Expert Challenge**: Maximum complexity with irregular shapes
5. **Tight Space**: Large shapes in smaller containers

## 🎛️ Configuration

### Difficulty Levels

| Level | Shape Types | Size Range | Advancement Criteria |
|-------|-------------|------------|---------------------|
| 1 | Rectangle, Circle | 8-20 units | 70% success, 7+ shapes |
| 2 | + Triangle | 6-22 units | 60% success, 6+ shapes |
| 3 | + L-Shape | 5-25 units | 50% success, 5+ shapes |
| 4 | + Irregular | 4-28 units | 40% success, 4+ shapes |
| 5 | All types | 3-30 units | 30% success, 3+ shapes |

### Hyperparameters

```python
# PPO Hyperparameters
lr = 3e-4              # Learning rate
clip_ratio = 0.2       # PPO clipping ratio
value_coef = 0.5       # Value function coefficient
entropy_coef = 0.01    # Entropy bonus coefficient
gamma = 0.99           # Discount factor
gae_lambda = 0.95      # GAE lambda parameter
```

## 📈 Monitoring

### Training Metrics
- Average reward per episode
- Number of shapes fitted
- Success rate (percentage of shapes fitted)
- Policy and value losses
- Training curves and visualizations

### Files Generated
- `models/`: Saved model checkpoints
- `metrics/`: Training metrics and evaluation results
- `training_plots_*.png`: Visualizations of training progress

## 🧪 Visualization

```python
# Render environment during training/testing
env.render()  # Shows container with fitted shapes + remaining shapes

# Plot training curves
trainer.plot_training_curves()

# Evaluate on specific scenario
from train import run_scenario_evaluation, create_evaluation_scenarios
scenarios = create_evaluation_scenarios()
results = run_scenario_evaluation(trainer, scenarios[0])
```

## 🔧 Key Features

- **Discrete Rotation**: 18 rotation angles (20° increments) for more stable learning
- **Shape Fitting Focus**: Reward based on number of shapes fitted, not space utilization
- **Progressive Difficulty**: Automatic curriculum learning with 5 difficulty levels
- **Realistic Physics**: Proper collision detection using Shapely geometry
- **Multiple Shape Types**: Rectangles, circles, triangles, L-shapes, irregular polygons
- **Comprehensive Evaluation**: 5 different test scenarios measuring various skills
- **Functional Code Design**: Clean, modular, and extensible architecture

## 📁 Project Structure

```
space_RL/
├── env.py              # Shape fitting environment
├── agent.py            # PPO agent and neural networks
├── shapes.py           # Shape classes and factory
├── train.py            # Training pipeline and evaluation
├── requirements.txt    # Dependencies
├── models/             # Saved model checkpoints
├── metrics/            # Training metrics and results
└── README.md          # This file
```

## 🎯 Success Metrics

The agent is considered successful when it can:
- Fit 70%+ of shapes in basic scenarios (Level 1)
- Fit 50%+ of shapes in complex scenarios (Level 3+)
- Demonstrate spatial reasoning and rotation selection
- Progress through all 5 difficulty levels
- Achieve consistent performance across evaluation scenarios

## 🤝 Contributing

This project follows functional programming principles and maintains clean, modular code. Key areas for improvement:
- Advanced shape types and constraints
- Multi-container environments
- Hierarchical planning approaches
- Transfer learning between difficulty levels

---

**Note**: This project transforms container packing into shape fitting with discrete rotations, focusing on the number of shapes successfully fitted rather than space utilization optimization.