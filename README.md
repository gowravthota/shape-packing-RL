# Space RL - 2D Container Packing with Reinforcement Learning

A robust 2D space optimization and container packing environment using Reinforcement Learning (RL) with comprehensive error handling and fallback systems.

## Overview
This project implements a container-packing problem where RL agents learn to optimally place shapes inside containers with various constraints. The environment is built with OpenAI Gym and PyTorch, featuring multiple RL algorithms with improved architectures and error handling.

## Key Features
- **Custom 2D Environment**: Container-packing environment (`ContainerEnv`) with discrete action space
- **Multiple RL Algorithms**: DQN, PPO, and A2C implementations with modern improvements
- **Advanced Constraints**: Support for irregular containers, polygonal obstacles, and collision detection
- **Robust Error Handling**: Graceful degradation and comprehensive fallback systems
- **Flexible Visualization**: Multiple backends (pygame/matplotlib) with automatic fallback
- **Action Masking**: Prevents invalid moves to improve learning efficiency
- **Device Management**: Automatic CUDA/GPU detection and utilization

## Container Types
- **Regular Containers**: Standard rectangular boundaries
- **Irregular Containers**: Arbitrary polygonal shapes for realistic scenarios
- **Obstacle Support**: Define forbidden zones as polygons within containers
- **Constraint Validation**: Robust boundary and collision checking

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Setup
```bash
git clone <repository-url>
cd space_RL
pip install -r requirements.txt
```

### Dependencies
The project automatically handles missing dependencies with graceful fallbacks:
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computations
- `gym>=0.26.0` - RL environment framework
- `pygame>=2.1.0` - Interactive visualization (optional)
- `matplotlib>=3.5.0` - Static visualization fallback

## Usage

### Basic Training
Run the default DQN training:
```bash
python main.py
```

### Test Core Functionality (No ML Dependencies)
```bash
python test_basic.py
```

### Visualization
```bash
python visualization.py
```

## Architecture

### Project Structure
```
space_RL/
├── main.py              # Main training loop and environment
├── constraints.py       # Shape and container system
├── models/             
│   ├── dqn.py          # Deep Q-Network implementation
│   ├── ppo.py          # Proximal Policy Optimization
│   └── a2c.py          # Advantage Actor-Critic
├── visualization.py     # Flexible visualization system
├── test_basic.py       # Dependency-free testing
├── requirements.txt    # Project dependencies
└── notes.txt          # Project documentation
```

### RL Models Available
- **DQN**: Enhanced with dropout, gradient clipping, and improved target updates
- **PPO**: Features entropy regularization and advantage normalization
- **A2C**: Includes entropy bonuses and robust loss computation

## Environment Details

### State Space
- 10×10 grid representation (configurable)
- Binary occupancy values (0=empty, 1=occupied)
- Container boundary and obstacle information

### Action Space
- Discrete: 100 possible grid cell placements
- Action masking prevents invalid placements
- Automatic validation against constraints

### Reward Structure
- +1 for successful shape placement
- -1 for invalid placement attempts
- Episode terminates when grid is full or step limit reached

## Model Training Examples

### Using Different Algorithms
```python
from models.dqn import create_dqn_agent, dqn_agent_step
from models.ppo import create_ppo_agent, ppo_update
from models.a2c import create_a2c_agent, a2c_update

# Create agents with automatic device detection
dqn_agent = create_dqn_agent(state_size=100, action_size=100)
ppo_agent = create_ppo_agent(state_dim=100, action_dim=100)
a2c_agent = create_a2c_agent(state_dim=100, action_dim=100)
```

### Custom Container Configurations
```python
# Regular rectangular container
env = ContainerEnv(use_irregular=False)

# Irregular container with obstacles
env = ContainerEnv(use_irregular=True)  # Uses predefined polygon and obstacles
```

## Key Improvements

### Robustness
- Comprehensive error handling throughout
- Graceful degradation for missing dependencies
- Automatic device detection (CUDA/CPU)
- Input validation and bounds checking

### Performance
- Action masking for efficient learning
- Gradient clipping and normalization
- Optimized memory usage
- Device-aware tensor operations

### Maintainability
- Type hints throughout codebase
- Modular architecture
- Comprehensive documentation
- Consistent naming conventions

## Testing

The project includes a comprehensive test suite that works without ML dependencies:

```bash
python test_basic.py
```

This validates:
- Shape creation and manipulation
- Container constraint enforcement
- Collision detection algorithms
- Visualization backend availability

## Configuration

### Environment Parameters
- `GRID_CELLS`: Grid resolution (default: 10×10)
- `CONTAINER_W/H`: Container dimensions (default: 100×100)
- `max_steps`: Episode length limit

### Training Parameters
- Configurable learning rates per algorithm
- Adjustable network architectures
- Customizable hyperparameters

## Error Handling

The system includes robust error handling for:
- Missing dependencies (graceful fallbacks)
- Invalid shape placements
- Device availability issues
- Memory constraints
- Visualization backend failures

## Performance Monitoring

Track training progress with built-in metrics:
- Episode rewards and scores
- Placement efficiency
- Container utilization rates
- Training convergence indicators

## License

MIT License - see `LICENSE` file for details.

## Contributing

Contributions are welcome! The codebase is designed for easy extension and modification. Please ensure new code includes appropriate error handling and type hints.