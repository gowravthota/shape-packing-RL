# Space RL - 2D Container Packing with Reinforcement Learning

A robust 2D space optimization and container packing environment using Reinforcement Learning (RL) with real-time visualization and comprehensive training capabilities.

## Overview
This project implements a **continuous container-packing problem** where RL agents learn to optimally place complex shapes inside containers with multiple constraints. The environment features continuous positioning, rotation, realistic collision detection, and an interactive pygame visualization system.

## 🎮 Live Visualization
**NEW**: Interactive real-time visualization showing agent performance!

![Agent Demo](https://img.shields.io/badge/Status-Live%20Demo%20Ready-brightgreen)

## Key Features
- **🎯 Continuous Action Space**: Precise positioning (x,y) and rotation (0-360°) for each shape
- **🧠 Advanced PPO Training**: Proximal Policy Optimization with curriculum learning
- **🎮 Interactive Visualization**: Real-time pygame display with controls and metrics
- **📊 Multiple Shape Types**: Rectangles, circles, triangles, L-shapes, and irregular polygons
- **🎓 Curriculum Learning**: Progressive difficulty levels for improved training
- **⚡ Robust Architecture**: Comprehensive error handling and device management
- **📈 Comprehensive Metrics**: Utilization tracking, success rates, and performance analysis

## Container Types
- **Rectangular Containers**: Standard 100x100 unit containers (configurable)
- **Curriculum Levels**: 5 difficulty levels with increasing shape complexity
- **Collision Detection**: Realistic physics using Shapely geometry
- **Space Optimization**: Reward system optimized for maximum utilization

## Installation

### Prerequisites
- Python 3.9+ (tested on 3.9.6)
- pip package manager

### Quick Setup
```bash
git clone <repository-url>
cd space_RL
pip install -r requirements.txt
```

### Dependencies
```bash
pip install torch>=2.0.0 numpy>=1.21.0 gym>=0.26.0 pygame>=2.1.0 matplotlib>=3.5.0 shapely>=2.0.0 pandas>=1.5.0 seaborn>=0.11.0
```

## 🚀 Usage

### 1. Test the System
```bash
python demo.py
```
**Output**: Validates shapes, environment, and curriculum system

### 2. Interactive Visualization (Recommended!)
```bash
python visualize_agent.py
```
**Features**:
- Real-time agent performance display
- Interactive controls (pause, speed adjustment, episode reset)
- Live metrics (reward, utilization, shapes placed)
- Visual feedback for successful/failed placements

**Controls**:
- `SPACE`: Pause/Resume
- `R`: Reset episode
- `Q`: Quit
- `UP/DOWN`: Adjust animation speed

### 3. Training
```bash
# Original training script
python train.py

# Alternative continuous training
python train_continuous_agent.py  
```

## 🏗️ Architecture

### Current Project Structure
```
space_RL/
├── 🎮 visualize_agent.py    # NEW: Interactive pygame visualization
├── 🧪 demo.py               # System validation and testing
├── 🚀 train.py              # Main PPO training script
├── 🚀 train_continuous_agent.py  # Alternative training script
├── 🧠 agent.py              # PPO trainer and neural network
├── 🌍 env.py                # Continuous container environment
├── 🔷 shapes.py             # Shape definitions and factory
├── 📊 analyze_metrics.py    # Training analysis tools
├── 📋 requirements.txt      # Project dependencies
├── 📝 notes.txt            # Development notes
└── 📈 metrics/             # Training metrics and plots
```

### 🧠 Neural Network Architecture
- **Actor-Critic PPO**: Separate policy and value networks
- **Continuous Actions**: Multi-head output for shape selection, positioning, rotation
- **Device Aware**: Automatic CUDA/CPU detection
- **674,843 Parameters**: Optimized for complex spatial reasoning

## 🎯 Environment Details

### Action Space (Continuous)
```python
Box(4,) = [shape_id, x_position, y_position, rotation_angle]
# shape_id: 0-20 (discrete selection from available shapes)
# x_position: 0-100 (continuous positioning)  
# y_position: 0-100 (continuous positioning)
# rotation_angle: 0-360 (continuous rotation in degrees)
```

### Observation Space
```python
Box(264,) = [
    container_metrics(4) +      # Utilization, free space, area, perimeter
    occupancy_grid(100) +       # 10x10 grid of occupied spaces
    available_shapes(160)       # Encoded shape information (8 features × 20 shapes)
]
```

### Reward Structure
- **✅ Successful Placement**: +50 base reward + utilization bonus
- **❌ Collision**: -10 penalty + shape overlap penalty
- **🎯 Utilization Bonus**: Exponential reward for efficient space usage
- **🏆 Completion Bonus**: +100 for placing all shapes
- **📦 Compactness Bonus**: Reward for tightly packed arrangements

## 🎓 Curriculum Learning

### 5 Progressive Difficulty Levels
1. **Level 1**: 5-8 simple shapes, basic complexity
2. **Level 2**: 8-12 shapes, moderate complexity
3. **Level 3**: 12-16 shapes, tetris-like challenges
4. **Level 4**: 16-20 shapes, efficiency focus
5. **Level 5**: 20-25 shapes, ultimate challenge

**Advancement Criteria**: Success rate + utilization thresholds

## 📊 Performance Monitoring

### Real-time Metrics
- Episode rewards and cumulative scores
- Container space utilization (%)
- Successful shape placements vs. collisions
- Training convergence indicators
- Curriculum advancement tracking

### Available Metrics Files
```bash
📈 metrics/
├── training_metrics_*.csv    # Raw training data
├── training_metrics_*.json   # Structured metrics
├── training_plots_*.png      # Visualization plots
└── analysis_report.txt       # Performance analysis
```

## 🔧 Recent Major Updates

### ✅ Fixed Issues
- **Tensor Dimension Errors**: Resolved stack size mismatches in action selection
- **Import Dependencies**: Fixed module paths and missing imports
- **Device Management**: Proper CUDA/CPU tensor handling
- **Gradient Tracking**: Added `.detach()` for numpy conversions
- **Visualization Backend**: Working pygame display with shape rendering

### 🆕 New Features
- **Interactive Visualization**: Real-time agent performance display
- **Curriculum Manager**: Automatic difficulty progression
- **Enhanced Metrics**: Comprehensive performance tracking
- **Shape Rendering**: Accurate visualization of rotated shapes
- **Control Interface**: Pause, speed control, episode management

## 🎮 Visualization Features

### Real-time Display
- **Container View**: 400x400 pixel scaled container (4x zoom)
- **Shape Rendering**: Accurate rectangles and circles with rotation
- **Color Coding**: Different colors for placed vs. current shapes
- **Success Indicators**: Visual feedback for placements

### Information Panel
- Episode and step counters
- Current reward and utilization metrics
- Last action details (shape, position, rotation)
- Interactive controls help

## 🚀 Training Performance

### Current Results (Untrained Agent)
- **Average Episode Reward**: -52 to -120 (baseline)
- **Space Utilization**: 1-3% (random placement)
- **Success Rate**: ~10% shape placement success
- **Episode Length**: 15-30 steps average

### Expected Trained Performance
- **Target Utilization**: 60-80% with proper training
- **Success Rate**: 80%+ with curriculum learning
- **Reward Range**: +200 to +500 for well-trained agents

## 🔬 Testing

### Comprehensive Test Suite
```bash
python demo.py  # Full system validation
```

**Validates**:
- ✅ Shape creation and manipulation
- ✅ Environment functionality and action space
- ✅ Curriculum system progression
- ✅ Collision detection accuracy
- ✅ Reward calculation system

## ⚙️ Configuration

### Environment Parameters (Configurable)
```python
env = ContinuousContainerEnv(
    container_width=100,      # Container dimensions
    container_height=100,
    max_shapes=20,           # Maximum shapes per episode
    curriculum_level=1,      # Difficulty level (1-5)
    reward_mode="utilization" # Reward calculation method
)
```

### Training Parameters
```python
trainer = PPOTrainer(
    lr=3e-4,              # Learning rate
    clip_ratio=0.2,       # PPO clipping parameter
    value_coef=0.5,       # Value loss coefficient
    entropy_coef=0.01     # Exploration bonus
)
```

## 🔄 Development Workflow

### For New Users
1. `python demo.py` - Validate installation
2. `python visualize_agent.py` - See agent in action
3. `python train.py` - Start training (optional)

### For Developers
1. Modify environment parameters in `env.py`
2. Adjust network architecture in `agent.py`
3. Test changes with `demo.py`
4. Visualize results with `visualize_agent.py`

## 🤝 Contributing

Contributions welcome! The codebase follows functional programming principles and includes:
- Type hints throughout
- Comprehensive error handling
- Modular architecture
- Extensive documentation

## 📄 License

MIT License - see `LICENSE` file for details.

---

## 🎯 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the system  
python demo.py

# 3. Watch the agent in action!
python visualize_agent.py
```

**🎮 Enjoy watching your RL agent learn to pack shapes efficiently!**