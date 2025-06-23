

import gym
from gym import spaces
import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from shapes import (
    PackingShape, ShapeFactory, RectangleShape, CircleShape, 
    TriangleShape, LShapeShape, IrregularShape, BASIC_CHALLENGE, TETRIS_CHALLENGE
)

class Container:
    """Represents the container where shapes are packed."""
    
    def __init__(self, width: float, height: float, shape: str = "rectangle"):
        self.width = width
        self.height = height
        self.shape = shape
        self._geometry = self._create_container_geometry()
        self.placed_shapes: List[PackingShape] = []
        self._occupied_union = None
    
    def _create_container_geometry(self) -> Polygon:
        """Create the container boundary geometry."""
        if self.shape == "rectangle":
            return box(0, 0, self.width, self.height)
        elif self.shape == "circle":
            # Circular container
            radius = min(self.width, self.height) / 2
            center = (self.width/2, self.height/2)
            angles = np.linspace(0, 2*math.pi, 64, endpoint=False)
            points = [(center[0] + radius * math.cos(a), 
                      center[1] + radius * math.sin(a)) for a in angles]
            return Polygon(points)
        else:
            return box(0, 0, self.width, self.height)
    
    @property
    def geometry(self) -> Polygon:
        return self._geometry
    
    @property
    def area(self) -> float:
        return self._geometry.area
    
    @property
    def occupied_area(self) -> float:
        """Get total area occupied by placed shapes."""
        if not self.placed_shapes:
            return 0.0
        if self._occupied_union is None:
            self._update_occupied_union()
        return self._occupied_union.area if self._occupied_union else 0.0
    
    @property
    def utilization(self) -> float:
        """Get space utilization ratio (0-1)."""
        return self.occupied_area / self.area
    
    def _update_occupied_union(self):
        """Update the union of all occupied spaces."""
        if not self.placed_shapes:
            self._occupied_union = None
        else:
            geometries = [shape.geometry for shape in self.placed_shapes]
            self._occupied_union = unary_union(geometries)
    
    def can_place_shape(self, shape: PackingShape) -> bool:
        """Check if a shape can be placed without collisions."""
        # Check if shape is within container bounds
        if not self._geometry.contains(shape.geometry):
            return False
        
        # Check for collisions with existing shapes
        for existing_shape in self.placed_shapes:
            if shape.intersects(existing_shape):
                return False
        
        return True
    
    def place_shape(self, shape: PackingShape) -> bool:
        """Attempt to place a shape in the container."""
        if self.can_place_shape(shape):
            self.placed_shapes.append(shape)
            self._occupied_union = None  # Reset union cache
            return True
        return False
    
    def remove_shape(self, shape: PackingShape) -> bool:
        """Remove a shape from the container."""
        if shape in self.placed_shapes:
            self.placed_shapes.remove(shape)
            self._occupied_union = None  # Reset union cache
            return True
        return False
    
    def clear(self):
        """Remove all shapes from the container."""
        self.placed_shapes.clear()
        self._occupied_union = None
    
    def get_free_space_ratio(self) -> float:
        """Get the ratio of free space to total space."""
        return 1.0 - self.utilization
    
    def get_shape_bounds_penalty(self, shape: PackingShape) -> float:
        """Calculate penalty for shapes extending outside container."""
        if self._geometry.contains(shape.geometry):
            return 0.0
        
        # Calculate how much of the shape is outside
        intersection = self._geometry.intersection(shape.geometry)
        if intersection.is_empty:
            return 1.0  # Completely outside
        
        inside_area = intersection.area
        return 1.0 - (inside_area / shape.area)

class ContinuousContainerEnv(gym.Env):
    """
    Continuous container packing environment.
    
    Action Space:
        - shape_id: Discrete(num_available_shapes) - which shape to place
        - x: Box(0, container_width) - x position
        - y: Box(0, container_height) - y position  
        - rotation: Box(0, 360) - rotation angle in degrees
    
    Observation Space:
        - Container state representation
        - Available shapes information
        - Current utilization metrics
    """
    
    def __init__(self, 
                 container_width: float = 100.0,
                 container_height: float = 100.0,
                 max_shapes: int = 20,
                 curriculum_level: int = 1,
                 container_shape: str = "rectangle",
                 reward_mode: str = "utilization"):
        
        super().__init__()
        
        # Environment parameters
        self.container_width = container_width
        self.container_height = container_height
        self.max_shapes = max_shapes
        self.curriculum_level = curriculum_level
        self.reward_mode = reward_mode
        
        # Create container
        self.container = Container(container_width, container_height, container_shape)
        
        # Shape management
        self.available_shapes: List[PackingShape] = []
        self.current_shape_idx = 0
        self.shapes_placed = 0
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.best_utilization = 0.0
        
        # Difficulty settings
        self.difficulty_settings = {
            1: {"min_shapes": 5, "max_shapes": 8, "shape_complexity": 1.0},
            2: {"min_shapes": 8, "max_shapes": 12, "shape_complexity": 1.5},
            3: {"min_shapes": 12, "max_shapes": 16, "shape_complexity": 2.0},
            4: {"min_shapes": 16, "max_shapes": 20, "shape_complexity": 2.5},
            5: {"min_shapes": 20, "max_shapes": 25, "shape_complexity": 3.0},
        }
        
        self.reset()
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        
        # Action space: [shape_id, x, y, rotation]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([20, self.container_width, self.container_height, 360]),
            dtype=np.float32
        )
        
        # Observation space: container state + shape info + metrics
        obs_size = (
            4 +  # Container metrics (utilization, free_space, area, perimeter)
            100 +  # Container occupancy grid (10x10)
            60 +  # Current shape info (position, rotation, bounds, area, etc.)
            100  # Available shapes summary
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset the environment for a new episode."""
        # Clear container
        self.container.clear()
        
        # Generate new shape set based on curriculum
        self._generate_shape_set()
        
        # Reset episode tracking
        self.current_shape_idx = 0
        self.shapes_placed = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.best_utilization = 0.0
        
        return self._get_observation()
    
    def _generate_shape_set(self):
        """Generate a challenging set of shapes for this episode."""
        settings = self.difficulty_settings.get(self.curriculum_level, self.difficulty_settings[5])
        
        num_shapes = random.randint(settings["min_shapes"], settings["max_shapes"])
        complexity = settings["shape_complexity"]
        
        self.available_shapes = []
        
        # Mix of different shape types based on complexity
        for _ in range(num_shapes):
            shape = ShapeFactory.create_random_shape(complexity)
            self.available_shapes.append(shape)
        
        # Add some challenging combinations
        if complexity >= 2.0:
            # Add shapes that require rotation to fit efficiently
            self.available_shapes.extend([
                RectangleShape(35, 8),  # Long thin rectangle
                RectangleShape(8, 35),  # Tall thin rectangle
                LShapeShape(20, 6),     # L-shape requiring careful placement
            ])
        
        # Shuffle for randomness
        random.shuffle(self.available_shapes)
    
    def step(self, action):
        """Execute one step in the environment."""
        self.episode_length += 1
        
        # Parse action
        shape_idx = int(action[0]) % len(self.available_shapes)
        x, y = float(action[1]), float(action[2])
        rotation = float(action[3]) % 360
        
        # Get current shape to place
        if shape_idx < len(self.available_shapes):
            shape = self.available_shapes[shape_idx]
            
            # Set position and rotation
            shape.move_to((x, y))
            shape.rotate_to(rotation)
            
            # Try to place the shape
            reward, done, info = self._attempt_placement(shape, shape_idx)
        else:
            reward = -0.1  # Invalid shape selection
            done = False
            info = {"placement": "invalid_shape"}
        
        self.episode_reward += reward
        
        # Check if episode should end
        if not done:
            done = self._check_episode_end()
        
        return self._get_observation(), reward, done, info
    
    def _attempt_placement(self, shape: PackingShape, shape_idx: int) -> Tuple[float, bool, Dict]:
        """Attempt to place a shape and calculate reward."""
        
        if self.container.can_place_shape(shape):
            # Successful placement
            self.container.place_shape(shape)
            self.available_shapes.pop(shape_idx)
            self.shapes_placed += 1
            
            # Calculate reward based on multiple factors
            reward = self._calculate_placement_reward(shape)
            
            # Update best utilization
            current_util = self.container.utilization
            if current_util > self.best_utilization:
                self.best_utilization = current_util
            
            info = {
                "placement": "success",
                "area_placed": shape.area,
                "utilization": current_util,
                "shapes_remaining": len(self.available_shapes)
            }
            
            # Check if all shapes placed
            done = len(self.available_shapes) == 0
            if done:
                reward += self._calculate_completion_bonus()
            
        else:
            # Failed placement
            reward = self._calculate_collision_penalty(shape)
            done = False
            info = {
                "placement": "collision",
                "bounds_penalty": self.container.get_shape_bounds_penalty(shape)
            }
        
        return reward, done, info
    
    def _calculate_placement_reward(self, shape: PackingShape) -> float:
        """Calculate reward for successful shape placement."""
        base_reward = shape.area * 0.1  # Base reward proportional to area
        
        # Efficiency bonus - reward for good space utilization
        utilization = self.container.utilization
        efficiency_bonus = utilization * 10.0
        
        # Difficulty bonus - harder shapes worth more
        difficulty_bonus = shape.shape_def.difficulty * shape.area * 0.05
        
        # Compactness bonus - reward for filling gaps
        compactness_bonus = self._calculate_compactness_bonus(shape)
        
        total_reward = base_reward + efficiency_bonus + difficulty_bonus + compactness_bonus
        
        return max(0.1, total_reward)  # Minimum positive reward
    
    def _calculate_collision_penalty(self, shape: PackingShape) -> float:
        """Calculate penalty for collision or invalid placement."""
        
        # Base penalty
        penalty = -1.0
        
        # Extra penalty for going out of bounds
        bounds_penalty = self.container.get_shape_bounds_penalty(shape)
        penalty -= bounds_penalty * 2.0
        
        # Opportunity cost - larger shapes get larger penalties
        penalty -= shape.area * 0.01
        
        return penalty
    
    def _calculate_compactness_bonus(self, shape: PackingShape) -> float:
        """Reward for placing shapes in compact arrangements."""
        if len(self.container.placed_shapes) <= 1:
            return 0.0
        
        # Calculate how well this shape fits with existing shapes
        # (simplified - could be more sophisticated)
        adjacency_score = 0.0
        for existing_shape in self.container.placed_shapes[:-1]:  # Exclude the just-placed shape
            distance = self._shape_distance(shape, existing_shape)
            if distance < 5.0:  # Close proximity
                adjacency_score += max(0, 5.0 - distance)
        
        return adjacency_score * 0.1
    
    def _shape_distance(self, shape1: PackingShape, shape2: PackingShape) -> float:
        """Calculate distance between two shapes."""
        return math.sqrt((shape1.position[0] - shape2.position[0])**2 + 
                        (shape1.position[1] - shape2.position[1])**2)
    
    def _calculate_completion_bonus(self) -> float:
        """Bonus reward for completing the episode (placing all shapes)."""
        base_bonus = 50.0
        utilization_bonus = self.container.utilization * 100.0
        efficiency_bonus = (self.best_utilization > 0.8) * 25.0
        
        return base_bonus + utilization_bonus + efficiency_bonus
    
    def _check_episode_end(self) -> bool:
        """Check if episode should end."""
        # End if no shapes left
        if len(self.available_shapes) == 0:
            return True
        
        # End if too many steps
        if self.episode_length >= self.max_shapes * 3:
            return True
        
        # End if no progress possible (all remaining shapes too big)
        if self._no_shapes_can_fit():
            return True
        
        return False
    
    def _no_shapes_can_fit(self) -> bool:
        """Check if any remaining shapes can possibly fit."""
        free_space = self.container.get_free_space_ratio() * self.container.area
        
        # Simple heuristic: check if smallest remaining shape can fit
        if not self.available_shapes:
            return True
        
        min_area = min(shape.area for shape in self.available_shapes)
        return free_space < min_area * 0.5  # Buffer for positioning
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        obs = []
        
        # Container metrics (4 values)
        obs.extend([
            self.container.utilization,
            self.container.get_free_space_ratio(),
            self.container.area / 10000.0,  # Normalized
            len(self.container.placed_shapes) / self.max_shapes
        ])
        
        # Container occupancy grid (10x10 = 100 values)
        occupancy_grid = self._get_occupancy_grid()
        obs.extend(occupancy_grid.flatten())
        
        # Current shape info (60 values)
        if self.available_shapes:
            current_shape = self.available_shapes[0]
            obs.extend(self._encode_shape_info(current_shape))
        else:
            obs.extend([0.0] * 60)
        
        # Available shapes summary (100 values)
        shapes_summary = self._encode_shapes_summary()
        obs.extend(shapes_summary)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_occupancy_grid(self, grid_size: int = 10) -> np.ndarray:
        """Create occupancy grid representation."""
        grid = np.zeros((grid_size, grid_size))
        
        cell_width = self.container_width / grid_size
        cell_height = self.container_height / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * cell_width + cell_width / 2
                y = j * cell_height + cell_height / 2
                
                # Check if any shape occupies this cell
                point = Point(x, y)
                for shape in self.container.placed_shapes:
                    if shape.geometry.contains(point):
                        grid[i, j] = 1.0
                        break
        
        return grid
    
    def _encode_shape_info(self, shape: PackingShape) -> List[float]:
        """Encode shape information into numerical features."""
        info = []
        
        # Basic properties
        info.extend([
            shape.position[0] / self.container_width,    # Normalized x
            shape.position[1] / self.container_height,   # Normalized y
            shape.rotation / 360.0,                      # Normalized rotation
            shape.area / self.container.area,            # Normalized area
            shape.shape_def.difficulty / 3.0,            # Normalized difficulty
        ])
        
        # Bounding box
        bounds = shape.bounds
        info.extend([
            bounds[0] / self.container_width,   # min_x
            bounds[1] / self.container_height,  # min_y
            bounds[2] / self.container_width,   # max_x
            bounds[3] / self.container_height,  # max_y
        ])
        
        # Shape type encoding (one-hot-ish)
        shape_type_features = [0.0] * 6
        if isinstance(shape, RectangleShape):
            shape_type_features[0] = 1.0
        elif isinstance(shape, CircleShape):
            shape_type_features[1] = 1.0
        elif isinstance(shape, TriangleShape):
            shape_type_features[2] = 1.0
        elif isinstance(shape, LShapeShape):
            shape_type_features[3] = 1.0
        elif isinstance(shape, IrregularShape):
            shape_type_features[4] = 1.0
        else:
            shape_type_features[5] = 1.0
        
        info.extend(shape_type_features)
        
        # Pad to 60 total features
        while len(info) < 60:
            info.append(0.0)
        
        return info[:60]
    
    def _encode_shapes_summary(self) -> List[float]:
        """Encode summary of all available shapes."""
        summary = []
        
        if not self.available_shapes:
            return [0.0] * 100
        
        # Basic statistics
        areas = [s.area for s in self.available_shapes]
        summary.extend([
            len(self.available_shapes) / self.max_shapes,
            min(areas) / self.container.area if areas else 0.0,
            max(areas) / self.container.area if areas else 0.0,
            sum(areas) / self.container.area if areas else 0.0,
        ])
        
        # Shape type counts
        type_counts = [0, 0, 0, 0, 0, 0]
        for shape in self.available_shapes:
            if isinstance(shape, RectangleShape):
                type_counts[0] += 1
            elif isinstance(shape, CircleShape):
                type_counts[1] += 1
            elif isinstance(shape, TriangleShape):
                type_counts[2] += 1
            elif isinstance(shape, LShapeShape):
                type_counts[3] += 1
            elif isinstance(shape, IrregularShape):
                type_counts[4] += 1
            else:
                type_counts[5] += 1
        
        # Normalize counts
        total_shapes = len(self.available_shapes)
        summary.extend([count / total_shapes if total_shapes > 0 else 0.0 for count in type_counts])
        
        # Pad to 100 features
        while len(summary) < 100:
            summary.append(0.0)
        
        return summary[:100]
    
    def render(self, mode='human'):
        """Render the current state."""
        if mode == 'human':
            self._render_matplotlib()
    
    def _render_matplotlib(self):
        """Render using matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw container boundary
        if self.container.shape == "rectangle":
            container_patch = patches.Rectangle(
                (0, 0), self.container_width, self.container_height,
                linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
            )
            ax.add_patch(container_patch)
        
        # Draw placed shapes
        for i, shape in enumerate(self.container.placed_shapes):
            color = shape.shape_def.color
            self._add_shape_patch(ax, shape, color, alpha=0.7)
        
        # Draw current shape being placed (if any)
        if self.available_shapes:
            current_shape = self.available_shapes[0]
            self._add_shape_patch(ax, current_shape, 'red', alpha=0.3, linestyle='--')
        
        # Set axis properties
        ax.set_xlim(-5, self.container_width + 5)
        ax.set_ylim(-5, self.container_height + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add information text
        info_text = f"Placed: {len(self.container.placed_shapes)}/{self.shapes_placed + len(self.available_shapes)}\n"
        info_text += f"Utilization: {self.container.utilization:.2%}\n"
        info_text += f"Episode Reward: {self.episode_reward:.1f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(f"Container Packing - Level {self.curriculum_level}")
        plt.tight_layout()
        plt.show()
    
    def _add_shape_patch(self, ax, shape: PackingShape, color: str, alpha: float = 0.7, linestyle: str = '-'):
        """Add a shape patch to the matplotlib axes."""
        geom = shape.geometry
        
        if hasattr(geom, 'exterior'):
            # Polygon shape
            coords = list(geom.exterior.coords)
            patch = patches.Polygon(coords, facecolor=color, alpha=alpha, 
                                  edgecolor='black', linestyle=linestyle)
            ax.add_patch(patch)
        else:
            # Fallback for other geometries
            bounds = geom.bounds
            patch = patches.Rectangle((bounds[0], bounds[1]), 
                                    bounds[2] - bounds[0], bounds[3] - bounds[1],
                                    facecolor=color, alpha=alpha, 
                                    edgecolor='black', linestyle=linestyle)
            ax.add_patch(patch)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current environment metrics for logging."""
        return {
            "utilization": self.container.utilization,
            "shapes_placed": len(self.container.placed_shapes),
            "shapes_remaining": len(self.available_shapes),
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "best_utilization": self.best_utilization,
            "curriculum_level": self.curriculum_level,
            "occupied_area": self.container.occupied_area,
            "free_space_ratio": self.container.get_free_space_ratio(),
        } 