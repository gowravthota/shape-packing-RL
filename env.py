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
import os
import time

from shapes import (
    PackingShape, ShapeFactory, RectangleShape, CircleShape, 
    TriangleShape, LShape, IrregularShape
)

class Container:
    """Represents the container where shapes are packed."""
    
    def __init__(self, width: float, height: float, shape: str = "rectangle", vertices: Optional[List[Tuple[float, float]]] = None):
        self.width = width
        self.height = height
        self.shape = shape
        self.vertices = vertices
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
        elif self.shape == "arbitrary" and self.vertices is not None:
            # Arbitrary polygon container defined by vertices
            if len(self.vertices) < 3:
                raise ValueError("Arbitrary container requires at least 3 vertices")
            return Polygon(self.vertices)
        else:
            # Default to rectangle if shape type is unknown
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

class ShapeFittingEnv(gym.Env):
    """
    Shape fitting environment where agent tries to fit a group of shapes in a container.
    
    Action Space:
        - shape_id: Discrete(num_shapes_to_fit) - which shape from the group to place
        - x: Box(0, container_width) - x position
        - y: Box(0, container_height) - y position  
        - rotation: Discrete(18) - rotation in 20-degree intervals (0, 20, 40, ..., 340)
    
    Observation Space:
        - Container state representation
        - Shapes to fit information
        - Current fitting progress
    """
    
    def __init__(self, 
                 container_width: float = 100.0,
                 container_height: float = 100.0,
                 num_shapes_to_fit: int = 10,
                 difficulty_level: int = 1,
                 container_shape: str = "rectangle",
                 container_vertices: Optional[List[Tuple[float, float]]] = None,
                 max_steps: int = 50):
        
        super().__init__()
        
        # Environment parameters
        self.container_width = container_width
        self.container_height = container_height
        self.num_shapes_to_fit = num_shapes_to_fit
        self.difficulty_level = difficulty_level
        self.max_steps = max_steps
        self.container_vertices = container_vertices
        
        # Create a unique directory for this run's images
        self.image_save_path = f"metrics/images/run_{int(time.time())}"
        os.makedirs(self.image_save_path, exist_ok=True)
        
        # Rotation angles in 20-degree intervals
        self.rotation_angles = [i * 20 for i in range(18)]  # 0, 20, 40, ..., 340
        
        # Create container
        self.container = Container(container_width, container_height, container_shape, container_vertices)
        
        # Shape management
        self.shapes_to_fit: List[PackingShape] = []
        self.shapes_fitted_count = 0
        self.current_step = 0
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_reward = 0.0
        self.best_shapes_fitted = 0
        
        # Difficulty settings for shape generation
        self.difficulty_settings = {
            1: {"shape_types": ["rectangle", "circle"], "size_range": (8, 20), "complexity": 1.0},
            2: {"shape_types": ["rectangle", "circle", "triangle"], "size_range": (6, 22), "complexity": 1.5},
            3: {"shape_types": ["rectangle", "circle", "triangle", "l_shape"], "size_range": (5, 25), "complexity": 2.0},
            4: {"shape_types": ["rectangle", "circle", "triangle", "l_shape", "irregular"], "size_range": (4, 28), "complexity": 2.5},
            5: {"shape_types": ["rectangle", "circle", "triangle", "l_shape", "irregular"], "size_range": (3, 30), "complexity": 3.0},
        }
        
        self.reset()
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        
        # For arbitrary containers, use the bounding box for action space limits
        if self.container.shape == "arbitrary":
            minx, miny, maxx, maxy = self.container.geometry.bounds
            action_x_max = maxx
            action_y_max = maxy
        else:
            action_x_max = self.container_width
            action_y_max = self.container_height
        
        # Action space: [shape_id, x, y, rotation_discrete]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.num_shapes_to_fit-1, action_x_max, action_y_max, 17], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: container state + shapes to fit info + progress
        obs_size = (
            4 +  # Container metrics (utilization, fitted_count, remaining_count, step_ratio)
            100 +  # Container occupancy grid (10x10)
            self.num_shapes_to_fit * 8  # Shape info: (shape_type, width, height, area, fitted_flag, x, y, rotation)
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
        
        # Generate new group of shapes to fit
        self._generate_shape_group()
        
        # Reset episode state
        self.shapes_fitted_count = 0
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self._get_observation()
    
    def _generate_shape_group(self):
        """Generate a group of shapes that the agent needs to fit."""
        self.shapes_to_fit = []
        settings = self.difficulty_settings[self.difficulty_level]
        
        for i in range(self.num_shapes_to_fit):
            shape_type = random.choice(settings["shape_types"])
            size_min, size_max = settings["size_range"]
            
            if shape_type == "rectangle":
                width = random.uniform(size_min, size_max)
                height = random.uniform(size_min, size_max)
                shape = RectangleShape(width, height)
            elif shape_type == "circle":
                radius = random.uniform(size_min/2, size_max/2)
                shape = CircleShape(radius)
            elif shape_type == "triangle":
                base = random.uniform(size_min, size_max)
                height = random.uniform(size_min, size_max)
                shape = TriangleShape(base, height)
            elif shape_type == "l_shape":
                arm_length = random.uniform(size_min, size_max)
                arm_width = random.uniform(size_min/2, size_max/2)
                shape = LShape(arm_length, arm_width)
            elif shape_type == "irregular":
                # Create simple irregular shape
                num_vertices = random.randint(4, 6)
                vertices = []
                angle_step = 2 * math.pi / num_vertices
                radius_range = (size_min/2, size_max/2)
                for j in range(num_vertices):
                    angle = j * angle_step + random.uniform(-0.3, 0.3)
                    radius = random.uniform(*radius_range)
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    vertices.append((x, y))
                shape = IrregularShape(vertices)
            
            # Mark shape as not fitted yet
            shape.is_fitted = False
            self.shapes_to_fit.append(shape)
    
    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Parse action
        shape_id = int(np.clip(action[0], 0, len(self.shapes_to_fit) - 1))
        x = np.clip(action[1], 0, self.container_width)
        y = np.clip(action[2], 0, self.container_height)
        rotation_idx = int(np.clip(action[3], 0, len(self.rotation_angles) - 1))
        rotation = self.rotation_angles[rotation_idx]
        
        # Try to fit the selected shape
        reward, success, info = self._attempt_shape_fitting(shape_id, x, y, rotation)
        
        # Update fitted count
        if success:
            self.shapes_fitted_count += 1
            self.shapes_to_fit[shape_id].is_fitted = True
        
        # Check if episode is done
        done = self._check_episode_end()
        
        # Update tracking
        self.episode_reward += reward
        self.best_shapes_fitted = max(self.best_shapes_fitted, self.shapes_fitted_count)
        
        info.update({
            'shapes_fitted': self.shapes_fitted_count,
            'shapes_remaining': self.num_shapes_to_fit - self.shapes_fitted_count,
            'total_reward': self.episode_reward,
            'step': self.current_step,
            'success_rate': self.shapes_fitted_count / self.num_shapes_to_fit
        })
        
        # Save a screenshot of the step
        self._save_step_image()
        
        return self._get_observation(), reward, done, info
    
    def _attempt_shape_fitting(self, shape_id: int, x: float, y: float, rotation: float) -> Tuple[float, bool, Dict]:
        """Attempt to fit a shape at the specified position and rotation."""
        
        # Check if shape is already fitted
        if shape_id >= len(self.shapes_to_fit) or self.shapes_to_fit[shape_id].is_fitted:
            return -5.0, False, {'placement': 'already_fitted', 'collision': False}
        
        # Create a copy of the shape at the specified position and rotation
        shape = self.shapes_to_fit[shape_id]
        test_shape = self._create_positioned_shape(shape, x, y, rotation)
        
        # Try to place the shape
        if self.container.can_place_shape(test_shape):
            # Successful placement
            self.container.place_shape(test_shape)
            
            # Reward is primarily based on successfully fitting shapes
            base_reward = 100.0  # High reward for each successful fit
            
            # Small bonus for efficient placement (compact fitting)
            efficiency_bonus = self._calculate_efficiency_bonus(test_shape)
            
            total_reward = base_reward + efficiency_bonus
            
            return total_reward, True, {
                'placement': 'success', 
                'collision': False,
                'efficiency_bonus': efficiency_bonus
            }
        else:
            # Failed placement (collision or out of bounds)
            collision_penalty = -10.0
            
            # Additional penalty if shape is completely outside container
            bounds_penalty = self.container.get_shape_bounds_penalty(test_shape) * -5.0
            
            total_penalty = collision_penalty + bounds_penalty
            
            return total_penalty, False, {
                'placement': 'collision', 
                'collision': True,
                'bounds_penalty': bounds_penalty
            }
    
    def _create_positioned_shape(self, original_shape: PackingShape, x: float, y: float, rotation: float) -> PackingShape:
        """Create a new shape instance at the specified position and rotation."""
        if isinstance(original_shape, RectangleShape):
            new_shape = RectangleShape(original_shape.width, original_shape.height, 
                                     position=(x, y), rotation=rotation)
        elif isinstance(original_shape, CircleShape):
            new_shape = CircleShape(original_shape.radius, 
                                  position=(x, y), rotation=rotation)
        elif isinstance(original_shape, TriangleShape):
            new_shape = TriangleShape(original_shape.base_width, original_shape.height,
                                    position=(x, y), rotation=rotation)
        elif isinstance(original_shape, LShape):
            new_shape = LShape(original_shape.arm_length, original_shape.arm_width,
                                position=(x, y), rotation=rotation)
        elif isinstance(original_shape, IrregularShape):
            new_shape = IrregularShape(original_shape.vertices,
                                     position=(x, y), rotation=rotation)
        else:
            # Fallback - copy the original shape and update position/rotation
            new_shape = original_shape
            new_shape.move_to((x, y))
            new_shape.rotate_to(rotation)
        
        return new_shape
    
    def _calculate_efficiency_bonus(self, shape: PackingShape) -> float:
        """Calculate bonus for efficient space usage."""
        # Small bonus for shapes placed near other shapes (compactness)
        if len(self.container.placed_shapes) <= 1:
            return 0.0
        
        min_distance = float('inf')
        for placed_shape in self.container.placed_shapes[:-1]:  # Exclude the just-placed shape
            distance = self._shape_distance(shape, placed_shape)
            min_distance = min(min_distance, distance)
        
        # Bonus inversely proportional to distance (closer = better)
        if min_distance < float('inf'):
            compactness_bonus = max(0, 10.0 - min_distance * 0.5)
            return compactness_bonus
        
        return 0.0
    
    def _shape_distance(self, shape1: PackingShape, shape2: PackingShape) -> float:
        """Calculate distance between two shapes."""
        return shape1.geometry.distance(shape2.geometry)
    
    def _check_episode_end(self) -> bool:
        """Check if the episode should end."""
        # Episode ends if all shapes are fitted
        if self.shapes_fitted_count >= self.num_shapes_to_fit:
            return True
        
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Episode ends if no more shapes can possibly fit
        if self._no_shapes_can_fit():
            return True
        
        return False
    
    def _no_shapes_can_fit(self) -> bool:
        """Check if any remaining shapes can still fit in the container."""
        unfitted_shapes = [shape for shape in self.shapes_to_fit if not shape.is_fitted]
        
        if not unfitted_shapes:
            return True
        
        # Quick check: try a few random positions for each unfitted shape
        for shape in unfitted_shapes:
            for _ in range(5):  # Try 5 random positions
                x = random.uniform(0, self.container_width)
                y = random.uniform(0, self.container_height)
                rotation = random.choice(self.rotation_angles)
                
                test_shape = self._create_positioned_shape(shape, x, y, rotation)
                if self.container.can_place_shape(test_shape):
                    return False  # At least one shape can still fit
        
        return True  # No shapes can fit
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        obs = []
        
        # Container metrics
        obs.extend([
            self.container.utilization,
            self.shapes_fitted_count / self.num_shapes_to_fit,  # Fitting progress
            (self.num_shapes_to_fit - self.shapes_fitted_count) / self.num_shapes_to_fit,  # Remaining ratio
            self.current_step / self.max_steps  # Step progress
        ])
        
        # Container occupancy grid
        occupancy_grid = self._get_occupancy_grid()
        obs.extend(occupancy_grid.flatten())
        
        # Shapes to fit information
        for shape in self.shapes_to_fit:
            shape_info = self._encode_shape_info(shape)
            obs.extend(shape_info)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_occupancy_grid(self, grid_size: int = 10) -> np.ndarray:
        """Get a grid representation of container occupancy."""
        grid = np.zeros((grid_size, grid_size))
        
        cell_width = self.container_width / grid_size
        cell_height = self.container_height / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Check if this grid cell is occupied
                x1 = i * cell_width
                y1 = j * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                cell_box = box(x1, y1, x2, y2)
                
                for shape in self.container.placed_shapes:
                    if shape.geometry.intersects(cell_box):
                        grid[i, j] = 1.0
                        break
        
        return grid
    
    def _save_step_image(self):
        """Save a screenshot of the current environment state."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use a similar rendering logic as the interactive renderer
        self._render_matplotlib(ax=ax)
        
        # Save the figure
        filepath = os.path.join(self.image_save_path, f"step_{self.current_step:03d}.png")
        fig.savefig(filepath)
        plt.close(fig)  # Close the figure to free up memory
    
    def _encode_shape_info(self, shape: PackingShape) -> List[float]:
        """Encode shape information for observation."""
        # Shape type encoding (one-hot style)
        shape_type = 0.0
        if isinstance(shape, RectangleShape):
            shape_type = 1.0
        elif isinstance(shape, CircleShape):
            shape_type = 2.0
        elif isinstance(shape, TriangleShape):
            shape_type = 3.0
        elif isinstance(shape, LShape):
            shape_type = 4.0
        elif isinstance(shape, IrregularShape):
            shape_type = 5.0
        
        # Shape dimensions
        bounds = shape.geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area = shape.area
        
        # Fitting status
        fitted_flag = 1.0 if shape.is_fitted else 0.0
        
        # Current position and rotation (if fitted)
        if shape.is_fitted:
            x, y = shape.position
            rotation = shape.rotation
        else:
            x, y = 0.0, 0.0
            rotation = 0.0
        
        shape_features = [shape_type, width, height, area, fitted_flag, x, y, rotation]
        return shape_features
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(8, 8))
            self._render_matplotlib(ax=ax)
            if plt.get_fignums():
                plt.show(block=False)
                plt.pause(0.1)
    
    def _render_matplotlib(self, ax=None):
        """Render using Matplotlib, showing container and shapes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Determine bounds for rendering
        minx, miny, maxx, maxy = self.container.geometry.bounds
        ax.set_xlim(minx - 10, maxx + 10)
        ax.set_ylim(miny - 10, maxy + 10)
        ax.set_aspect('equal')

        # Draw container boundary
        container_patch = patches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                                          linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(container_patch)
        
        # Draw fitted shapes
        for shape in self.container.placed_shapes:
            self._add_shape_patch(ax, shape, 'green', alpha=0.7)
        
        ax.set_title(f"Step: {self.current_step}, Fitted: {self.shapes_fitted_count}/{self.num_shapes_to_fit}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        
        # Display shapes to be fit on the side
        unfitted_shapes = [s for s in self.shapes_to_fit if not s.is_fitted]
        if unfitted_shapes:
            ax.text(maxx + 15, maxy, "Shapes to Fit:", fontsize=12)
        
        for i, shape in enumerate(unfitted_shapes):
            display_shape = self._create_positioned_shape(shape, 0, 0, 0)
            s_minx, s_miny, s_maxx, s_maxy = display_shape.bounds
            
            # Position shape for display outside container
            display_x = maxx + 15 + (s_maxx - s_minx) / 2
            display_y = maxy - 10 * (i + 1) * (s_maxy - s_miny) / 2
            
            display_shape.move_to((display_x, display_y))
            self._add_shape_patch(ax, display_shape, color=shape.shape_def.color, alpha=0.5)
    
    def _add_shape_patch(self, ax, shape: PackingShape, color: str, alpha: float = 0.7):
        """Add a shape patch to the matplotlib axes."""
        if isinstance(shape, RectangleShape):
            x, y = shape.position
            patch = patches.Rectangle(
                (x - shape.width/2, y - shape.height/2),
                shape.width, shape.height,
                angle=shape.rotation,
                facecolor=color, alpha=alpha, edgecolor='black'
            )
            ax.add_patch(patch)
        elif isinstance(shape, CircleShape):
            x, y = shape.position
            patch = patches.Circle(
                (x, y), shape.radius,
                facecolor=color, alpha=alpha, edgecolor='black'
            )
            ax.add_patch(patch)
        else:
            # For other shapes, use the geometry directly
            coords = list(shape.geometry.exterior.coords)
            shape_patch = patches.Polygon(coords, edgecolor="black", facecolor=color, alpha=alpha)
            ax.add_patch(shape_patch)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current environment metrics."""
        return {
            'shapes_fitted': self.shapes_fitted_count,
            'shapes_remaining': self.num_shapes_to_fit - self.shapes_fitted_count,
            'utilization': self.container.utilization,
            'episode_reward': self.episode_reward,
            'success_rate': self.shapes_fitted_count / self.num_shapes_to_fit,
            'step': self.current_step,
            'max_steps': self.max_steps,
            'difficulty_level': self.difficulty_level
        }

    @staticmethod
    def create_hexagon_container(center_x: float, center_y: float, radius: float) -> List[Tuple[float, float]]:
        """Create vertices for a hexagonal container."""
        angles = np.linspace(0, 2*math.pi, 6, endpoint=False)
        vertices = [(center_x + radius * math.cos(a), center_y + radius * math.sin(a)) for a in angles]
        return vertices
    
    @staticmethod
    def create_octagon_container(center_x: float, center_y: float, radius: float) -> List[Tuple[float, float]]:
        """Create vertices for an octagonal container."""
        angles = np.linspace(0, 2*math.pi, 8, endpoint=False)
        vertices = [(center_x + radius * math.cos(a), center_y + radius * math.sin(a)) for a in angles]
        return vertices
    
    @staticmethod
    def create_star_container(center_x: float, center_y: float, outer_radius: float, inner_radius: float, num_points: int = 5) -> List[Tuple[float, float]]:
        """Create vertices for a star-shaped container."""
        vertices = []
        for i in range(num_points * 2):
            angle = i * math.pi / num_points
            if i % 2 == 0:  # Outer point
                radius = outer_radius
            else:  # Inner point
                radius = inner_radius
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            vertices.append((x, y))
        return vertices
    
    @staticmethod
    def create_l_shaped_container(width: float, height: float, cutout_width: float, cutout_height: float) -> List[Tuple[float, float]]:
        """Create vertices for an L-shaped container."""
        vertices = [
            (0, 0),
            (width, 0),
            (width, height - cutout_height),
            (width - cutout_width, height - cutout_height),
            (width - cutout_width, height),
            (0, height)
        ]
        return vertices

# Backward compatibility alias
ContinuousContainerEnv = ShapeFittingEnv 