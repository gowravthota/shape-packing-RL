

import numpy as np
import math
from typing import List, Tuple, Optional, Union
from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate, translate
from dataclasses import dataclass
import random

@dataclass
class ShapeDefinition:
    """Definition of a shape type with its properties."""
    name: str
    base_area: float
    difficulty: float  # 1.0 = easy, 2.0 = medium, 3.0 = hard
    value: float  # Points awarded for successful placement
    color: str

class PackingShape:
    """A geometric shape that can be placed and rotated in 2D space."""
    
    def __init__(self, shape_def: ShapeDefinition, position: Tuple[float, float] = (0, 0), 
                 rotation: float = 0, scale: float = 1.0):
        self.shape_def = shape_def
        self.position = position
        self.rotation = rotation  # In degrees
        self.scale = scale
        self._geometry = None
        self._update_geometry()
    
    def _create_base_geometry(self) -> Polygon:
        """Create the base geometry for this shape type. Override in subclasses."""
        raise NotImplementedError
    
    def _update_geometry(self):
        """Update the shapely geometry based on current position, rotation, and scale."""
        base_geom = self._create_base_geometry()
        
        # Apply scaling
        if self.scale != 1.0:
            base_geom = base_geom.buffer(0)  # Ensure valid geometry
            centroid = base_geom.centroid
            base_geom = translate(base_geom, -centroid.x, -centroid.y)
            # Manual scaling by transforming coordinates
            coords = list(base_geom.exterior.coords)
            scaled_coords = [(x * self.scale, y * self.scale) for x, y in coords]
            base_geom = Polygon(scaled_coords)
            base_geom = translate(base_geom, centroid.x, centroid.y)
        
        # Apply rotation
        if self.rotation != 0:
            base_geom = rotate(base_geom, self.rotation, origin='centroid')
        
        # Apply translation
        self._geometry = translate(base_geom, self.position[0], self.position[1])
    
    def move_to(self, position: Tuple[float, float]):
        """Move shape to new position."""
        self.position = position
        self._update_geometry()
    
    def rotate_to(self, rotation: float):
        """Rotate shape to new angle (degrees)."""
        self.rotation = rotation % 360
        self._update_geometry()
    
    def set_scale(self, scale: float):
        """Set shape scale."""
        self.scale = scale
        self._update_geometry()
    
    @property
    def geometry(self) -> Polygon:
        """Get the current shapely geometry."""
        return self._geometry
    
    @property
    def area(self) -> float:
        """Get the actual area of the shape."""
        return self._geometry.area
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (minx, miny, maxx, maxy)."""
        return self._geometry.bounds
    
    def intersects(self, other: 'PackingShape') -> bool:
        """Check if this shape intersects with another."""
        return self._geometry.intersects(other.geometry)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if shape contains a point."""
        return self._geometry.contains(Point(point))

class RectangleShape(PackingShape):
    """Rectangular shape with customizable width and height."""
    
    def __init__(self, width: float, height: float, **kwargs):
        self.width = width
        self.height = height
        # Create shape definition
        shape_def = ShapeDefinition(
            name=f"Rectangle_{width}x{height}",
            base_area=width * height,
            difficulty=1.0,
            value=width * height * 10,
            color="blue"
        )
        super().__init__(shape_def, **kwargs)
    
    def _create_base_geometry(self) -> Polygon:
        """Create rectangular geometry centered at origin."""
        return box(-self.width/2, -self.height/2, self.width/2, self.height/2)

class CircleShape(PackingShape):
    """Circular shape."""
    
    def __init__(self, radius: float, **kwargs):
        self.radius = radius
        shape_def = ShapeDefinition(
            name=f"Circle_r{radius}",
            base_area=math.pi * radius * radius,
            difficulty=1.5,  # Harder to pack efficiently
            value=math.pi * radius * radius * 12,
            color="green"
        )
        super().__init__(shape_def, **kwargs)
    
    def _create_base_geometry(self) -> Polygon:
        """Create circular geometry as a polygon approximation."""
        # Create circle as 32-sided polygon for good approximation
        angles = np.linspace(0, 2*math.pi, 32, endpoint=False)
        points = [(self.radius * math.cos(a), self.radius * math.sin(a)) for a in angles]
        return Polygon(points)

class TriangleShape(PackingShape):
    """Triangular shape."""
    
    def __init__(self, base_width: float, height: float, **kwargs):
        self.base_width = base_width
        self.height = height
        shape_def = ShapeDefinition(
            name=f"Triangle_{base_width}x{height}",
            base_area=0.5 * base_width * height,
            difficulty=2.0,  # Triangles are harder to pack
            value=0.5 * base_width * height * 15,
            color="red"
        )
        super().__init__(shape_def, **kwargs)
    
    def _create_base_geometry(self) -> Polygon:
        """Create triangular geometry centered at origin."""
        points = [
            (-self.base_width/2, -self.height/3),  # Bottom left
            (self.base_width/2, -self.height/3),   # Bottom right
            (0, 2*self.height/3)                   # Top center
        ]
        return Polygon(points)

class LShapeShape(PackingShape):
    """L-shaped piece (like Tetris)."""
    
    def __init__(self, arm_length: float, arm_width: float, **kwargs):
        self.arm_length = arm_length
        self.arm_width = arm_width
        shape_def = ShapeDefinition(
            name=f"LShape_{arm_length}x{arm_width}",
            base_area=3 * arm_width * arm_width,  # Approximate
            difficulty=2.5,
            value=3 * arm_width * arm_width * 18,
            color="purple"
        )
        super().__init__(shape_def, **kwargs)
    
    def _create_base_geometry(self) -> Polygon:
        """Create L-shaped geometry."""
        w = self.arm_width
        l = self.arm_length
        points = [
            (-w/2, -w/2),     # Bottom left of base
            (l-w/2, -w/2),    # Bottom right of horizontal arm
            (l-w/2, w/2),     # Top right of horizontal arm
            (w/2, w/2),       # Connection point
            (w/2, l-w/2),     # Top right of vertical arm
            (-w/2, l-w/2),    # Top left of vertical arm
        ]
        return Polygon(points)

class IrregularShape(PackingShape):
    """Irregular polygon shape."""
    
    def __init__(self, vertices: List[Tuple[float, float]], **kwargs):
        self.vertices = vertices
        # Calculate area using shoelace formula
        area = 0.5 * abs(sum(x1*y2 - x2*y1 for (x1, y1), (x2, y2) in 
                            zip(vertices, vertices[1:] + [vertices[0]])))
        
        shape_def = ShapeDefinition(
            name=f"Irregular_{len(vertices)}pts",
            base_area=area,
            difficulty=3.0,  # Very hard to pack
            value=area * 20,
            color="orange"
        )
        super().__init__(shape_def, **kwargs)
    
    def _create_base_geometry(self) -> Polygon:
        """Create irregular geometry from vertices."""
        return Polygon(self.vertices)

class ShapeFactory:
    """Factory for creating various shapes with different difficulty levels."""
    
    @staticmethod
    def create_basic_shapes() -> List[PackingShape]:
        """Create a set of basic shapes for training."""
        shapes = []
        
        # Small rectangles
        shapes.extend([
            RectangleShape(8, 6),
            RectangleShape(10, 4),
            RectangleShape(6, 8),
        ])
        
        # Medium rectangles
        shapes.extend([
            RectangleShape(15, 10),
            RectangleShape(12, 12),
            RectangleShape(18, 8),
        ])
        
        # Circles
        shapes.extend([
            CircleShape(5),
            CircleShape(8),
            CircleShape(12),
        ])
        
        # Triangles
        shapes.extend([
            TriangleShape(12, 10),
            TriangleShape(16, 8),
            TriangleShape(10, 14),
        ])
        
        return shapes
    
    @staticmethod
    def create_tetris_shapes() -> List[PackingShape]:
        """Create Tetris-like shapes for advanced training."""
        shapes = []
        
        # L-shapes of different sizes
        shapes.extend([
            LShapeShape(12, 4),
            LShapeShape(16, 5),
            LShapeShape(10, 3),
        ])
        
        # T-shapes (using irregular)
        t_shape_vertices = [(-6, -2), (6, -2), (6, 2), (2, 2), (2, 6), (-2, 6), (-2, 2), (-6, 2)]
        shapes.append(IrregularShape(t_shape_vertices))
        
        # Z-shapes
        z_shape_vertices = [(-4, -2), (0, -2), (0, 0), (4, 0), (4, 2), (0, 2), (0, 4), (-4, 4)]
        shapes.append(IrregularShape(z_shape_vertices))
        
        return shapes
    
    @staticmethod
    def create_random_shape(difficulty: float = 1.0) -> PackingShape:
        """Create a random shape based on difficulty level."""
        
        if difficulty < 1.5:
            # Easy: mostly rectangles and circles
            if random.random() < 0.7:
                w = random.uniform(5, 20)
                h = random.uniform(5, 20)
                return RectangleShape(w, h)
            else:
                r = random.uniform(3, 12)
                return CircleShape(r)
        
        elif difficulty < 2.5:
            # Medium: add triangles
            shape_type = random.choice(['rect', 'circle', 'triangle'])
            if shape_type == 'rect':
                w = random.uniform(4, 25)
                h = random.uniform(4, 25)
                return RectangleShape(w, h)
            elif shape_type == 'circle':
                r = random.uniform(3, 15)
                return CircleShape(r)
            else:
                base = random.uniform(6, 20)
                height = random.uniform(6, 20)
                return TriangleShape(base, height)
        
        else:
            # Hard: include complex shapes
            shape_type = random.choice(['rect', 'circle', 'triangle', 'lshape', 'irregular'])
            if shape_type == 'lshape':
                arm_length = random.uniform(8, 20)
                arm_width = random.uniform(3, 8)
                return LShapeShape(arm_length, arm_width)
            elif shape_type == 'irregular':
                # Create random polygon
                num_vertices = random.randint(5, 8)
                radius = random.uniform(5, 15)
                vertices = []
                for i in range(num_vertices):
                    angle = 2 * math.pi * i / num_vertices
                    # Add some randomness to make it irregular
                    r = radius * random.uniform(0.6, 1.0)
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    vertices.append((x, y))
                return IrregularShape(vertices)
            else:
                # Fall back to basic shapes
                return ShapeFactory.create_random_shape(difficulty - 1.0)
    
    @staticmethod
    def create_curriculum_batch(level: int, batch_size: int = 5) -> List[PackingShape]:
        """Create a batch of shapes for curriculum learning."""
        difficulty = min(3.0, 0.5 + level * 0.3)  # Progressive difficulty
        return [ShapeFactory.create_random_shape(difficulty) for _ in range(batch_size)]

# Predefined challenging shape sets
TETRIS_CHALLENGE = ShapeFactory.create_tetris_shapes()
BASIC_CHALLENGE = ShapeFactory.create_basic_shapes()
MIXED_CHALLENGE = BASIC_CHALLENGE + TETRIS_CHALLENGE 