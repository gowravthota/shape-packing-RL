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
        self.is_fitted = False  # Track if shape has been successfully fitted
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
    
    def mark_fitted(self):
        """Mark this shape as successfully fitted."""
        self.is_fitted = True
    
    def mark_unfitted(self):
        """Mark this shape as not fitted."""
        self.is_fitted = False
    
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
            name=f"Rectangle_{width:.1f}x{height:.1f}",
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
            name=f"Circle_r{radius:.1f}",
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
            name=f"Triangle_{base_width:.1f}x{height:.1f}",
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
            name=f"LShape_{arm_length:.1f}x{arm_width:.1f}",
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
            difficulty=3.0,  # Most complex
            value=area * 20,
            color="orange"
        )
        super().__init__(shape_def, **kwargs)
    
    def _create_base_geometry(self) -> Polygon:
        """Create irregular polygon geometry."""
        return Polygon(self.vertices)

class ShapeFactory:
    """Factory for creating different types of shapes."""
    
    @staticmethod
    def create_basic_shapes() -> List[PackingShape]:
        """Create a set of basic shapes for testing."""
        shapes = []
        
        # Rectangles
        shapes.extend([
            RectangleShape(15, 10),
            RectangleShape(20, 8),
            RectangleShape(12, 12),
            RectangleShape(25, 6),
            RectangleShape(8, 18),
        ])
        
        # Circles
        shapes.extend([
            CircleShape(8),
            CircleShape(6),
            CircleShape(10),
            CircleShape(5),
        ])
        
        # Triangles
        shapes.extend([
            TriangleShape(15, 12),
            TriangleShape(20, 8),
            TriangleShape(10, 15),
        ])
        
        return shapes
    
    @staticmethod
    def create_tetris_shapes() -> List[PackingShape]:
        """Create Tetris-like shapes."""
        shapes = []
        
        # L-shapes with different orientations
        shapes.extend([
            LShapeShape(15, 5),
            LShapeShape(20, 6),
            LShapeShape(12, 4),
        ])
        
        # Create some irregular Tetris-like pieces
        t_piece = IrregularShape([(-5, -5), (5, -5), (5, 0), (2.5, 0), (2.5, 10), (-2.5, 10), (-2.5, 0), (-5, 0)])
        shapes.append(t_piece)
        
        z_piece = IrregularShape([(-7.5, -2.5), (-2.5, -2.5), (-2.5, 2.5), (2.5, 2.5), (2.5, 7.5), (7.5, 7.5), (7.5, 2.5), (-2.5, 2.5)])
        shapes.append(z_piece)
        
        return shapes
    
    @staticmethod
    def create_random_shape(difficulty: float = 1.0) -> PackingShape:
        """Create a random shape based on difficulty level."""
        shape_types = []
        size_range = (5, 25)
        
        # Adjust shape types and sizes based on difficulty
        if difficulty <= 1.0:
            shape_types = ["rectangle", "circle"]
            size_range = (8, 20)
        elif difficulty <= 2.0:
            shape_types = ["rectangle", "circle", "triangle"]
            size_range = (6, 22)
        elif difficulty <= 2.5:
            shape_types = ["rectangle", "circle", "triangle", "l_shape"]
            size_range = (5, 25)
        else:
            shape_types = ["rectangle", "circle", "triangle", "l_shape", "irregular"]
            size_range = (4, 28)
        
        shape_type = random.choice(shape_types)
        
        if shape_type == "rectangle":
            width = random.uniform(*size_range)
            height = random.uniform(*size_range)
            return RectangleShape(width, height)
        
        elif shape_type == "circle":
            radius = random.uniform(size_range[0]/2, size_range[1]/2)
            return CircleShape(radius)
        
        elif shape_type == "triangle":
            base = random.uniform(*size_range)
            height = random.uniform(*size_range)
            return TriangleShape(base, height)
        
        elif shape_type == "l_shape":
            arm_length = random.uniform(*size_range)
            arm_width = random.uniform(size_range[0]/2, size_range[1]/2)
            return LShapeShape(arm_length, arm_width)
        
        elif shape_type == "irregular":
            # Create random irregular polygon
            num_vertices = random.randint(4, 8)
            vertices = []
            
            # Generate vertices in a rough circle to avoid self-intersecting polygons
            angle_step = 2 * math.pi / num_vertices
            radius_range = (size_range[0]/2, size_range[1]/2)
            
            for i in range(num_vertices):
                angle = i * angle_step + random.uniform(-0.3, 0.3)
                radius = random.uniform(*radius_range)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                vertices.append((x, y))
            
            return IrregularShape(vertices)
        
        # Fallback to rectangle
        return RectangleShape(random.uniform(*size_range), random.uniform(*size_range))
    
    @staticmethod
    def create_curriculum_batch(level: int, batch_size: int = 10) -> List[PackingShape]:
        """Create a batch of shapes for curriculum learning."""
        shapes = []
        
        # Define difficulty progression
        difficulty_map = {
            1: 1.0,
            2: 1.5,
            3: 2.0,
            4: 2.5,
            5: 3.0
        }
        
        difficulty = difficulty_map.get(level, 3.0)
        
        for _ in range(batch_size):
            shape = ShapeFactory.create_random_shape(difficulty)
            shapes.append(shape)
        
        return shapes
    
    @staticmethod
    def create_standard_set(num_shapes: int = 10, difficulty: int = 1) -> List[PackingShape]:
        """Create a standard set of shapes for training/testing."""
        shapes = []
        
        # Define shape distribution based on difficulty
        if difficulty == 1:
            # Easy: mostly rectangles and circles
            for i in range(num_shapes):
                if i % 2 == 0:
                    shapes.append(RectangleShape(
                        random.uniform(8, 20), 
                        random.uniform(8, 20)
                    ))
                else:
                    shapes.append(CircleShape(random.uniform(4, 10)))
        
        elif difficulty == 2:
            # Medium: add triangles
            for i in range(num_shapes):
                shape_type = i % 3
                if shape_type == 0:
                    shapes.append(RectangleShape(
                        random.uniform(6, 22),
                        random.uniform(6, 22)
                    ))
                elif shape_type == 1:
                    shapes.append(CircleShape(random.uniform(3, 11)))
                else:
                    shapes.append(TriangleShape(
                        random.uniform(8, 20),
                        random.uniform(8, 20)
                    ))
        
        elif difficulty == 3:
            # Hard: add L-shapes
            for i in range(num_shapes):
                shape_type = i % 4
                if shape_type == 0:
                    shapes.append(RectangleShape(
                        random.uniform(5, 25),
                        random.uniform(5, 25)
                    ))
                elif shape_type == 1:
                    shapes.append(CircleShape(random.uniform(3, 12)))
                elif shape_type == 2:
                    shapes.append(TriangleShape(
                        random.uniform(6, 22),
                        random.uniform(6, 22)
                    ))
                else:
                    shapes.append(LShapeShape(
                        random.uniform(10, 20),
                        random.uniform(3, 8)
                    ))
        
        elif difficulty == 4:
            # Very hard: add irregular shapes
            for i in range(num_shapes):
                shape_type = i % 5
                if shape_type == 0:
                    shapes.append(RectangleShape(
                        random.uniform(4, 28),
                        random.uniform(4, 28)
                    ))
                elif shape_type == 1:
                    shapes.append(CircleShape(random.uniform(2, 14)))
                elif shape_type == 2:
                    shapes.append(TriangleShape(
                        random.uniform(5, 25),
                        random.uniform(5, 25)
                    ))
                elif shape_type == 3:
                    shapes.append(LShapeShape(
                        random.uniform(8, 22),
                        random.uniform(3, 10)
                    ))
                else:
                    # Irregular shape
                    num_vertices = random.randint(4, 6)
                    vertices = []
                    angle_step = 2 * math.pi / num_vertices
                    for j in range(num_vertices):
                        angle = j * angle_step + random.uniform(-0.3, 0.3)
                        radius = random.uniform(5, 12)
                        x = radius * math.cos(angle)
                        y = radius * math.sin(angle)
                        vertices.append((x, y))
                    shapes.append(IrregularShape(vertices))
        
        else:  # difficulty >= 5
            # Expert: maximum variety and challenge
            shapes = ShapeFactory.create_curriculum_batch(5, num_shapes)
        
        return shapes

# Predefined challenging shape sets
TETRIS_CHALLENGE = ShapeFactory.create_tetris_shapes()
BASIC_CHALLENGE = ShapeFactory.create_basic_shapes()
MIXED_CHALLENGE = BASIC_CHALLENGE + TETRIS_CHALLENGE 