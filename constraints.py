import math
import numpy as np
from typing import List, Tuple, Optional, Union

class Shape:
    """Represents a geometric shape with position, orientation, size, and color."""
    
    def __init__(self, x: float, y: float, orientation: float = 0, size: float = 10, 
                 color: str = "blue"):
        self.x = float(x)
        self.y = float(y)
        self.orientation = float(orientation)
        self.size = float(max(1, size))  # Ensure positive size
        self.color = str(color)

    def _rotate(self, dx: float, dy: float) -> Tuple[float, float]:
        """Rotate a point (dx, dy) around the shape's center by its orientation."""
        rad = math.radians(self.orientation)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return self.x + rx, self.y + ry

    def get_vertices(self, shape_type: str = "rectangle") -> List[Tuple[float, float]]:
        """Get the vertices of the shape based on its type."""
        if shape_type == "rectangle":
            h = self.size / 2
            pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
            return [self._rotate(px, py) for px, py in pts]
        
        elif shape_type == "triangle":
            r = self.size / 2
            pts = [
                (0, -r),
                (r * math.cos(math.radians(210)), r * math.sin(math.radians(210))),
                (r * math.cos(math.radians(330)), r * math.sin(math.radians(330)))
            ]
            return [self._rotate(px, py) for px, py in pts]
        
        elif shape_type == "polygon":
            r = self.size / 2
            pts = []
            for i in range(5):  # Pentagon
                angle = math.radians(360 / 5 * i)
                px = r * math.cos(angle)
                py = r * math.sin(angle)
                pts.append((px, py))
            return [self._rotate(px, py) for px, py in pts]
        
        elif shape_type == "circle":
            # For circles, return the center and radius
            return [(self.x, self.y, self.size / 2)]
        
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the axis-aligned bounding box (left, top, right, bottom)."""
        r = self.size / 2
        # Simple bounding box - could be improved for rotated shapes
        left = self.x - r
        top = self.y - r
        right = self.x + r
        bottom = self.y + r
        return left, top, right, bottom

    def overlaps_with(self, other: 'Shape') -> bool:
        """Check if this shape overlaps with another shape using bounding boxes."""
        left1, top1, right1, bottom1 = self.bounding_box()
        left2, top2, right2, bottom2 = other.bounding_box()
        
        return not (right1 <= left2 or left1 >= right2 or bottom1 <= top2 or top1 >= bottom2)

    def distance_to(self, other: 'Shape') -> float:
        """Calculate Euclidean distance to another shape's center."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Container:
    """Represents a container that can hold shapes with various constraints."""
    
    def __init__(self, x: float, y: float, width: float, height: float, 
                 color: str = "red", mask: Optional[np.ndarray] = None, 
                 polygon: Optional[List[Tuple[float, float]]] = None, 
                 obstacles: Optional[List[List[Tuple[float, float]]]] = None):
        self.x = float(x)
        self.y = float(y)
        self.width = float(max(1, width))  # Ensure positive dimensions
        self.height = float(max(1, height))
        self.color = str(color)
        self.shapes: List[Shape] = []
        self.mask = mask  # 2D numpy array mask (1=valid, 0=invalid)
        self.polygon = polygon  # List of (x, y) tuples for irregular shape
        self.obstacles = obstacles if obstacles is not None else []  # List of polygons

    def add_shape(self, shape: Shape) -> bool:
        """Add a shape to the container if it fits within constraints."""
        try:
            if self.contains(shape) and not self._overlaps_existing(shape):
                self.shapes.append(shape)
                return True
            return False
        except Exception as e:
            print(f"Error adding shape: {e}")
            return False

    def remove_shape(self, shape: Shape) -> bool:
        """Remove a shape from the container."""
        try:
            if shape in self.shapes:
                self.shapes.remove(shape)
                return True
            return False
        except Exception as e:
            print(f"Error removing shape: {e}")
            return False

    def contains(self, shape: Shape) -> bool:
        """Check if a shape is within the container's boundaries and constraints."""
        try:
            # Check basic bounding box first
            left, top, right, bottom = shape.bounding_box()
            if (left < self.x or right > self.x + self.width or 
                top < self.y or bottom > self.y + self.height):
                return False
            
            # If mask is provided, check all points in the shape are valid
            if self.mask is not None:
                cx, cy = int(shape.x - self.x), int(shape.y - self.y)
                if (cx < 0 or cy < 0 or 
                    cx >= self.mask.shape[1] or cy >= self.mask.shape[0]):
                    return False
                if self.mask[cy, cx] == 0:
                    return False
            
            # If polygon is provided, check center is inside polygon
            if self.polygon is not None:
                if not point_in_polygon((shape.x, shape.y), self.polygon):
                    return False
            
            # Check obstacles (list of polygons)
            for obstacle in self.obstacles:
                if point_in_polygon((shape.x, shape.y), obstacle):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error in contains check: {e}")
            return False

    def _overlaps_existing(self, new_shape: Shape) -> bool:
        """Check if a new shape overlaps with any existing shapes."""
        try:
            for existing_shape in self.shapes:
                if new_shape.overlaps_with(existing_shape):
                    return True
            return False
        except Exception as e:
            print(f"Error checking overlaps: {e}")
            return True  # Conservative approach - assume overlap if error

    def get_free_space(self) -> float:
        """Calculate approximate free space percentage."""
        try:
            total_area = self.width * self.height
            occupied_area = sum(shape.size ** 2 for shape in self.shapes)
            return max(0, (total_area - occupied_area) / total_area)
        except Exception as e:
            print(f"Error calculating free space: {e}")
            return 0.0

    def get_shape_count(self) -> int:
        """Get the number of shapes in the container."""
        return len(self.shapes)

    def clear(self) -> None:
        """Remove all shapes from the container."""
        self.shapes.clear()

def point_in_polygon(point: Tuple[float, float], 
                     polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm for point-in-polygon test.
    Returns True if point is inside the polygon.
    """
    if not polygon or len(polygon) < 3:
        return False
    
    try:
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
        
    except Exception as e:
        print(f"Error in point-in-polygon test: {e}")
        return False

def shapes_intersect(shape1: Shape, shape2: Shape) -> bool:
    """Check if two shapes intersect (more accurate than bounding boxes)."""
    try:
        # For now, use distance-based collision detection
        distance = shape1.distance_to(shape2)
        min_distance = (shape1.size + shape2.size) / 2
        return distance < min_distance
    except Exception as e:
        print(f"Error checking shape intersection: {e}")
        return True  # Conservative approach