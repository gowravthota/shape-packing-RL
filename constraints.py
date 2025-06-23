import math
import numpy as np

class Shape:
    def __init__(self, x, y, orientation=0, size=10, color="blue"):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.size = size
        self.color = color

    def _rotate(self, dx, dy):
        rad = math.radians(self.orientation)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return self.x + rx, self.y + ry

    def create_rectangle(self, canvas):
        h = self.size / 2
        pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
        coords = []
        for px, py in pts:
            coords.extend(self._rotate(px, py))
        return canvas.create_polygon(coords, fill=self.color)

    def create_circle(self, canvas):
        r = self.size / 2
        return canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill=self.color)

    def create_triangle(self, canvas):
        r = self.size / 2
        pts = [(0, -r), (r * math.cos(math.radians(210)), r * math.sin(math.radians(210))), (r * math.cos(math.radians(330)), r * math.sin(math.radians(330)))]
        coords = []
        for px, py in pts:
            coords.extend(self._rotate(px, py))
        return canvas.create_polygon(coords, fill=self.color)

    def create_polygon(self, canvas):
        r = self.size / 2
        coords = []
        for i in range(5):
            angle = math.radians(360 / 5 * i)
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            coords.extend(self._rotate(px, py))
        return canvas.create_polygon(coords, fill=self.color)

    def bounding_box(self):
        r = self.size / 2
        return self.x - r, self.y - r, self.x + r, self.y + r

class Container:
    def __init__(self, x, y, width, height, color="red", mask=None, polygon=None, obstacles=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.shapes = []
        self.mask = mask  # 2D numpy array mask (1=valid, 0=invalid)
        self.polygon = polygon  # List of (x, y) tuples for irregular shape
        self.obstacles = obstacles if obstacles is not None else []  # List of polygons or mask

    def draw_rectangle(self, canvas):
        return canvas.create_rectangle(self.x, self.y, self.x + self.width, self.y + self.height, outline=self.color)
    
    def add_shape(self, shape):
        if self.contains(shape):
            self.shapes.append(shape)
            return True
        return False

    def contains(self, shape):
        # Check bounding box first
        left, top, right, bottom = shape.bounding_box()
        if left < self.x or right > self.x + self.width or top < self.y or bottom > self.y + self.height:
            return False
        # If mask is provided, check all points in the shape are valid
        if self.mask is not None:
            cx, cy = int(shape.x - self.x), int(shape.y - self.y)
            if cx < 0 or cy < 0 or cx >= self.mask.shape[1] or cy >= self.mask.shape[0]:
                return False
            if self.mask[cy, cx] == 0:
                return False
        # If polygon is provided, check center is inside polygon
        if self.polygon is not None:
            if not point_in_polygon((shape.x, shape.y), self.polygon):
                return False
        # Check obstacles (list of polygons)
        for obs in self.obstacles:
            if point_in_polygon((shape.x, shape.y), obs):
                return False
        return True

def point_in_polygon(point, polygon):
    # Ray casting algorithm for point-in-polygon
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside