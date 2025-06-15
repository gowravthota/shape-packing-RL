import math

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
    def __init__(self, x, y, width, height, color="red"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.shapes = []

    def draw_rectangle(self, canvas):
        return canvas.create_rectangle(self.x, self.y, self.x + self.width, self.y + self.height, outline=self.color)
    
    def add_shape(self, shape):
        if self.contains(shape):
            self.shapes.append(shape)
            return True
        return False

    def contains(self, shape):
        left, top, right, bottom = shape.bounding_box()
        return left >= self.x and right <= self.x + self.width and top >= self.y and bottom <= self.y + self.heights