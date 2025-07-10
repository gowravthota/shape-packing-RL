import unittest
from shapes import RectangleShape, CircleShape, TriangleShape, LShapeShape, IrregularShape
from env import Container

class TestShapeGeometry(unittest.TestCase):
    def test_positive_area_and_valid_geometry(self):
        shapes = [
            RectangleShape(10, 5),
            CircleShape(7),
            TriangleShape(8, 5),
            LShapeShape(10, 3),
            IrregularShape([(0, 0), (3, 0), (4, 2), (1, 3), (-1, 1)])
        ]
        for shape in shapes:
            with self.subTest(shape=shape.shape_def.name):
                self.assertGreater(shape.area, 0.0)
                self.assertTrue(shape.geometry.is_valid)

class TestContainerPlacement(unittest.TestCase):
    def test_can_place_shape(self):
        container = Container(100, 100)
        shape1 = RectangleShape(10, 10, position=(10, 10))
        self.assertTrue(container.can_place_shape(shape1))
        container.place_shape(shape1)

        # Overlapping with shape1 should be rejected
        shape2 = RectangleShape(10, 10, position=(15, 10))
        self.assertFalse(container.can_place_shape(shape2))

        # Touching or outside container boundary should be rejected
        shape3 = RectangleShape(10, 10, position=(99, 99))
        self.assertFalse(container.can_place_shape(shape3))

        # Free space inside container should be allowed
        shape4 = RectangleShape(10, 10, position=(30, 30))
        self.assertTrue(container.can_place_shape(shape4))

if __name__ == '__main__':
    unittest.main()
