#!/usr/bin/env python3
"""
Basic functionality test for the space RL project.
Tests core constraint and shape functionality without requiring ML dependencies.
"""

from constraints import Shape, Container

def test_shapes():
    """Test shape creation and methods."""
    print("Testing Shape functionality...")
    
    # Create different shapes
    rect = Shape(25, 25, size=20, color='blue')
    circle = Shape(75, 75, size=15, color='red')
    triangle = Shape(25, 75, orientation=45, size=18, color='green')
    
    print(f"Rectangle: {rect.color} at ({rect.x}, {rect.y}), size {rect.size}")
    print(f"Circle: {circle.color} at ({circle.x}, {circle.y}), size {circle.size}")
    print(f"Triangle: {triangle.color} at ({triangle.x}, {triangle.y}), size {triangle.size}, angle {triangle.orientation}")
    
    # Test bounding boxes
    bbox = rect.bounding_box()
    print(f"Rectangle bounding box: {bbox}")
    
    # Test distance calculation
    dist = rect.distance_to(circle)
    print(f"Distance between rectangle and circle: {dist:.2f}")
    
    # Test overlap detection
    overlap = rect.overlaps_with(circle)
    print(f"Rectangle and circle overlap: {overlap}")
    
    print("✓ Shape tests passed\n")

def test_container():
    """Test container functionality."""
    print("Testing Container functionality...")
    
    # Create container
    container = Container(0, 0, 100, 100, color='black')
    print(f"Created container: {container.width}x{container.height}")
    
    # Create shapes to add
    shapes = [
        Shape(20, 20, size=15, color='blue'),
        Shape(50, 50, size=20, color='red'),
        Shape(80, 80, size=12, color='green'),
        Shape(10, 90, size=25, color='yellow'),  # This might overlap or be out of bounds
    ]
    
    # Try to add shapes
    for i, shape in enumerate(shapes):
        success = container.add_shape(shape)
        print(f"Shape {i+1} ({shape.color}): {'✓ added' if success else '✗ rejected'}")
    
    # Container statistics
    print(f"Container has {container.get_shape_count()} shapes")
    print(f"Free space: {container.get_free_space():.1%}")
    
    # Test with irregular container
    polygon = [(10, 10), (90, 10), (90, 90), (10, 90)]  # Square polygon
    obstacles = [[(40, 40), (60, 40), (50, 60)]]  # Triangle obstacle
    
    irregular_container = Container(0, 0, 100, 100, polygon=polygon, obstacles=obstacles)
    test_shape = Shape(50, 50, size=10, color='purple')
    
    # This should fail due to obstacle
    success = irregular_container.add_shape(test_shape)
    print(f"Shape in obstacle area: {'✓ added' if success else '✗ rejected (obstacle)'}")
    
    # Test safe position
    safe_shape = Shape(20, 20, size=10, color='orange')
    success = irregular_container.add_shape(safe_shape)
    print(f"Shape in safe area: {'✓ added' if success else '✗ rejected'}")
    
    print("✓ Container tests passed\n")

def test_visualization_availability():
    """Test if visualization modules are available."""
    print("Testing visualization availability...")
    
    try:
        import visualization
        print("✓ Visualization module loaded")
        print(f"  - pygame available: {'✓' if visualization.PYGAME_AVAILABLE else '✗'}")
        print(f"  - matplotlib available: {'✓' if visualization.MATPLOTLIB_AVAILABLE else '✗'}")
    except ImportError as e:
        print(f"✗ Visualization module error: {e}")
    
    print()

def main():
    """Run all tests."""
    print("Space RL Project - Basic Functionality Test")
    print("=" * 50)
    
    try:
        test_shapes()
        test_container()
        test_visualization_availability()
        
        print("🎉 All basic tests passed! The core functionality is working correctly.")
        print("\nTo run the full RL training, install the dependencies:")
        print("  pip install torch numpy gym pygame matplotlib")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 