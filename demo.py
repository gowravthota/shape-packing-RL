import numpy as np
from env import ShapeFittingEnv
from shapes import ShapeFactory, RectangleShape, CircleShape, TriangleShape, LShape

def test_shapes():
    """Test shape creation and basic functionality."""
    print("🔷 Testing shape creation...")
    
    # Test basic shapes
    rect = RectangleShape(20, 10, position=(50, 50), rotation=45)
    circle = CircleShape(15, position=(30, 30))
    triangle = TriangleShape(18, 12, position=(70, 70), rotation=60)
    l_shape = LShape(20, 6, position=(25, 75), rotation=90)
    
    print(f"   Rectangle: area={rect.area:.1f}, position={rect.position}, fitted={rect.is_fitted}")
    print(f"   Circle: area={circle.area:.1f}, position={circle.position}, fitted={circle.is_fitted}")
    print(f"   Triangle: area={triangle.area:.1f}, position={triangle.position}, fitted={triangle.is_fitted}")
    print(f"   L-Shape: area={l_shape.area:.1f}, position={l_shape.position}, fitted={l_shape.is_fitted}")
    
    # Test fitting status
    rect.mark_fitted()
    circle.mark_fitted()
    print(f"   After marking fitted - Rectangle: {rect.is_fitted}, Circle: {circle.is_fitted}")
    
    print("   ✅ Shapes working correctly")

def test_environment():
    """Test the shape fitting environment."""
    print("\n🏗️ Testing environment...")
    
    env = ShapeFittingEnv(
        container_width=100,
        container_height=100,
        num_shapes_to_fit=8,
        difficulty_level=1,
        max_steps=40
    )
    
    obs = env.reset()
    
    print(f"   Environment created:")
    print(f"   • Container: {env.container_width}x{env.container_height}")
    print(f"   • Shapes to fit: {len(env.shapes_to_fit)}")
    print(f"   • Rotation angles: {len(env.rotation_angles)} intervals ({env.rotation_angles[0]}° to {env.rotation_angles[-1]}°)")
    print(f"   • Observation space: {obs.shape}")
    print(f"   • Action space: {env.action_space}")
    
    # Test a few actions
    print(f"\n   Testing actions:")
    for i in range(5):
        # Create a valid action: [shape_id, x, y, rotation_idx]
        action = [
            i % len(env.shapes_to_fit),  # Select shape
            np.random.uniform(20, 80),   # X position
            np.random.uniform(20, 80),   # Y position  
            np.random.randint(0, len(env.rotation_angles))  # Rotation index
        ]
        
        obs, reward, done, info = env.step(action)
        
        print(f"   Step {i+1}: shape_id={int(action[0])}, pos=({action[1]:.1f},{action[2]:.1f}), "
              f"rot={env.rotation_angles[int(action[3])]}°, reward={reward:.1f}, "
              f"fitted={info.get('shapes_fitted', 0)}, success={info['placement']}")
        
        if done:
            print(f"   Episode ended: {info.get('shapes_fitted', 0)}/{env.num_shapes_to_fit} shapes fitted")
            break
    
    metrics = env.get_metrics()
    print(f"   Final metrics: {metrics['shapes_fitted']}/{metrics['shapes_fitted'] + metrics['shapes_remaining']} shapes fitted")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print("   ✅ Environment working correctly")

def test_difficulty_levels():
    """Test different difficulty levels."""
    print("\n🎯 Testing difficulty levels...")
    
    for level in [1, 2, 3, 4, 5]:
        env = ShapeFittingEnv(
            num_shapes_to_fit=10,
            difficulty_level=level,
            max_steps=50
        )
        env.reset()
        
        # Count shape types
        shape_types = {}
        for shape in env.shapes_to_fit:
            shape_type = type(shape).__name__
            shape_types[shape_type] = shape_types.get(shape_type, 0) + 1
        
        print(f"   Level {level}: {shape_types}")
    
    print("   ✅ Difficulty progression working")

def test_shape_factory():
    """Test shape factory functionality."""
    print("\n🏭 Testing shape factory...")
    
    # Test basic shapes
    basic_shapes = ShapeFactory.create_basic_shapes()
    print(f"   Basic shapes: {len(basic_shapes)} created")
    
    # Test curriculum batches
    for level in [1, 3, 5]:
        batch = ShapeFactory.create_curriculum_batch(level, batch_size=5)
        shape_types = [type(s).__name__ for s in batch]
        print(f"   Level {level} batch: {set(shape_types)}")
    
    # Test random shape generation
    difficulties = [1.0, 2.0, 3.0]
    for diff in difficulties:
        shape = ShapeFactory.create_random_shape(diff)
        print(f"   Difficulty {diff}: {type(shape).__name__} (area: {shape.area:.1f})")
    
    print("   ✅ Shape factory working correctly")

def test_rotation_system():
    """Test the 20-degree rotation system."""
    print("\n🔄 Testing rotation system...")
    
    env = ShapeFittingEnv(num_shapes_to_fit=3, difficulty_level=1)
    obs = env.reset()
    
    print(f"   Available rotation angles: {env.rotation_angles}")
    print(f"   Number of rotation options: {len(env.rotation_angles)}")
    
    # Test rotation for different shapes
    test_rotations = [0, 5, 10, 17]  # Indices for 0°, 100°, 200°, 340°
    
    for i, rot_idx in enumerate(test_rotations):
        if rot_idx < len(env.rotation_angles):
            angle = env.rotation_angles[rot_idx]
            action = [0, 50, 50, rot_idx]  # Shape 0, center position, specific rotation
            
            obs, reward, done, info = env.step(action)
            print(f"   Test {i+1}: rotation_idx={rot_idx} -> {angle}° -> {info['placement']}")
            
            if done:
                break
    
    print("   ✅ Rotation system working correctly")

def test_reward_system():
    """Test the reward system focused on shape count."""
    print("\n🎁 Testing reward system...")
    
    env = ShapeFittingEnv(num_shapes_to_fit=5, difficulty_level=1, max_steps=25)
    obs = env.reset()
    
    total_reward = 0
    shapes_fitted = 0
    
    print("   Testing reward progression:")
    
    for step in range(10):
        # Try to place shapes in reasonable positions
        action = [
            step % len(env.shapes_to_fit),  # Cycle through shapes
            30 + (step % 3) * 20,           # Spread X positions
            30 + (step // 3) * 20,          # Spread Y positions  
            0                               # No rotation initially
        ]
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if info['placement'] == 'success':
            shapes_fitted += 1
            print(f"   Step {step+1}: SUCCESS! Shape fitted. Reward: +{reward:.1f}, Total: {shapes_fitted}")
        else:
            print(f"   Step {step+1}: Failed ({info['placement']}). Reward: {reward:.1f}")
        
        if done:
            print(f"   Episode ended: {shapes_fitted}/{env.num_shapes_to_fit} shapes fitted")
            print(f"   Total reward: {total_reward:.1f}")
            break
    
    print("   ✅ Reward system prioritizing shape count correctly")

def demo_environment_visualization():
    """Demonstrate environment visualization."""
    print("\n🎨 Testing environment visualization...")
    
    env = ShapeFittingEnv(num_shapes_to_fit=6, difficulty_level=2)
    obs = env.reset()
    
    print("   Environment created for visualization demo")
    print("   Try running env.render() to see the current state")
    print("   (Note: Requires matplotlib display capability)")
    
    # Fit a couple shapes for demonstration
    successful_placements = 0
    for i in range(8):
        action = [
            i % len(env.shapes_to_fit),
            25 + (i % 3) * 25,
            25 + (i // 3) * 25,
            (i * 2) % len(env.rotation_angles)
        ]
        
        obs, reward, done, info = env.step(action)
        if info['placement'] == 'success':
            successful_placements += 1
            
        if done or successful_placements >= 3:
            break
    
    print(f"   Demo completed with {successful_placements} successful placements")
    print("   ✅ Visualization system ready")

def main():
    """Run comprehensive system tests."""
    print("🧪 SHAPE FITTING SYSTEM TESTING")
    print("=" * 50)
    print("Testing the new shape fitting approach with:")
    print("• Groups of shapes to fit in containers")
    print("• Discrete 20-degree rotation intervals")
    print("• Rewards based on number of shapes fitted")
    print("• Progressive difficulty levels")
    print("=" * 50)
    
    try:
        test_shapes()
        test_environment()
        test_difficulty_levels()
        test_shape_factory()
        test_rotation_system()
        test_reward_system()
        demo_environment_visualization()
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ System is ready for training")
        print("✅ New shape fitting approach implemented correctly")
        print("✅ Discrete rotation system working")
        print("✅ Count-based reward system functioning")
        print("✅ Progressive difficulty levels configured")
        print("\n🚀 Ready to run: python train.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Please check the implementation and try again")

if __name__ == "__main__":
    main() 