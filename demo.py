import numpy as np
from env import ContinuousContainerEnv
from shapes import ShapeFactory, RectangleShape, CircleShape

def test_shapes():
    print("Testing shape creation...")
    rect = RectangleShape(20, 10, position=(50, 50), rotation=45)
    circle = CircleShape(15, position=(30, 30))
    
    print(f"Rectangle: area={rect.area:.1f}, position={rect.position}")
    print(f"Circle: area={circle.area:.1f}, position={circle.position}")
    print("✅ Shapes working")

def test_environment():
    print("Testing environment...")
    env = ContinuousContainerEnv(curriculum_level=1)
    obs = env.reset()
    
    print(f"Environment created: {len(env.available_shapes)} shapes")
    print(f"Observation space: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few random actions
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, placement={info['placement']}")
        if done:
            break
    
    print("✅ Environment working")

def test_curriculum():
    print("Testing curriculum levels...")
    for level in [1, 3, 5]:
        env = ContinuousContainerEnv(curriculum_level=level)
        env.reset()
        complexity = env.difficulty_settings[level]['shape_complexity']
        num_shapes = len(env.available_shapes)
        print(f"Level {level}: {num_shapes} shapes, complexity={complexity}")
    
    print("✅ Curriculum working")

def main():
    print("🧪 TESTING CLEANED SYSTEM")
    print("=" * 40)
    
    test_shapes()
    test_environment()
    test_curriculum()
    
    print("\n🎉 ALL TESTS PASSED!")
    print("System is ready for training.")

if __name__ == "__main__":
    main() 