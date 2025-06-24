import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from env import ShapeFittingEnv
from agent import PPOTrainer
from shapes import ShapeFactory
import pandas as pd

class ProgressiveTrainingManager:
    """Manages progressive training difficulty."""
    
    def __init__(self, start_difficulty: int = 1, max_difficulty: int = 5):
        self.current_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_episodes = 0
        self.difficulty_successes = 0
        
        # Criteria for advancing to next difficulty level
        self.advancement_requirements = {
            1: {"min_episodes": 50, "success_rate": 0.7, "avg_shapes_fitted": 7},
            2: {"min_episodes": 75, "success_rate": 0.6, "avg_shapes_fitted": 6},
            3: {"min_episodes": 100, "success_rate": 0.5, "avg_shapes_fitted": 5},
            4: {"min_episodes": 125, "success_rate": 0.4, "avg_shapes_fitted": 4},
            5: {"min_episodes": 150, "success_rate": 0.3, "avg_shapes_fitted": 3},
        }
    
    def should_advance(self, recent_success_rate: float, recent_avg_fitted: float) -> bool:
        """Check if agent should advance to next difficulty level."""
        if self.current_difficulty >= self.max_difficulty:
            return False
        
        requirements = self.advancement_requirements[self.current_difficulty]
        
        if (self.difficulty_episodes >= requirements["min_episodes"] and
            recent_success_rate >= requirements["success_rate"] and 
            recent_avg_fitted >= requirements["avg_shapes_fitted"]):
            return True
        
        return False
    
    def advance_difficulty(self):
        """Advance to next difficulty level."""
        if self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            self.difficulty_episodes = 0
            self.difficulty_successes = 0
            print(f"\n🎓 DIFFICULTY ADVANCED TO LEVEL {self.current_difficulty}!")
            print(f"   Training now using more complex shapes and arrangements!")
    
    def update_stats(self, success: bool):
        """Update difficulty level statistics."""
        self.difficulty_episodes += 1
        if success:
            self.difficulty_successes += 1

def create_evaluation_scenarios():
    """Create specific evaluation scenarios for testing."""
    scenarios = [
        {
            "name": "Basic Shapes",
            "description": "Simple rectangles and circles - test fundamental fitting",
            "difficulty_level": 1,
            "num_shapes": 8,
            "container_size": (100, 100),
            "max_steps": 40
        },
        {
            "name": "Mixed Shapes",
            "description": "Rectangles, circles, and triangles - test variety handling",
            "difficulty_level": 2,
            "num_shapes": 10,
            "container_size": (100, 100),
            "max_steps": 50
        },
        {
            "name": "Complex Shapes",
            "description": "All shape types including L-shapes - test complex fitting",
            "difficulty_level": 3,
            "num_shapes": 12,
            "container_size": (100, 100),
            "max_steps": 60
        },
        {
            "name": "Expert Challenge",
            "description": "Maximum complexity with irregular shapes",
            "difficulty_level": 4,
            "num_shapes": 15,
            "container_size": (100, 100),
            "max_steps": 75
        },
        {
            "name": "Tight Space",
            "description": "Large shapes in smaller container - ultimate precision test",
            "difficulty_level": 5,
            "num_shapes": 10,
            "container_size": (80, 80),
            "max_steps": 50
        }
    ]
    return scenarios

def run_scenario_evaluation(trainer, scenario):
    """Evaluate agent on a specific scenario."""
    print(f"\n🎯 SCENARIO: {scenario['name']}")
    print(f"   {scenario['description']}")
    
    # Create environment for this scenario
    env = ShapeFittingEnv(
        container_width=scenario['container_size'][0],
        container_height=scenario['container_size'][1],
        num_shapes_to_fit=scenario['num_shapes'],
        difficulty_level=scenario['difficulty_level'],
        max_steps=scenario['max_steps']
    )
    
    results = []
    
    for episode in range(5):  # Test 5 episodes per scenario
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
            action, _ = trainer.network.get_action(obs_tensor, deterministic=True)
            
            action_np = action.detach().cpu().numpy().squeeze()
            obs, reward, done, info = env.step(action_np)
            episode_reward += reward
        
        metrics = env.get_metrics()
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'shapes_fitted': metrics['shapes_fitted'],
            'success_rate': metrics['success_rate'],
            'utilization': metrics['utilization']
        })
    
    # Analyze results
    df = pd.DataFrame(results)
    
    print(f"   Results (5 episodes):")
    print(f"   • Avg Reward: {df['reward'].mean():.1f}")
    print(f"   • Avg Shapes Fitted: {df['shapes_fitted'].mean():.1f}/{scenario['num_shapes']}")
    print(f"   • Success Rate: {df['success_rate'].mean():.2%}")
    print(f"   • Avg Utilization: {df['utilization'].mean():.2%}")
    
    return df

def main():
    """Main training loop with progressive difficulty and comprehensive evaluation."""
    
    print("🚀 SHAPE FITTING REINFORCEMENT LEARNING TRAINING")
    print("=" * 65)
    print("Training an agent to fit groups of shapes in containers with:")
    print("• Multiple shape types (rectangles, circles, triangles, L-shapes, irregular)")
    print("• Discrete 20-degree rotation intervals")
    print("• Reward based on number of shapes successfully fitted")
    print("• Progressive difficulty levels")
    print("• Realistic collision detection and spatial reasoning")
    print("=" * 65)
    
    # Create environment
    print("\n🏗️  Creating shape fitting environment...")
    env = ShapeFittingEnv(
        container_width=100,
        container_height=100,
        num_shapes_to_fit=10,
        difficulty_level=1,
        max_steps=50
    )
    
    print(f"   Container: {env.container_width}x{env.container_height}")
    print(f"   Shapes to fit: {env.num_shapes_to_fit}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Rotation angles: {env.rotation_angles}")
    
    # Create trainer
    print("\n🧠 Creating PPO trainer...")
    trainer = PPOTrainer(
        env=env,
        obs_size=env.observation_space.shape[0],
        num_shapes=env.num_shapes_to_fit,
        num_rotations=len(env.rotation_angles),
        lr=3e-4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Progressive training manager
    progress_manager = ProgressiveTrainingManager()
    
    # Training parameters
    total_timesteps = 2000000  # 2M timesteps
    eval_frequency = 25000
    save_frequency = 100000
    scenario_eval_frequency = 100000
    
    print(f"\n🎯 Training configuration:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Evaluation frequency: {eval_frequency:,}")
    print(f"   Save frequency: {save_frequency:,}")
    print(f"   Progressive difficulty enabled")
    
    # Training metrics
    training_metrics = {
        'timesteps': [],
        'difficulty_level': [],
        'avg_reward': [],
        'avg_shapes_fitted': [],
        'avg_success_rate': [],
        'policy_loss': [],
        'value_loss': []
    }
    
    print(f"\n🔥 Starting training...")
    start_time = time.time()
    timesteps_completed = 0
    
    try:
        while timesteps_completed < total_timesteps:
            # Update environment difficulty if needed
            if env.difficulty_level != progress_manager.current_difficulty:
                env.difficulty_level = progress_manager.current_difficulty
                env.reset()  # Reset to apply new difficulty
            
            # Collect trajectories and update policy
            print(f"\nStep {timesteps_completed:,}/{total_timesteps:,} (Difficulty {progress_manager.current_difficulty})")
            batch = trainer.collect_trajectories(n_steps=2048)
            metrics = trainer.update_policy(batch)
            
            timesteps_completed += 2048
            
            # Track training metrics
            training_metrics['timesteps'].append(timesteps_completed)
            training_metrics['difficulty_level'].append(progress_manager.current_difficulty)
            training_metrics['avg_reward'].append(metrics['avg_reward'])
            training_metrics['avg_shapes_fitted'].append(metrics['avg_shapes_fitted'])
            training_metrics['avg_success_rate'].append(metrics['avg_success_rate'])
            training_metrics['policy_loss'].append(metrics['policy_loss'])
            training_metrics['value_loss'].append(metrics['value_loss'])
            
            # Print progress
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"  Avg Shapes Fitted: {metrics['avg_shapes_fitted']:.1f}/{env.num_shapes_to_fit}")
            print(f"  Avg Success Rate: {metrics['avg_success_rate']:.2%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            
            # Check for difficulty progression
            if len(trainer.episode_success_rates) >= 20:  # Need sufficient data
                recent_success_rate = np.mean(list(trainer.episode_success_rates)[-20:])
                recent_avg_fitted = np.mean(list(trainer.episode_shapes_fitted)[-20:])
                
                if progress_manager.should_advance(recent_success_rate, recent_avg_fitted):
                    progress_manager.advance_difficulty()
            
            # Regular evaluation
            if timesteps_completed % eval_frequency == 0:
                print("\n📊 Running evaluation...")
                eval_metrics = trainer.evaluate(n_episodes=10)
                print(f"  Eval Avg Reward: {eval_metrics['avg_reward']:.2f}")
                print(f"  Eval Avg Shapes Fitted: {eval_metrics['avg_shapes_fitted']:.1f}")
                print(f"  Eval Success Rate: {eval_metrics['success_rate']:.2%}")
            
            # Comprehensive scenario evaluation
            if timesteps_completed % scenario_eval_frequency == 0:
                print("\n🧪 Running comprehensive scenario evaluation...")
                scenarios = create_evaluation_scenarios()
                scenario_results = {}
                
                for scenario in scenarios:
                    scenario_df = run_scenario_evaluation(trainer, scenario)
                    scenario_results[scenario['name']] = scenario_df
                
                # Save scenario results
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                results_path = f"metrics/scenario_evaluation_{timesteps_completed}_{timestamp}.csv"
                
                all_results = []
                for scenario_name, df in scenario_results.items():
                    df['scenario'] = scenario_name
                    df['timestep'] = timesteps_completed
                    all_results.append(df)
                
                combined_df = pd.concat(all_results, ignore_index=True)
                os.makedirs('metrics', exist_ok=True)
                combined_df.to_csv(results_path, index=False)
                print(f"  Scenario results saved to {results_path}")
            
            # Save model
            if timesteps_completed % save_frequency == 0:
                model_path = f"models/shape_fitting_model_{timesteps_completed}.pth"
                trainer.save_model(model_path)
                print(f"  Model saved to {model_path}")
            
            # Save training metrics
            if timesteps_completed % (save_frequency // 2) == 0:
                metrics_df = pd.DataFrame(training_metrics)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                metrics_path = f"metrics/training_metrics_{timestamp}.csv"
                os.makedirs('metrics', exist_ok=True)
                metrics_df.to_csv(metrics_path, index=False)
                print(f"  Training metrics saved to {metrics_path}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Final evaluation and save
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"   Total time: {elapsed_time/3600:.2f} hours")
    print(f"   Timesteps completed: {timesteps_completed:,}")
    print(f"   Final difficulty level: {progress_manager.current_difficulty}")
    
    # Final comprehensive evaluation
    print("\n🏆 Final comprehensive evaluation...")
    scenarios = create_evaluation_scenarios()
    final_results = {}
    
    for scenario in scenarios:
        scenario_df = run_scenario_evaluation(trainer, scenario)
        final_results[scenario['name']] = scenario_df
    
    # Save final model and results
    final_model_path = f"models/shape_fitting_final_{timesteps_completed}.pth"
    trainer.save_model(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save final training metrics
    final_metrics_df = pd.DataFrame(training_metrics)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_metrics_path = f"metrics/final_training_metrics_{timestamp}.csv"
    final_metrics_df.to_csv(final_metrics_path, index=False)
    print(f"Final metrics saved to {final_metrics_path}")
    
    # Plot training curves
    print("\n📈 Generating training plots...")
    trainer.plot_training_curves()
    
    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 