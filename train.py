

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from env import ContinuousContainerEnv
from agent import PPOTrainer
from shapes import ShapeFactory
import pandas as pd

class CurriculumManager:
    """Manages curriculum learning progression."""
    
    def __init__(self, start_level: int = 1, max_level: int = 5):
        self.current_level = start_level
        self.max_level = max_level
        self.level_episodes = 0
        self.level_successes = 0
        
        # Criteria for advancing to next level
        self.level_requirements = {
            1: {"episodes": 100, "success_rate": 0.6, "avg_utilization": 0.4},
            2: {"episodes": 150, "success_rate": 0.5, "avg_utilization": 0.5},
            3: {"episodes": 200, "success_rate": 0.4, "avg_utilization": 0.6},
            4: {"episodes": 250, "success_rate": 0.3, "avg_utilization": 0.7},
            5: {"episodes": 300, "success_rate": 0.2, "avg_utilization": 0.75},
        }
    
    def should_advance(self, recent_success_rate: float, recent_utilization: float) -> bool:
        """Check if agent should advance to next curriculum level."""
        if self.current_level >= self.max_level:
            return False
        
        requirements = self.level_requirements[self.current_level]
        
        if (self.level_episodes >= requirements["episodes"] and
            recent_success_rate >= requirements["success_rate"] and 
            recent_utilization >= requirements["avg_utilization"]):
            return True
        
        return False
    
    def advance_level(self):
        """Advance to next curriculum level."""
        if self.current_level < self.max_level:
            self.current_level += 1
            self.level_episodes = 0
            self.level_successes = 0
            print(f"\n🎓 CURRICULUM ADVANCED TO LEVEL {self.current_level}!")
            print(f"   Difficulty increased - more complex shapes and challenges ahead!")
    
    def update_stats(self, success: bool):
        """Update level statistics."""
        self.level_episodes += 1
        if success:
            self.level_successes += 1

def create_challenging_scenarios():
    """Create specific challenging scenarios for testing."""
    scenarios = [
        {
            "name": "Tetris Challenge",
            "description": "Complex interlocking shapes requiring precise rotation",
            "curriculum_level": 3,
            "container_size": (80, 80),
            "emphasis": "rotation_critical"
        },
        {
            "name": "Efficiency Master",
            "description": "Large container with many small pieces - maximize utilization",
            "curriculum_level": 4,
            "container_size": (120, 120),
            "emphasis": "utilization_critical"
        },
        {
            "name": "Tight Fit",
            "description": "Small container with large shapes - every placement matters",
            "curriculum_level": 5,
            "container_size": (60, 60),
            "emphasis": "precision_critical"
        },
        {
            "name": "Mixed Madness",
            "description": "Ultimate challenge - mixed shapes, sizes, and rotations",
            "curriculum_level": 5,
            "container_size": (100, 100),
            "emphasis": "all_skills"
        }
    ]
    return scenarios

def run_scenario_evaluation(trainer, scenario):
    """Evaluate agent on a specific challenging scenario."""
    print(f"\n🎯 SCENARIO EVALUATION: {scenario['name']}")
    print(f"   {scenario['description']}")
    
    # Create environment for this scenario
    env = ContinuousContainerEnv(
        container_width=scenario['container_size'][0],
        container_height=scenario['container_size'][1],
        curriculum_level=scenario['curriculum_level'],
        max_shapes=25
    )
    
    results = []
    
    for episode in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            obs_tensor = trainer.network.get_action(
                trainer.network.device.type(obs).unsqueeze(0), 
                deterministic=True
            )[0]
            
            action = obs_tensor.cpu().numpy().squeeze()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        metrics = env.get_metrics()
        results.append({
            'episode': episode,
            'reward': metrics['episode_reward'],
            'utilization': metrics['utilization'],
            'shapes_placed': metrics['shapes_placed'],
            'success': metrics['shapes_remaining'] == 0
        })
    
    # Analyze results
    df = pd.DataFrame(results)
    
    print(f"   Results over 10 episodes:")
    print(f"   • Average Reward: {df['reward'].mean():.2f}")
    print(f"   • Average Utilization: {df['utilization'].mean():.2%}")
    print(f"   • Success Rate: {df['success'].mean():.2%}")
    print(f"   • Average Shapes Placed: {df['shapes_placed'].mean():.1f}")
    
    return df

def main():
    """Main training loop with curriculum learning and advanced challenges."""
    
    print("🚀 ADVANCED CONTAINER PACKING TRAINING")
    print("=" * 60)
    print("Training an agent to pack complex shapes with:")
    print("• Multiple shape types (rectangles, circles, triangles, L-shapes, irregular)")
    print("• Continuous positioning and rotation")
    print("• Realistic collision detection")
    print("• Progressive difficulty curriculum")
    print("• Challenging reward structure")
    print("=" * 60)
    
    # Create environment
    print("\n🏗️  Creating advanced environment...")
    env = ContinuousContainerEnv(
        container_width=100,
        container_height=100,
        curriculum_level=1,
        max_shapes=20
    )
    
    print(f"   Container: {env.container_width}x{env.container_height}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space.shape}")
    
    # Create trainer
    print("\n🧠 Creating PPO trainer...")
    trainer = PPOTrainer(
        env=env,
        obs_size=env.observation_space.shape[0],
        max_shapes=20,
        lr=3e-4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    print(f"   Device: {trainer.device}")
    print(f"   Network parameters: {sum(p.numel() for p in trainer.network.parameters()):,}")
    
    # Curriculum manager
    curriculum = CurriculumManager()
    
    # Training parameters
    total_timesteps = 2000000  # 2M timesteps
    eval_frequency = 20000
    save_frequency = 100000
    
    print(f"\n🎯 Training configuration:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Evaluation frequency: {eval_frequency:,}")
    print(f"   Save frequency: {save_frequency:,}")
    
    # Training metrics
    training_metrics = {
        'timesteps': [],
        'curriculum_level': [],
        'avg_reward': [],
        'avg_utilization': [],
        'success_rate': [],
        'avg_episode_length': []
    }
    
    # Create challenging scenarios
    scenarios = create_challenging_scenarios()
    
    print(f"\n🎮 Created {len(scenarios)} challenging scenarios for evaluation")
    
    # Start training
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    timestep = 0
    
    try:
        while timestep < total_timesteps:
            print(f"\n⏰ Timestep {timestep:,} / {total_timesteps:,}")
            print(f"📚 Curriculum Level: {curriculum.current_level}")
            
            # Update environment curriculum level
            env.curriculum_level = curriculum.current_level
            
            # Collect trajectories and train
            batch = trainer.collect_trajectories(n_steps=2048)
            losses = trainer.update_policy(batch)
            
            timestep += len(batch['rewards'])
            
            # Update curriculum statistics
            recent_episodes = trainer.episode_rewards[-20:] if len(trainer.episode_rewards) >= 20 else trainer.episode_rewards
            recent_success = trainer.success_rates[-20:] if len(trainer.success_rates) >= 20 else trainer.success_rates
            recent_utilization = trainer.utilization_scores[-20:] if len(trainer.utilization_scores) >= 20 else trainer.utilization_scores
            
            if recent_episodes:
                avg_reward = np.mean(recent_episodes)
                success_rate = np.mean(recent_success)
                avg_utilization = np.mean(recent_utilization)
                
                print(f"📊 Recent Performance (last 20 episodes):")
                print(f"   • Average Reward: {avg_reward:.2f}")
                print(f"   • Success Rate: {success_rate:.2%}")
                print(f"   • Average Utilization: {avg_utilization:.2%}")
                print(f"   • Policy Loss: {losses['policy_loss']:.4f}")
                print(f"   • Value Loss: {losses['value_loss']:.4f}")
                
                # Check curriculum advancement
                if curriculum.should_advance(success_rate, avg_utilization):
                    curriculum.advance_level()
                
                # Store metrics
                training_metrics['timesteps'].append(timestep)
                training_metrics['curriculum_level'].append(curriculum.current_level)
                training_metrics['avg_reward'].append(avg_reward)
                training_metrics['avg_utilization'].append(avg_utilization)
                training_metrics['success_rate'].append(success_rate)
                training_metrics['avg_episode_length'].append(np.mean(trainer.episode_lengths[-20:]) if len(trainer.episode_lengths) >= 20 else 0)
            
            # Evaluation
            if timestep % eval_frequency == 0:
                print(f"\n🔍 EVALUATION AT TIMESTEP {timestep:,}")
                trainer.evaluate(n_episodes=10)
                
                # Test on challenging scenarios
                if timestep >= 200000:  # After some training
                    print("\n🎯 CHALLENGING SCENARIO TESTS:")
                    for scenario in scenarios:
                        if curriculum.current_level >= scenario['curriculum_level'] - 1:
                            scenario_results = run_scenario_evaluation(trainer, scenario)
            
            # Save model
            if timestep % save_frequency == 0:
                model_path = f"continuous_ppo_checkpoint_{timestep}.pt"
                trainer.save_model(model_path)
                
                # Save training metrics
                metrics_df = pd.DataFrame(training_metrics)
                metrics_df.to_csv(f"training_metrics_{timestep}.csv", index=False)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    finally:
        # Final evaluation and save
        end_time = time.time()
        training_duration = end_time - start_time
        
        print(f"\n" + "=" * 60)
        print("🏁 TRAINING COMPLETED")
        print("=" * 60)
        print(f"⏱️  Training Duration: {training_duration/3600:.2f} hours")
        print(f"📈 Final Curriculum Level: {curriculum.current_level}")
        print(f"🎯 Total Episodes: {len(trainer.episode_rewards)}")
        
        # Final evaluation
        print(f"\n🔍 FINAL EVALUATION:")
        trainer.evaluate(n_episodes=20)
        
        # Test all scenarios
        print(f"\n🎯 FINAL SCENARIO EVALUATION:")
        final_results = {}
        for scenario in scenarios:
            results = run_scenario_evaluation(trainer, scenario)
            final_results[scenario['name']] = results
        
        # Save final model and results
        trainer.save_model("continuous_ppo_final.pt")
        trainer.plot_training_curves()
        
        # Save final metrics
        final_metrics_df = pd.DataFrame(training_metrics)
        final_metrics_df.to_csv("final_training_metrics.csv", index=False)
        
        print(f"\n✅ All files saved!")
        print(f"   • Model: continuous_ppo_final.pt")
        print(f"   • Training curves: ppo_training_curves.png")
        print(f"   • Metrics: final_training_metrics.csv")
        
        # Show summary statistics
        if trainer.episode_rewards:
            final_rewards = trainer.episode_rewards[-50:] if len(trainer.episode_rewards) >= 50 else trainer.episode_rewards
            final_utilization = trainer.utilization_scores[-50:] if len(trainer.utilization_scores) >= 50 else trainer.utilization_scores
            final_success = trainer.success_rates[-50:] if len(trainer.success_rates) >= 50 else trainer.success_rates
            
            print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
            print(f"   • Average Reward (last 50): {np.mean(final_rewards):.2f}")
            print(f"   • Average Utilization (last 50): {np.mean(final_utilization):.2%}")
            print(f"   • Success Rate (last 50): {np.mean(final_success):.2%}")
            print(f"   • Best Episode Reward: {max(trainer.episode_rewards):.2f}")
            print(f"   • Best Utilization: {max(trainer.utilization_scores):.2%}")

if __name__ == "__main__":
    main() 