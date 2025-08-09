import torch
from env import ShapeFittingEnv
from agent import PPOTrainer


def main():
    env = ShapeFittingEnv(
        container_width=100,
        container_height=100,
        num_shapes_to_fit=10,
        difficulty_level=1,
        max_steps=50,
        save_images=False,
    )

    trainer = PPOTrainer(
        env=env,
        obs_size=env.observation_space.shape[0],
        num_shapes=env.num_shapes_to_fit,
        num_rotations=len(env.rotation_angles),
    )

    # Short demo training loop
    total_steps = 20000
    steps_completed = 0

    while steps_completed < total_steps:
        batch = trainer.collect_trajectories(n_steps=1024)
        trainer.update_policy(batch, n_epochs=5, batch_size=128)
        steps_completed += len(batch['rewards'])
        print(f"Progress: {steps_completed}/{total_steps}")

    # Final evaluation
    eval_metrics = trainer.evaluate(n_episodes=5)
    print("Final:", eval_metrics)


if __name__ == "__main__":
    main()

