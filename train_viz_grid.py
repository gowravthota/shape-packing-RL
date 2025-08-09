import pygame
import numpy as np
import torch
from typing import List

from env import ShapeFittingEnv
from agent import PPOTrainer


def make_env() -> ShapeFittingEnv:
    return ShapeFittingEnv(
        container_width=100,
        container_height=100,
        num_shapes_to_fit=8,
        difficulty_level=1,
        max_steps=40,
        save_images=False,
    )


def draw_env(screen, env: ShapeFittingEnv, ox: int, oy: int, scale: float, font):
    width = int(env.container_width * scale)
    height = int(env.container_height * scale)
    pygame.draw.rect(screen, (245, 245, 245), (ox, oy, width, height))
    pygame.draw.rect(screen, (60, 60, 60), (ox, oy, width, height), 2)

    for shape in env.container.placed_shapes:
        cx = int(ox + shape.position[0] * scale)
        cy = int(oy + shape.position[1] * scale)
        name = type(shape).__name__
        if name == 'RectangleShape':
            w = int(shape.width * scale)
            h = int(shape.height * scale)
            surf = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
            pygame.draw.rect(surf, (110, 190, 110), (3, 3, w, h))
            pygame.draw.rect(surf, (30, 30, 30), (3, 3, w, h), 2)
            if shape.rotation != 0:
                surf = pygame.transform.rotate(surf, -shape.rotation)
            rect = surf.get_rect(center=(cx, cy))
            screen.blit(surf, rect)
        elif name == 'CircleShape':
            r = max(2, int(getattr(shape, 'radius', 4) * scale))
            pygame.draw.circle(screen, (110, 190, 110), (cx, cy), r)
            pygame.draw.circle(screen, (30, 30, 30), (cx, cy), r, 1)
        else:
            minx, miny, maxx, maxy = shape.bounds
            w = int((maxx - minx) * scale)
            h = int((maxy - miny) * scale)
            rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
            pygame.draw.rect(screen, (110, 190, 110), rect)
            pygame.draw.rect(screen, (30, 30, 30), rect, 1)

    m = env.get_metrics()
    label = f"Fitted {m['shapes_fitted']}/{env.num_shapes_to_fit} | Util {m['utilization']:.0%}"
    txt = font.render(label, True, (0, 0, 0))
    screen.blit(txt, (ox, oy + height + 6))


def main():
    pygame.init()
    rows, cols = 5, 5
    scale = 2.0
    cell_w, cell_h = int(100 * scale) + 20, int(100 * scale) + 36
    screen = pygame.display.set_mode((cols * cell_w, rows * cell_h))
    pygame.display.set_caption("Shape Packing RL - Live 5x5 PPO Visualization")
    font = pygame.font.Font(None, 18)

    envs: List[ShapeFittingEnv] = [make_env() for _ in range(rows * cols)]
    obses = [env.reset() for env in envs]

    # One PPO trainer; act deterministically for stable viz
    trainer = PPOTrainer(
        env=envs[0],
        obs_size=envs[0].observation_space.shape[0],
        num_shapes=envs[0].num_shapes_to_fit,
        num_rotations=len(envs[0].rotation_angles),
    )

    clock = pygame.time.Clock()
    running = True
    paused = False
    step_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            # Lightweight on-policy update: collect tiny batch from subset envs
            batch_obs, batch_actions, batch_rewards, batch_dones, batch_values, batch_logps = [], [], [], [], [], []
            for i, (env, obs) in enumerate(zip(envs, obses)):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    action_t, logp_t = trainer.network.get_action(obs_t, deterministic=False)
                    _, _, _, v_t = trainer.network.forward(obs_t)
                action = action_t.squeeze(0).cpu().numpy()

                next_obs, reward, done, _ = env.step(action)

                batch_obs.append(obs)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_dones.append(done)
                batch_values.append(v_t.item())
                batch_logps.append(logp_t.item())

                obses[i] = next_obs if not done else env.reset()

            # Convert to tensors and do a tiny PPO update step occasionally
            step_counter += 1
            if step_counter % 8 == 0:
                batch = {
                    'observations': torch.FloatTensor(np.array(batch_obs)).to(trainer.device),
                    'actions': torch.FloatTensor(np.array(batch_actions)).to(trainer.device),
                    'log_probs': torch.FloatTensor(np.array(batch_logps)).to(trainer.device),
                    'rewards': torch.FloatTensor(batch_rewards).to(trainer.device),
                    'dones': torch.BoolTensor(batch_dones).to(trainer.device),
                    'values': torch.FloatTensor(batch_values).to(trainer.device),
                }
                trainer.update_policy(batch, n_epochs=2, batch_size=64)

        screen.fill((252, 252, 252))
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                ox, oy = c * cell_w + 10, r * cell_h + 10
                draw_env(screen, envs[idx], ox, oy, scale, font)

        pygame.display.flip()
        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()

