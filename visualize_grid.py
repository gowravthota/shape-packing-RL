import pygame
import numpy as np
import torch
import time
from typing import List

from env import ShapeFittingEnv
from agent import PPOTrainer


def create_env() -> ShapeFittingEnv:
    return ShapeFittingEnv(
        container_width=100,
        container_height=100,
        num_shapes_to_fit=8,
        difficulty_level=1,
        max_steps=40,
    )


def draw_env(screen, env: ShapeFittingEnv, origin_x: int, origin_y: int, scale: float = 2.0):
    # Container
    width = int(env.container_width * scale)
    height = int(env.container_height * scale)
    pygame.draw.rect(screen, (240, 240, 240), (origin_x, origin_y, width, height))
    pygame.draw.rect(screen, (60, 60, 60), (origin_x, origin_y, width, height), 2)

    # Draw placed shapes (approximate)
    for shape in env.container.placed_shapes:
        cx = int(origin_x + shape.position[0] * scale)
        cy = int(origin_y + shape.position[1] * scale)
        shape_type = type(shape).__name__

        if shape_type == 'RectangleShape':
            w = int(shape.width * scale)
            h = int(shape.height * scale)
            surf = pygame.Surface((w + 6, h + 6), pygame.SRCALPHA)
            pygame.draw.rect(surf, (120, 200, 120), (3, 3, w, h))
            pygame.draw.rect(surf, (40, 40, 40), (3, 3, w, h), 2)
            if shape.rotation != 0:
                surf = pygame.transform.rotate(surf, -shape.rotation)
            rect = surf.get_rect(center=(cx, cy))
            screen.blit(surf, rect)
        elif shape_type == 'CircleShape':
            r = max(2, int(getattr(shape, 'radius', 4) * scale))
            pygame.draw.circle(screen, (120, 200, 120), (cx, cy), r)
            pygame.draw.circle(screen, (40, 40, 40), (cx, cy), r, 1)
        else:
            # Fallback to bounds box
            minx, miny, maxx, maxy = shape.bounds
            w = int((maxx - minx) * scale)
            h = int((maxy - miny) * scale)
            rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
            pygame.draw.rect(screen, (120, 200, 120), rect)
            pygame.draw.rect(screen, (40, 40, 40), rect, 1)


def main():
    pygame.init()

    grid_rows, grid_cols = 5, 5
    scale = 2.0
    cell_w, cell_h = int(100 * scale) + 20, int(100 * scale) + 40
    screen = pygame.display.set_mode((grid_cols * cell_w, grid_rows * cell_h))
    pygame.display.set_caption("Shape Packing RL - 5x5 Parallel Visualization")
    font = pygame.font.Font(None, 18)

    # Create envs
    envs: List[ShapeFittingEnv] = [create_env() for _ in range(grid_rows * grid_cols)]

    # Lightweight agent
    trainer = PPOTrainer(
        env=envs[0],
        obs_size=envs[0].observation_space.shape[0],
        num_shapes=envs[0].num_shapes_to_fit,
        num_rotations=len(envs[0].rotation_angles),
    )

    # States
    obses = [env.reset() for env in envs]
    running = True
    paused = False
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            # Step each env with current policy (greedy for stability)
            actions = []
            for i, obs in enumerate(obses):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    action_t, _ = trainer.network.get_action(obs_t, deterministic=True)
                action = action_t.squeeze(0).cpu().numpy()
                actions.append(action)

            next_obses = []
            infos = []
            for env, obs, action in zip(envs, obses, actions):
                next_obs, reward, done, info = env.step(action)
                next_obses.append(next_obs if not done else env.reset())
                infos.append(info)
            obses = next_obses

        screen.fill((250, 250, 250))

        # Draw grid
        for r in range(grid_rows):
            for c in range(grid_cols):
                idx = r * grid_cols + c
                origin_x = c * cell_w + 10
                origin_y = r * cell_h + 10
                draw_env(screen, envs[idx], origin_x, origin_y, scale)

                # Overlay small stats
                metrics = envs[idx].get_metrics()
                text = f"Fitted {metrics['shapes_fitted']}/{envs[idx].num_shapes_to_fit}"
                txt = font.render(text, True, (0, 0, 0))
                screen.blit(txt, (origin_x, origin_y + int(100 * scale) + 8))

        pygame.display.flip()
        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()

