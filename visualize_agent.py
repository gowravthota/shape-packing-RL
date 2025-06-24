import pygame
import numpy as np
import torch
import sys
import time
from env import ShapeFittingEnv
from agent import PPOTrainer

# Colors
COLORS = {
    'background': (245, 245, 245),
    'container': (60, 60, 60),
    'container_fill': (255, 255, 255),
    'fitted_shape': (100, 200, 100),
    'unfitted_shape': (200, 100, 100),
    'current_shape': (100, 100, 255),
    'shape_border': (40, 40, 40),
    'text': (0, 0, 0),
    'success': (50, 150, 50),
    'failure': (150, 50, 50),
    'info_bg': (230, 230, 230)
}

class ShapeFittingVisualizer:
    def __init__(self, env, agent=None, window_size=(1400, 800)):
        pygame.init()
        
        self.env = env
        self.agent = agent
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Shape Fitting RL - Agent Visualization")
        
        # Layout
        self.container_scale = 4
        self.container_offset = (50, 50)
        self.info_panel_x = 500
        self.shapes_panel_x = 800
        
        # Fonts
        self.small_font = pygame.font.Font(None, 20)
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        
        # Animation
        self.animation_speed = 0.8
        
        # Game state
        self.running = True
        self.paused = False
        self.episode_count = 0
        self.step_count = 0
        
        # Status message
        self.status_message = ""
        self.status_color = COLORS['text']
        
    def draw_container(self):
        """Draw the container where shapes are fitted."""
        x, y = self.container_offset
        width = self.env.container_width * self.container_scale
        height = self.env.container_height * self.container_scale
        
        # Container background
        pygame.draw.rect(self.screen, COLORS['container_fill'], (x, y, width, height))
        # Container border
        pygame.draw.rect(self.screen, COLORS['container'], (x, y, width, height), 3)
        
        # Draw grid for visual aid
        grid_size = 20 * self.container_scale
        for i in range(0, int(width), grid_size):
            pygame.draw.line(self.screen, (240, 240, 240), (x + i, y), (x + i, y + height), 1)
        for i in range(0, int(height), grid_size):
            pygame.draw.line(self.screen, (240, 240, 240), (x, y + i), (x + width, y + i), 1)
    
    def draw_shape(self, shape, color=None, offset=(0, 0)):
        """Draw a shape at its position."""
        if color is None:
            color = COLORS['fitted_shape'] if shape.is_fitted else COLORS['unfitted_shape']
            
        x_offset, y_offset = self.container_offset
        x_offset += offset[0]
        y_offset += offset[1]
        scale = self.container_scale
        
        x = shape.position[0] * scale + x_offset
        y = shape.position[1] * scale + y_offset
        
        shape_type = type(shape).__name__
        
        if shape_type == 'RectangleShape':
            # Rectangle
            width = shape.width * scale
            height = shape.height * scale
            
            # Create a surface for rotation
            surf = pygame.Surface((width + 10, height + 10), pygame.SRCALPHA)
            pygame.draw.rect(surf, color, (5, 5, width, height))
            pygame.draw.rect(surf, COLORS['shape_border'], (5, 5, width, height), 2)
            
            if shape.rotation != 0:
                surf = pygame.transform.rotate(surf, -shape.rotation)
            
            rect = surf.get_rect()
            rect.center = (x, y)
            self.screen.blit(surf, rect)
            
        elif shape_type == 'CircleShape':
            # Circle
            radius = max(3, int(shape.radius * scale))
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
            pygame.draw.circle(self.screen, COLORS['shape_border'], (int(x), int(y)), radius, 2)
            
        elif shape_type == 'TriangleShape':
            # Triangle - approximate with polygon
            size = shape.base_width * scale * 0.5
            points = [
                (x, y - size * 0.6),
                (x - size * 0.5, y + size * 0.4),
                (x + size * 0.5, y + size * 0.4)
            ]
            
            if shape.rotation != 0:
                # Rotate points around center
                angle_rad = np.radians(shape.rotation)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rotated_points = []
                for px, py in points:
                    rel_x, rel_y = px - x, py - y
                    new_x = rel_x * cos_a - rel_y * sin_a + x
                    new_y = rel_x * sin_a + rel_y * cos_a + y
                    rotated_points.append((new_x, new_y))
                points = rotated_points
            
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, COLORS['shape_border'], points, 2)
            
        else:
            # Generic shape (L-shape, irregular) - draw as bounding box
            bounds = shape.bounds
            width = (bounds[2] - bounds[0]) * scale
            height = (bounds[3] - bounds[1]) * scale
            
            rect = pygame.Rect(x - width/2, y - height/2, width, height)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, COLORS['shape_border'], rect, 2)
    
    def draw_shapes_panel(self):
        """Draw panel showing shapes to fit."""
        x = self.shapes_panel_x
        y = 50
        
        # Panel background
        panel_width = 250
        panel_height = 400
        pygame.draw.rect(self.screen, COLORS['info_bg'], (x - 10, y - 10, panel_width, panel_height))
        pygame.draw.rect(self.screen, COLORS['container'], (x - 10, y - 10, panel_width, panel_height), 2)
        
        title = self.title_font.render("Shapes to Fit", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        # Show fitted vs unfitted counts
        fitted_count = sum(1 for s in self.env.shapes_to_fit if s.is_fitted)
        unfitted_count = len(self.env.shapes_to_fit) - fitted_count
        
        fitted_text = self.font.render(f"✓ Fitted: {fitted_count}", True, COLORS['success'])
        self.screen.blit(fitted_text, (x, y))
        y += 25
        
        unfitted_text = self.font.render(f"⚬ Remaining: {unfitted_count}", True, COLORS['failure'])
        self.screen.blit(unfitted_text, (x, y))
        y += 40
        
        # Draw unfitted shapes in grid
        unfitted_shapes = [s for s in self.env.shapes_to_fit if not s.is_fitted]
        grid_cols = 3
        shape_size = 30
        
        for i, shape in enumerate(unfitted_shapes[:12]):  # Show first 12 unfitted shapes
            row = i // grid_cols
            col = i % grid_cols
            
            shape_x = x + col * (shape_size + 10) + shape_size // 2
            shape_y = y + row * (shape_size + 10) + shape_size // 2
            
            # Create a temporary positioned shape for drawing
            temp_shape = type(shape)(
                *[getattr(shape, attr) for attr in ['width', 'height'] if hasattr(shape, attr)] or
                 [getattr(shape, attr) for attr in ['radius'] if hasattr(shape, attr)] or  
                 [getattr(shape, attr) for attr in ['base_width', 'height'] if hasattr(shape, attr)] or
                 [getattr(shape, attr) for attr in ['arm_length', 'arm_width'] if hasattr(shape, attr)] or
                 [getattr(shape, 'vertices', [])],
                position=(shape_x / self.container_scale, shape_y / self.container_scale),
                rotation=0
            )
            temp_shape.is_fitted = False
            
            self.draw_shape(temp_shape, COLORS['unfitted_shape'], offset=(-self.container_offset[0], -self.container_offset[1]))
    
    def draw_info_panel(self, metrics=None, action=None):
        """Draw information panel."""
        x = self.info_panel_x
        y = 50
        
        # Panel background
        panel_width = 270
        panel_height = 450
        pygame.draw.rect(self.screen, COLORS['info_bg'], (x - 10, y - 10, panel_width, panel_height))
        pygame.draw.rect(self.screen, COLORS['container'], (x - 10, y - 10, panel_width, panel_height), 2)
        
        title = self.title_font.render("Agent Status", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 40
        
        episode_text = self.font.render(f"Episode: {self.episode_count}", True, COLORS['text'])
        self.screen.blit(episode_text, (x, y))
        y += 25
        
        step_text = self.font.render(f"Step: {self.step_count}", True, COLORS['text'])
        self.screen.blit(step_text, (x, y))
        y += 25
        
        difficulty_text = self.font.render(f"Difficulty: {self.env.difficulty_level}", True, COLORS['text'])
        self.screen.blit(difficulty_text, (x, y))
        y += 35
        
        if metrics:
            reward_text = self.font.render(f"Episode Reward: {metrics.get('episode_reward', 0):.1f}", True, COLORS['text'])
            self.screen.blit(reward_text, (x, y))
            y += 25
            
            shapes_fitted = self.font.render(f"Shapes Fitted: {metrics.get('shapes_fitted', 0)}/{self.env.num_shapes_to_fit}", True, COLORS['text'])
            self.screen.blit(shapes_fitted, (x, y))
            y += 25
            
            success_rate = self.font.render(f"Success Rate: {metrics.get('success_rate', 0):.1%}", True, COLORS['text'])
            self.screen.blit(success_rate, (x, y))
            y += 25
            
            util_text = self.font.render(f"Utilization: {metrics.get('utilization', 0):.1%}", True, COLORS['text'])
            self.screen.blit(util_text, (x, y))
            y += 35
        
        if action is not None:
            action_title = self.font.render("Last Action:", True, COLORS['text'])
            self.screen.blit(action_title, (x, y))
            y += 25
            
            shape_id = self.font.render(f"Shape ID: {int(action[0])}", True, COLORS['text'])
            self.screen.blit(shape_id, (x, y))
            y += 20
            
            position = self.font.render(f"Position: ({action[1]:.1f}, {action[2]:.1f})", True, COLORS['text'])
            self.screen.blit(position, (x, y))
            y += 20
            
            # Convert rotation index to angle
            rotation_idx = int(action[3])
            if 0 <= rotation_idx < len(self.env.rotation_angles):
                rotation_angle = self.env.rotation_angles[rotation_idx]
            else:
                rotation_angle = 0
            
            rotation = self.font.render(f"Rotation: {rotation_angle}°", True, COLORS['text'])
            self.screen.blit(rotation, (x, y))
            y += 30
        
        # Status message
        if self.status_message:
            status_text = self.font.render(self.status_message, True, self.status_color)
            self.screen.blit(status_text, (x, y))
            y += 30
        
        y += 10
        
        controls_title = self.font.render("Controls:", True, COLORS['text'])
        self.screen.blit(controls_title, (x, y))
        y += 25
        
        controls = ["SPACE: Pause/Resume", "R: Reset Episode", "Q: Quit", "UP/DOWN: Speed"]
        
        for control in controls:
            control_text = self.small_font.render(control, True, COLORS['text'])
            self.screen.blit(control_text, (x, y))
            y += 18
    
    def run_episode(self):
        """Run a single episode with visualization."""
        obs = self.env.reset()
        done = False
        self.step_count = 0
        last_action = None
        self.status_message = "Episode started"
        self.status_color = COLORS['text']
        
        while not done and self.running:
            self.handle_events()
            
            if not self.paused:
                # Get action from agent or random
                if self.agent:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                    with torch.no_grad():
                        action, _ = self.agent.network.get_action(obs_tensor, deterministic=True)
                    action_np = action.detach().cpu().numpy().squeeze()
                else:
                    # Create a reasonable random action
                    action_np = np.array([
                        np.random.randint(0, len(self.env.shapes_to_fit)),
                        np.random.uniform(10, 90),
                        np.random.uniform(10, 90),
                        np.random.randint(0, len(self.env.rotation_angles))
                    ])
                
                # Take action
                obs, reward, done, info = self.env.step(action_np)
                last_action = action_np
                self.step_count += 1
                
                # Update status message
                if info['placement'] == 'success':
                    self.status_message = f"✓ Shape fitted! +{reward:.1f}"
                    self.status_color = COLORS['success']
                elif info['placement'] == 'collision':
                    self.status_message = f"✗ Collision {reward:.1f}"
                    self.status_color = COLORS['failure']
                elif info['placement'] == 'already_fitted':
                    self.status_message = "⚠ Shape already fitted"
                    self.status_color = COLORS['failure']
                else:
                    self.status_message = f"Step {self.step_count}"
                    self.status_color = COLORS['text']
                
                # Draw everything
                self.screen.fill(COLORS['background'])
                self.draw_container()
                
                # Draw fitted shapes in container
                for shape in self.env.container.placed_shapes:
                    self.draw_shape(shape, COLORS['fitted_shape'])
                
                # Draw info panels
                metrics = self.env.get_metrics()
                self.draw_info_panel(metrics, last_action)
                self.draw_shapes_panel()
                
                pygame.display.flip()
                time.sleep(self.animation_speed)
            else:
                self.handle_events()
                time.sleep(0.1)
        
        # Episode finished
        final_metrics = self.env.get_metrics()
        print(f"Episode {self.episode_count} finished:")
        print(f"  Reward: {final_metrics['episode_reward']:.1f}")
        print(f"  Shapes Fitted: {final_metrics['shapes_fitted']}/{self.env.num_shapes_to_fit}")
        print(f"  Success Rate: {final_metrics['success_rate']:.1%}")
        print(f"  Utilization: {final_metrics['utilization']:.1%}")
        
        # Show final state
        self.screen.fill(COLORS['background'])
        self.draw_container()
        
        for shape in self.env.container.placed_shapes:
            self.draw_shape(shape, COLORS['fitted_shape'])
        
        self.draw_info_panel(final_metrics, last_action)
        self.draw_shapes_panel()
        
        # Show completion message
        if final_metrics['shapes_fitted'] == self.env.num_shapes_to_fit:
            self.status_message = "🎉 ALL SHAPES FITTED!"
            self.status_color = COLORS['success']
        else:
            self.status_message = f"Episode ended: {final_metrics['shapes_fitted']}/{self.env.num_shapes_to_fit} fitted"
            self.status_color = COLORS['text']
        
        self.draw_info_panel(final_metrics, last_action)
        pygame.display.flip()
        time.sleep(3)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")
                elif event.key == pygame.K_r:
                    print("Resetting episode...")
                    return True
                elif event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_UP:
                    self.animation_speed = max(0.1, self.animation_speed - 0.1)
                    print(f"Speed: {1/self.animation_speed:.1f}x")
                elif event.key == pygame.K_DOWN:
                    self.animation_speed = min(2.0, self.animation_speed + 0.1)
                    print(f"Speed: {1/self.animation_speed:.1f}x")
        return False
    
    def run(self, num_episodes=10):
        """Run the visualization for multiple episodes."""
        print("🎮 Starting Shape Fitting Visualization")
        print("Controls: SPACE=pause, R=reset, Q=quit, UP/DOWN=speed")
        
        for episode in range(num_episodes):
            if not self.running:
                break
                
            self.episode_count = episode + 1
            print(f"\n🎯 Starting Episode {self.episode_count}")
            
            self.run_episode()
        
        pygame.quit()
        print("👋 Visualization ended")

def main():
    """Main function to run the visualization."""
    print("🎮 Initializing Shape Fitting Visualization...")
    
    # Create environment
    env = ShapeFittingEnv(
        container_width=100,
        container_height=100,
        num_shapes_to_fit=8,
        difficulty_level=2,
        max_steps=40
    )
    
    # Try to create agent
    agent = None
    try:
        agent = PPOTrainer(
            env=env,
            obs_size=env.observation_space.shape[0],
            num_shapes=env.num_shapes_to_fit,
            num_rotations=len(env.rotation_angles)
        )
        print("✅ Agent created (untrained)")
    except Exception as e:
        print(f"⚠️ Could not create agent: {e}")
        print("Will use random actions instead")
    
    # Create and run visualizer
    visualizer = ShapeFittingVisualizer(env, agent)
    
    try:
        visualizer.run(num_episodes=50)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
