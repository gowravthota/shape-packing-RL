import pygame
import numpy as np
import torch
import sys
import time
from env import ContinuousContainerEnv
from agent import PPOTrainer

# Colors
COLORS = {
    'background': (240, 240, 240),
    'container': (50, 50, 50),
    'container_fill': (255, 255, 255),
    'shape_fill': (100, 150, 200),
    'shape_border': (50, 100, 150),
    'current_shape': (255, 100, 100),
    'text': (0, 0, 0),
    'success': (100, 200, 100),
    'failure': (200, 100, 100)
}

class AgentVisualizer:
    def __init__(self, env, agent=None, window_size=(1200, 800)):
        pygame.init()
        
        self.env = env
        self.agent = agent
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Space RL - Container Packing Agent")
        
        # Layout
        self.container_scale = 4
        self.container_offset = (50, 50)
        self.info_panel_x = 500
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        # Animation
        self.animation_speed = 0.5
        
        # Game state
        self.running = True
        self.paused = False
        self.episode_count = 0
        self.step_count = 0
        
    def draw_container(self):
        x, y = self.container_offset
        width = self.env.container_width * self.container_scale
        height = self.env.container_height * self.container_scale
        
        pygame.draw.rect(self.screen, COLORS['container_fill'], (x, y, width, height))
        pygame.draw.rect(self.screen, COLORS['container'], (x, y, width, height), 3)
        
    def draw_shape(self, shape, color=None):
        if color is None:
            color = COLORS['shape_fill']
            
        x_offset, y_offset = self.container_offset
        scale = self.container_scale
        
        x = shape.position[0] * scale + x_offset
        y = shape.position[1] * scale + y_offset
        
        if hasattr(shape, 'width') and hasattr(shape, 'height'):
            # Rectangle
            width = shape.width * scale
            height = shape.height * scale
            
            surf = pygame.Surface((width, height), pygame.SRCALPHA)
            surf.fill(color)
            
            if shape.rotation != 0:
                surf = pygame.transform.rotate(surf, -shape.rotation)
            
            rect = surf.get_rect()
            rect.center = (x, y)
            
            self.screen.blit(surf, rect)
            pygame.draw.rect(self.screen, COLORS['shape_border'], rect, 2)
            
        elif hasattr(shape, 'radius'):
            # Circle
            radius = shape.radius * scale
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
            pygame.draw.circle(self.screen, COLORS['shape_border'], (int(x), int(y)), int(radius), 2)
    
    def draw_info_panel(self, metrics=None, action=None):
        x = self.info_panel_x
        y = 50
        
        title = self.title_font.render("Agent Performance", True, COLORS['text'])
        self.screen.blit(title, (x, y))
        y += 50
        
        episode_text = self.font.render(f"Episode: {self.episode_count}", True, COLORS['text'])
        self.screen.blit(episode_text, (x, y))
        y += 30
        
        step_text = self.font.render(f"Step: {self.step_count}", True, COLORS['text'])
        self.screen.blit(step_text, (x, y))
        y += 30
        
        if metrics:
            reward_text = self.font.render(f"Episode Reward: {metrics.get('episode_reward', 0):.2f}", True, COLORS['text'])
            self.screen.blit(reward_text, (x, y))
            y += 25
            
            util_text = self.font.render(f"Utilization: {metrics.get('utilization', 0):.1%}", True, COLORS['text'])
            self.screen.blit(util_text, (x, y))
            y += 25
            
            shapes_placed = self.font.render(f"Shapes Placed: {metrics.get('shapes_placed', 0)}", True, COLORS['text'])
            self.screen.blit(shapes_placed, (x, y))
            y += 25
            
            shapes_remaining = self.font.render(f"Shapes Remaining: {metrics.get('shapes_remaining', 0)}", True, COLORS['text'])
            self.screen.blit(shapes_remaining, (x, y))
            y += 40
        
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
            
            rotation = self.font.render(f"Rotation: {action[3]:.1f}°", True, COLORS['text'])
            self.screen.blit(rotation, (x, y))
            y += 40
        
        controls_title = self.font.render("Controls:", True, COLORS['text'])
        self.screen.blit(controls_title, (x, y))
        y += 25
        
        controls = ["SPACE: Pause/Resume", "R: Reset Episode", "Q: Quit", "UP/DOWN: Speed"]
        
        for control in controls:
            control_text = self.font.render(control, True, COLORS['text'])
            self.screen.blit(control_text, (x, y))
            y += 20
    
    def run_episode(self):
        obs = self.env.reset()
        done = False
        self.step_count = 0
        last_action = None
        
        while not done and self.running:
            self.handle_events()
            
            if not self.paused:
                if self.agent:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                    with torch.no_grad():
                        action, _ = self.agent.network.get_action(obs_tensor, deterministic=True)
                    action_np = action.detach().cpu().numpy().squeeze()
                else:
                    action_np = self.env.action_space.sample()
                
                obs, reward, done, info = self.env.step(action_np)
                last_action = action_np
                self.step_count += 1
                
                self.screen.fill(COLORS['background'])
                self.draw_container()
                
                for shape in self.env.container.placed_shapes:
                    self.draw_shape(shape)
                
                if hasattr(self.env, 'current_shape') and self.env.current_shape:
                    self.draw_shape(self.env.current_shape, COLORS['current_shape'])
                
                metrics = self.env.get_metrics()
                self.draw_info_panel(metrics, last_action)
                
                if info.get('placement') == 'success':
                    result_text = self.font.render("✓ SUCCESSFUL PLACEMENT", True, COLORS['success'])
                elif info.get('placement') == 'collision':
                    result_text = self.font.render("✗ COLLISION", True, COLORS['failure'])
                else:
                    result_text = self.font.render("", True, COLORS['text'])
                
                self.screen.blit(result_text, (self.info_panel_x, 400))
                
                pygame.display.flip()
                time.sleep(self.animation_speed)
            else:
                self.handle_events()
                time.sleep(0.1)
        
        final_metrics = self.env.get_metrics()
        print(f"Episode {self.episode_count} finished:")
        print(f"  Reward: {final_metrics['episode_reward']:.2f}")
        print(f"  Utilization: {final_metrics['utilization']:.2%}")
        print(f"  Shapes Placed: {final_metrics['shapes_placed']}")
        
        self.screen.fill(COLORS['background'])
        self.draw_container()
        
        for shape in self.env.container.placed_shapes:
            self.draw_shape(shape)
        
        self.draw_info_panel(final_metrics, last_action)
        
        if final_metrics['shapes_remaining'] == 0:
            completion_text = self.title_font.render("EPISODE COMPLETED!", True, COLORS['success'])
        else:
            completion_text = self.title_font.render("EPISODE ENDED", True, COLORS['text'])
        
        self.screen.blit(completion_text, (self.info_panel_x, 450))
        pygame.display.flip()
        time.sleep(2)
    
    def handle_events(self):
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
        print("🎮 Starting Agent Visualization")
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
    print("🎮 Initializing Container Packing Visualization...")
    
    env = ContinuousContainerEnv(
        container_width=100,
        container_height=100,
        curriculum_level=1,
        max_shapes=15
    )
    
    agent = None
    try:
        agent = PPOTrainer(
            env=env,
            obs_size=env.observation_space.shape[0],
            max_shapes=15
        )
        print("✅ Agent created (untrained)")
    except Exception as e:
        print(f"⚠️  Could not create agent: {e}")
        print("Will use random actions instead")
    
    visualizer = AgentVisualizer(env, agent)
    
    try:
        visualizer.run(num_episodes=50)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
