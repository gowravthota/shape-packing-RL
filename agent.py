import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from typing import List, Tuple, Dict, Any
import random
from collections import deque
import matplotlib.pyplot as plt

class ShapeFittingActorCritic(nn.Module):
    """
    Actor-Critic network for shape fitting with discrete rotation.
    
    Actions: [shape_id (discrete), x (continuous), y (continuous), rotation_idx (discrete)]
    """
    
    def __init__(self, obs_size: int, num_shapes: int = 10, num_rotations: int = 18, hidden_size: int = 512,
                 max_x: float = 100.0, max_y: float = 100.0):
        super().__init__()
        
        self.obs_size = obs_size
        self.num_shapes = num_shapes
        self.num_rotations = num_rotations
        self.max_x = float(max_x)
        self.max_y = float(max_y)
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor heads
        # Discrete actions: shape selection and rotation
        self.shape_selector = nn.Linear(hidden_size, num_shapes)
        self.rotation_selector = nn.Linear(hidden_size, num_rotations)
        
        # Continuous actions: position
        self.position_x_mean = nn.Linear(hidden_size, 1)
        self.position_x_std = nn.Linear(hidden_size, 1)
        
        self.position_y_mean = nn.Linear(hidden_size, 1)
        self.position_y_std = nn.Linear(hidden_size, 1)
        
        # Critic head
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Returns:
            shape_logits: Logits for shape selection
            rotation_logits: Logits for rotation selection  
            position_params: (x_mean, x_std, y_mean, y_std)
            value: State value
        """
        # Shared features
        features = self.shared_net(obs)
        
        # Shape selection (discrete)
        shape_logits = self.shape_selector(features)
        
        # Rotation selection (discrete)
        rotation_logits = self.rotation_selector(features)
        
        # Position (continuous)
        x_mean = torch.sigmoid(self.position_x_mean(features)) * self.max_x  # Scale to container size
        x_std = F.softplus(self.position_x_std(features)) + 1e-5
        
        y_mean = torch.sigmoid(self.position_y_mean(features)) * self.max_y
        y_std = F.softplus(self.position_y_std(features)) + 1e-5
        
        # Value
        value = self.value_head(features)
        
        return shape_logits, rotation_logits, (x_mean, x_std, y_mean, y_std), value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Returns:
            action: [shape_id, x, y, rotation_idx]
            log_prob: Log probability of the action
        """
        shape_logits, rotation_logits, pos_params, _ = self.forward(obs)
        
        # Shape selection (discrete)
        shape_dist = Categorical(logits=shape_logits)
        if deterministic:
            shape_action = torch.argmax(shape_logits, dim=-1)
        else:
            shape_action = shape_dist.sample()
        shape_log_prob = shape_dist.log_prob(shape_action)
        
        # Rotation selection (discrete)
        rotation_dist = Categorical(logits=rotation_logits)
        if deterministic:
            rotation_action = torch.argmax(rotation_logits, dim=-1)
        else:
            rotation_action = rotation_dist.sample()
        rotation_log_prob = rotation_dist.log_prob(rotation_action)
        
        # Position (continuous)
        x_mean, x_std, y_mean, y_std = pos_params
        x_dist = Normal(x_mean, x_std)
        y_dist = Normal(y_mean, y_std)
        
        if deterministic:
            x_action = x_mean
            y_action = y_mean
        else:
            x_action = x_dist.sample()
            y_action = y_dist.sample()
        
        # Clamp to bounds
        x_action = torch.clamp(x_action, 0, self.max_x)
        y_action = torch.clamp(y_action, 0, self.max_y)
        
        x_log_prob = x_dist.log_prob(x_action)
        y_log_prob = y_dist.log_prob(y_action)
        
        # Ensure all tensors have consistent shape for stacking
        shape_action_tensor = shape_action.float()
        if shape_action_tensor.dim() == 0:
            shape_action_tensor = shape_action_tensor.unsqueeze(0)
        
        x_tensor = x_action.squeeze()
        if x_tensor.dim() == 0:
            x_tensor = x_tensor.unsqueeze(0)
            
        y_tensor = y_action.squeeze()
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)
            
        rotation_tensor = rotation_action.float()
        if rotation_tensor.dim() == 0:
            rotation_tensor = rotation_tensor.unsqueeze(0)
        
        # Combine actions
        action = torch.stack([shape_action_tensor, x_tensor, y_tensor, rotation_tensor], dim=-1)
        
        # Ensure log probs have consistent shape
        shape_log_prob_tensor = shape_log_prob
        if shape_log_prob_tensor.dim() == 0:
            shape_log_prob_tensor = shape_log_prob_tensor.unsqueeze(0)
            
        x_log_prob_tensor = x_log_prob.squeeze()
        if x_log_prob_tensor.dim() == 0:
            x_log_prob_tensor = x_log_prob_tensor.unsqueeze(0)
            
        y_log_prob_tensor = y_log_prob.squeeze()
        if y_log_prob_tensor.dim() == 0:
            y_log_prob_tensor = y_log_prob_tensor.unsqueeze(0)
            
        rotation_log_prob_tensor = rotation_log_prob
        if rotation_log_prob_tensor.dim() == 0:
            rotation_log_prob_tensor = rotation_log_prob_tensor.unsqueeze(0)
        
        total_log_prob = shape_log_prob_tensor + x_log_prob_tensor + y_log_prob_tensor + rotation_log_prob_tensor
        
        return action, total_log_prob
    
    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probability and entropy of given actions.
        
        Returns:
            log_prob: Log probability of the actions
            entropy: Entropy of the policy
            value: State value
        """
        shape_logits, rotation_logits, pos_params, value = self.forward(obs)
        
        # Parse actions
        shape_action = action[:, 0].long()
        x_action = action[:, 1]
        y_action = action[:, 2]
        rotation_action = action[:, 3].long()
        
        # Shape selection
        shape_dist = Categorical(logits=shape_logits)
        shape_log_prob = shape_dist.log_prob(shape_action)
        shape_entropy = shape_dist.entropy()
        
        # Rotation selection
        rotation_dist = Categorical(logits=rotation_logits)
        rotation_log_prob = rotation_dist.log_prob(rotation_action)
        rotation_entropy = rotation_dist.entropy()
        
        # Position
        x_mean, x_std, y_mean, y_std = pos_params
        x_dist = Normal(x_mean, x_std)
        y_dist = Normal(y_mean, y_std)
        
        x_log_prob = x_dist.log_prob(x_action)
        y_log_prob = y_dist.log_prob(y_action)
        
        x_entropy = x_dist.entropy()
        y_entropy = y_dist.entropy()
        
        # Combine log probabilities and entropies
        total_log_prob = shape_log_prob + rotation_log_prob + x_log_prob + y_log_prob
        total_entropy = shape_entropy + rotation_entropy + x_entropy + y_entropy
        
        return total_log_prob, total_entropy, value.squeeze()

class PPOTrainer:
    """PPO trainer for shape fitting environment."""
    
    def __init__(self, env, obs_size: int, num_shapes: int = 10, num_rotations: int = 18,
                 lr: float = 3e-4, clip_ratio: float = 0.2, value_coef: float = 0.5, 
                 entropy_coef: float = 0.01, gamma: float = 0.99, gae_lambda: float = 0.95):
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Networks
        self.network = ShapeFittingActorCritic(
            obs_size,
            num_shapes,
            num_rotations,
            hidden_size=512,
            max_x=getattr(env, 'container_width', 100.0),
            max_y=getattr(env, 'container_height', 100.0),
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_shapes_fitted = deque(maxlen=100)
        self.episode_success_rates = deque(maxlen=100)
        self.utilization_scores = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_losses = []
        
        print(f"PPO Trainer initialized on {self.device}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
    
    def collect_trajectories(self, n_steps: int = 2048) -> Dict[str, torch.Tensor]:
        """Collect trajectories from the environment."""
        
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        obs = self.env.reset()
        episode_reward = 0
        episode_shapes_fitted = 0
        episode_length = 0
        
        for step in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.network.get_action(obs_tensor)
                _, _, _, value = self.network.forward(obs_tensor)
            
            # Store trajectory data
            observations.append(obs)
            actions.append(action.cpu().numpy().squeeze())
            log_probs.append(log_prob.cpu().numpy().squeeze())
            values.append(value.cpu().item())
            
            # Take action in environment
            action_np = action.detach().cpu().numpy().squeeze()
            next_obs, reward, done, info = self.env.step(action_np)

            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            if 'shapes_fitted' in info:
                episode_shapes_fitted = info['shapes_fitted']

            episode_length += 1
            
            obs = next_obs
            
            if done:
                # Episode finished
                self.episode_rewards.append(episode_reward)
                self.episode_shapes_fitted.append(episode_shapes_fitted)
                self.episode_success_rates.append(episode_shapes_fitted / self.env.num_shapes_to_fit)
                self.utilization_scores.append(self.env.container.utilization)
                self.episode_lengths.append(episode_length)

                obs = self.env.reset()
                episode_reward = 0
                episode_shapes_fitted = 0
                episode_length = 0
        
        # Convert to tensors
        batch = {
            'observations': torch.FloatTensor(np.array(observations)).to(self.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(log_probs)).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'dones': torch.BoolTensor(dones).to(self.device),
            'values': torch.FloatTensor(values).to(self.device)
        }
        
        return batch
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update_policy(self, batch: Dict[str, torch.Tensor], n_epochs: int = 10, batch_size: int = 256) -> Dict[str, float]:
        """Update the policy using PPO."""
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(batch['rewards'], batch['values'], batch['dones'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare data
        n_samples = len(batch['observations'])
        indices = np.arange(n_samples)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                obs_batch = batch['observations'][batch_indices]
                actions_batch = batch['actions'][batch_indices]
                old_log_probs_batch = batch['log_probs'][batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy, values = self.network.evaluate_action(obs_batch, actions_batch)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns_batch)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # Store training metrics
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates
        
        self.training_losses.append({
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + self.value_coef * avg_value_loss + self.entropy_coef * avg_entropy_loss
        })
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_shapes_fitted': np.mean(self.episode_shapes_fitted) if self.episode_shapes_fitted else 0,
            'avg_success_rate': np.mean(self.episode_success_rates) if self.episode_success_rates else 0
        }
    
    def train(self, total_steps: int = 1000000, eval_freq: int = 10000, save_freq: int = 50000):
        """Train the agent."""
        
        print(f"Starting training for {total_steps:,} steps")
        print(f"Evaluation frequency: {eval_freq:,} steps")
        print(f"Save frequency: {save_freq:,} steps")
        
        steps_completed = 0
        
        while steps_completed < total_steps:
            # Collect trajectories
            print(f"\nCollecting trajectories... (Step {steps_completed:,}/{total_steps:,})")
            batch = self.collect_trajectories(n_steps=2048)
            
            # Update policy
            print("Updating policy...")
            metrics = self.update_policy(batch)
            
            steps_completed += 2048
            
            # Print progress
            print(f"Step {steps_completed:,}/{total_steps:,}")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print(f"  Avg Shapes Fitted: {metrics['avg_shapes_fitted']:.1f}/{self.env.num_shapes_to_fit}")
            print(f"  Avg Success Rate: {metrics['avg_success_rate']:.2%}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            
            # Evaluation
            if steps_completed % eval_freq == 0:
                print("\nRunning evaluation...")
                eval_metrics = self.evaluate(n_episodes=5)
                print(f"  Eval Avg Reward: {eval_metrics['avg_reward']:.2f}")
                print(f"  Eval Avg Shapes Fitted: {eval_metrics['avg_shapes_fitted']:.1f}")
                print(f"  Eval Success Rate: {eval_metrics['success_rate']:.2%}")
            
            # Save model
            if steps_completed % save_freq == 0:
                model_path = f"models/shape_fitting_model_{steps_completed}.pth"
                self.save_model(model_path)
                print(f"Model saved to {model_path}")
        
        print("\nTraining completed!")
        
    def evaluate(self, n_episodes: int = 5):
        """Evaluate the current policy."""
        
        self.network.eval()
        
        episode_rewards = []
        episode_shapes_fitted = []
        episode_success_rates = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _ = self.network.get_action(obs_tensor, deterministic=True)
                
                action_np = action.detach().cpu().numpy().squeeze()
                obs, reward, done, info = self.env.step(action_np)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            shapes_fitted = info.get('shapes_fitted', 0)
            episode_shapes_fitted.append(shapes_fitted)
            episode_success_rates.append(shapes_fitted / self.env.num_shapes_to_fit)
        
        self.network.train()
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'avg_shapes_fitted': np.mean(episode_shapes_fitted),
            'success_rate': np.mean(episode_success_rates),
            'episode_rewards': episode_rewards
        }
    
    def save_model(self, filepath: str):
        """Save the model."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'episode_rewards': list(self.episode_rewards),
            'episode_shapes_fitted': list(self.episode_shapes_fitted),
            'episode_success_rates': list(self.episode_success_rates)
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        if 'episode_shapes_fitted' in checkpoint:
            self.episode_shapes_fitted = deque(checkpoint['episode_shapes_fitted'], maxlen=100)
        if 'episode_success_rates' in checkpoint:
            self.episode_success_rates = deque(checkpoint['episode_success_rates'], maxlen=100)
    
    def plot_training_curves(self):
        """Plot training progress."""
        if not self.training_losses:
            print("No training data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        losses = self.training_losses
        ax1.plot([l['policy_loss'] for l in losses], label='Policy Loss')
        ax1.plot([l['value_loss'] for l in losses], label='Value Loss')
        ax1.plot([l['entropy_loss'] for l in losses], label='Entropy Loss')
        ax1.set_title('Training Losses')
        ax1.set_xlabel('Update')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Reward curve
        if self.episode_rewards:
            ax2.plot(list(self.episode_rewards))
            ax2.set_title('Episode Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.grid(True)
        
        # Shapes fitted curve
        if self.episode_shapes_fitted:
            ax3.plot(list(self.episode_shapes_fitted))
            ax3.set_title('Shapes Fitted per Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Shapes Fitted')
            ax3.grid(True)
        
        # Success rate curve
        if self.episode_success_rates:
            ax4.plot(list(self.episode_success_rates))
            ax4.set_title('Success Rate per Episode')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Success Rate')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

# Legacy compatibility
ContinuousActorCritic = ShapeFittingActorCritic 