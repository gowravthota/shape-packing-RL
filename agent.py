

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

class ContinuousActorCritic(nn.Module):
    """
    Actor-Critic network for continuous action space container packing.
    
    Actions: [shape_id (discrete), x (continuous), y (continuous), rotation (continuous)]
    """
    
    def __init__(self, obs_size: int, max_shapes: int = 20, hidden_size: int = 512):
        super().__init__()
        
        self.obs_size = obs_size
        self.max_shapes = max_shapes
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor heads
        # Discrete action: shape selection
        self.shape_selector = nn.Linear(hidden_size, max_shapes)
        
        # Continuous actions: position and rotation
        self.position_x_mean = nn.Linear(hidden_size, 1)
        self.position_x_std = nn.Linear(hidden_size, 1)
        
        self.position_y_mean = nn.Linear(hidden_size, 1)
        self.position_y_std = nn.Linear(hidden_size, 1)
        
        self.rotation_mean = nn.Linear(hidden_size, 1)
        self.rotation_std = nn.Linear(hidden_size, 1)
        
        # Critic head
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Returns:
            shape_logits: Logits for shape selection
            position_params: (x_mean, x_std, y_mean, y_std)
            rotation_params: (mean, std)
            value: State value
        """
        # Shared features
        features = self.shared_net(obs)
        
        # Shape selection (discrete)
        shape_logits = self.shape_selector(features)
        
        # Position (continuous)
        x_mean = torch.sigmoid(self.position_x_mean(features)) * 100.0  # Scale to container size
        x_std = F.softplus(self.position_x_std(features)) + 1e-5
        
        y_mean = torch.sigmoid(self.position_y_mean(features)) * 100.0
        y_std = F.softplus(self.position_y_std(features)) + 1e-5
        
        # Rotation (continuous, 0-360 degrees)
        rotation_mean = torch.sigmoid(self.rotation_mean(features)) * 360.0
        rotation_std = F.softplus(self.rotation_std(features)) + 1e-5
        
        # Value
        value = self.value_head(features)
        
        return shape_logits, (x_mean, x_std, y_mean, y_std), (rotation_mean, rotation_std), value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Returns:
            action: [shape_id, x, y, rotation]
            log_prob: Log probability of the action
        """
        shape_logits, pos_params, rot_params, _ = self.forward(obs)
        
        # Shape selection (discrete)
        shape_dist = Categorical(logits=shape_logits)
        if deterministic:
            shape_action = torch.argmax(shape_logits, dim=-1)
        else:
            shape_action = shape_dist.sample()
        shape_log_prob = shape_dist.log_prob(shape_action)
        
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
        x_action = torch.clamp(x_action, 0, 100)
        y_action = torch.clamp(y_action, 0, 100)
        
        x_log_prob = x_dist.log_prob(x_action)
        y_log_prob = y_dist.log_prob(y_action)
        
        # Rotation (continuous)
        rot_mean, rot_std = rot_params
        rot_dist = Normal(rot_mean, rot_std)
        
        if deterministic:
            rot_action = rot_mean
        else:
            rot_action = rot_dist.sample()
        
        rot_action = torch.clamp(rot_action, 0, 360)
        rot_log_prob = rot_dist.log_prob(rot_action)
        
        # Combine actions
        action = torch.stack([shape_action.float(), x_action.squeeze(), y_action.squeeze(), rot_action.squeeze()], dim=-1)
        total_log_prob = shape_log_prob + x_log_prob.squeeze() + y_log_prob.squeeze() + rot_log_prob.squeeze()
        
        return action, total_log_prob
    
    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probability and entropy of given actions.
        
        Returns:
            log_prob: Log probability of the actions
            entropy: Entropy of the policy
            value: State value
        """
        shape_logits, pos_params, rot_params, value = self.forward(obs)
        
        # Parse actions
        shape_action = action[:, 0].long()
        x_action = action[:, 1]
        y_action = action[:, 2]
        rot_action = action[:, 3]
        
        # Shape selection
        shape_dist = Categorical(logits=shape_logits)
        shape_log_prob = shape_dist.log_prob(shape_action)
        shape_entropy = shape_dist.entropy()
        
        # Position
        x_mean, x_std, y_mean, y_std = pos_params
        x_dist = Normal(x_mean.squeeze(), x_std.squeeze())
        y_dist = Normal(y_mean.squeeze(), y_std.squeeze())
        
        x_log_prob = x_dist.log_prob(x_action)
        y_log_prob = y_dist.log_prob(y_action)
        x_entropy = x_dist.entropy()
        y_entropy = y_dist.entropy()
        
        # Rotation
        rot_mean, rot_std = rot_params
        rot_dist = Normal(rot_mean.squeeze(), rot_std.squeeze())
        
        rot_log_prob = rot_dist.log_prob(rot_action)
        rot_entropy = rot_dist.entropy()
        
        # Combine
        total_log_prob = shape_log_prob + x_log_prob + y_log_prob + rot_log_prob
        total_entropy = shape_entropy + x_entropy + y_entropy + rot_entropy
        
        return total_log_prob, total_entropy, value.squeeze()

class PPOTrainer:
    """PPO trainer for continuous container packing."""
    
    def __init__(self, env, obs_size: int, max_shapes: int = 20, lr: float = 3e-4, 
                 clip_ratio: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create network
        self.network = ContinuousActorCritic(obs_size, max_shapes).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO parameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Training parameters
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.utilization_scores = []
        self.success_rates = []
        
    def collect_trajectories(self, n_steps: int = 2048) -> Dict[str, torch.Tensor]:
        """Collect trajectories for training."""
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        
        obs = self.env.reset()
        
        for step in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action and value
            with torch.no_grad():
                action, log_prob = self.network.get_action(obs_tensor)
                _, _, _, value = self.network.forward(obs_tensor)
            
            # Take action in environment
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store trajectory data
            observations.append(obs)
            actions.append(action.cpu().numpy().squeeze())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy().squeeze())
            
            obs = next_obs
            
            if done:
                # Episode ended, collect metrics
                metrics = self.env.get_metrics()
                self.episode_rewards.append(metrics['episode_reward'])
                self.episode_lengths.append(metrics['episode_length'])
                self.utilization_scores.append(metrics['utilization'])
                self.success_rates.append(1.0 if metrics['shapes_remaining'] == 0 else 0.0)
                
                obs = self.env.reset()
        
        # Convert to tensors
        return {
            'observations': torch.FloatTensor(np.array(observations)).to(self.device),
            'actions': torch.FloatTensor(np.array(actions)).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'dones': torch.BoolTensor(dones).to(self.device),
            'log_probs': torch.FloatTensor(log_probs).to(self.device),
            'values': torch.FloatTensor(values).to(self.device)
        }
    
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
                next_value = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the policy using PPO."""
        observations = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        rewards = batch['rewards']
        dones = batch['dones']
        old_values = batch['values']
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(observations))
            
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy, values = self.network.evaluate_action(batch_obs, batch_actions)
                
                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute policy loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_loss_total += entropy_loss.item()
        
        return {
            'total_loss': total_loss / (self.n_epochs * len(observations) // self.batch_size),
            'policy_loss': policy_loss_total / (self.n_epochs * len(observations) // self.batch_size),
            'value_loss': value_loss_total / (self.n_epochs * len(observations) // self.batch_size),
            'entropy_loss': entropy_loss_total / (self.n_epochs * len(observations) // self.batch_size)
        }
    
    def train(self, total_steps: int = 1000000, eval_freq: int = 10000, save_freq: int = 50000):
        """Train the PPO agent."""
        print(f"Training PPO agent on {self.device}")
        print(f"Total steps: {total_steps}")
        
        step = 0
        while step < total_steps:
            # Collect trajectories
            print(f"\nStep {step}: Collecting trajectories...")
            batch = self.collect_trajectories()
            
            # Update policy
            print("Updating policy...")
            losses = self.update_policy(batch)
            
            step += len(batch['rewards'])
            
            # Print progress
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                recent_utilization = self.utilization_scores[-10:] if len(self.utilization_scores) >= 10 else self.utilization_scores
                recent_success = self.success_rates[-10:] if len(self.success_rates) >= 10 else self.success_rates
                
                print(f"Step {step}:")
                print(f"  Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
                print(f"  Avg Utilization (last 10): {np.mean(recent_utilization):.2%}")
                print(f"  Success Rate (last 10): {np.mean(recent_success):.2%}")
                print(f"  Policy Loss: {losses['policy_loss']:.4f}")
                print(f"  Value Loss: {losses['value_loss']:.4f}")
                print(f"  Entropy Loss: {losses['entropy_loss']:.4f}")
            
            # Evaluation
            if step % eval_freq == 0:
                self.evaluate()
            
            # Save model
            if step % save_freq == 0:
                self.save_model(f"ppo_container_step_{step}.pt")
        
        print("Training completed!")
        self.save_model("ppo_container_final.pt")
        self.plot_training_curves()
    
    def evaluate(self, n_episodes: int = 5):
        """Evaluate the current policy."""
        print(f"\nEvaluating policy for {n_episodes} episodes...")
        
        eval_rewards = []
        eval_utilizations = []
        eval_successes = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _ = self.network.get_action(obs_tensor, deterministic=True)
                
                action_np = action.cpu().numpy().squeeze()
                obs, reward, done, info = self.env.step(action_np)
            
            metrics = self.env.get_metrics()
            eval_rewards.append(metrics['episode_reward'])
            eval_utilizations.append(metrics['utilization'])
            eval_successes.append(1.0 if metrics['shapes_remaining'] == 0 else 0.0)
        
        print(f"Evaluation Results:")
        print(f"  Avg Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Avg Utilization: {np.mean(eval_utilizations):.2%}")
        print(f"  Success Rate: {np.mean(eval_successes):.2%}")
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'utilization_scores': self.utilization_scores,
            'success_rates': self.success_rates
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.utilization_scores = checkpoint.get('utilization_scores', [])
        self.success_rates = checkpoint.get('success_rates', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_curves(self):
        """Plot training progress curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Utilization scores
        axes[0, 1].plot(self.utilization_scores)
        axes[0, 1].set_title('Container Utilization')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Utilization %')
        
        # Success rates (moving average)
        if len(self.success_rates) > 10:
            window_size = 50
            success_ma = np.convolve(self.success_rates, np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(success_ma)
            axes[1, 0].set_title(f'Success Rate (MA {window_size})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
        
        # Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig('ppo_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training curves saved to ppo_training_curves.png") 