import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with improved architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super(ActorCritic, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        
        # Shared layers
        shared_layers = []
        input_dim = state_dim
        
        for hidden_size in hidden_sizes:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_size
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Separate heads for policy and value
        self.policy_head = nn.Linear(input_dim, action_dim)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and state value."""
        shared_features = self.shared(x)
        policy_logits = self.policy_head(shared_features)
        state_value = self.value_head(shared_features)
        return policy_logits, state_value

def ppo_update(agent: Dict[str, Any], memory: Dict[str, List], 
               eps_clip: float = 0.2, gamma: float = 0.99) -> None:
    """Update PPO agent with collected experiences."""
    try:
        device = next(agent['model'].parameters()).device
        
        states = torch.FloatTensor(np.array(memory['states'])).to(device)
        actions = torch.LongTensor(np.array(memory['actions'])).to(device)
        rewards = torch.FloatTensor(np.array(memory['rewards'])).to(device)
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs'])).to(device)
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)

        for _ in range(agent['K_epochs']):
            # Get current policy and value predictions
            policy_logits, state_values = agent['model'](states)
            
            # Create categorical distribution and compute log probabilities
            dist = torch.distributions.Categorical(logits=policy_logits)
            logprobs = dist.log_prob(actions)
            
            # Compute advantages
            advantages = returns - state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute PPO loss
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - state_values.squeeze()).pow(2).mean()
            entropy_loss = -dist.entropy().mean()
            
            total_loss = policy_loss + value_loss + 0.01 * entropy_loss

            # Update model
            agent['optimizer'].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent['model'].parameters(), 0.5)
            agent['optimizer'].step()
            
    except Exception as e:
        print(f"Error in PPO update: {e}")

def compute_returns(rewards: torch.Tensor, gamma: float) -> List[float]:
    """Compute discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards.tolist()):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def create_ppo_agent(state_dim: int, action_dim: int, device: str = None) -> Dict[str, Any]:
    """Create a PPO agent with proper device management."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    agent = {
        'model': model,
        'optimizer': optimizer,
        'K_epochs': 4,
        'device': device
    }
    
    return agent 