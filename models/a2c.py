import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any

class A2CNet(nn.Module):
    """Actor-Critic network for A2C with improved architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super(A2CNet, self).__init__()
        
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

def a2c_update(agent: Dict[str, Any], memory: Dict[str, List], gamma: float = 0.99) -> None:
    """Update A2C agent with collected experiences."""
    try:
        device = next(agent['model'].parameters()).device
        
        states = torch.FloatTensor(np.array(memory['states'])).to(device)
        actions = torch.LongTensor(np.array(memory['actions'])).to(device)
        rewards = torch.FloatTensor(np.array(memory['rewards'])).to(device)
        next_states = torch.FloatTensor(np.array(memory['next_states'])).to(device)
        dones = torch.FloatTensor(np.array(memory['dones'])).to(device)

        # Get current policy and value predictions
        policy_logits, state_values = agent['model'](states)
        
        # Create categorical distribution and compute log probabilities
        dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        
        # Compute next state values
        with torch.no_grad():
            _, next_state_values = agent['model'](next_states)
        
        # Compute TD targets and advantages
        td_targets = rewards + gamma * next_state_values.squeeze() * (1 - dones)
        advantages = td_targets - state_values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * advantages.pow(2).mean()
        entropy_loss = -dist.entropy().mean()
        
        total_loss = policy_loss + value_loss + 0.01 * entropy_loss

        # Update model
        agent['optimizer'].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent['model'].parameters(), 0.5)
        agent['optimizer'].step()
        
    except Exception as e:
        print(f"Error in A2C update: {e}")

def create_a2c_agent(state_dim: int, action_dim: int, device: str = None) -> Dict[str, Any]:
    """Create an A2C agent with proper device management."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = A2CNet(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    agent = {
        'model': model,
        'optimizer': optimizer,
        'device': device
    }
    
    return agent 