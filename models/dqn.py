import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim

# Functional Q-network
class QNetwork(nn.Module):
    """Deep Q-Network with improved architecture and error handling."""
    
    def __init__(self, state_size: int, action_size: int, seed: int = 0, hidden_sizes: List[int] = None):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)

def dqn_agent_step(agent: Dict[str, Any], state: np.ndarray, action: int, 
                   reward: float, next_state: np.ndarray, done: bool) -> None:
    """Execute one step of the DQN agent."""
    try:
        agent['memory'].append((state, action, reward, next_state, done))
        
        if len(agent['memory']) > agent['batch_size']:
            experiences = random.sample(agent['memory'], agent['batch_size'])
            dqn_learn(agent, experiences)
    except Exception as e:
        print(f"Error in DQN agent step: {e}")

def dqn_learn(agent: Dict[str, Any], experiences: List[Tuple]) -> None:
    """Learn from a batch of experiences."""
    try:
        device = next(agent['qnetwork_local'].parameters()).device
        
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = agent['qnetwork_target'](next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (agent['gamma'] * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = agent['qnetwork_local'](states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize the loss
        agent['optimizer'].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent['qnetwork_local'].parameters(), 1.0)
        agent['optimizer'].step()

        # Soft update of target parameters
        soft_update(agent['qnetwork_target'], agent['qnetwork_local'], agent['tau'])
        
    except Exception as e:
        print(f"Error in DQN learning: {e}")

def soft_update(target_model: nn.Module, local_model: nn.Module, tau: float) -> None:
    """Soft update model parameters."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def create_dqn_agent(state_size: int, action_size: int, seed: int = 0, 
                     device: str = None) -> Dict[str, Any]:
    """Create a DQN agent with proper device management."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
    
    # Copy local to target
    qnetwork_target.load_state_dict(qnetwork_local.state_dict())
    
    optimizer = optim.Adam(qnetwork_local.parameters(), lr=5e-4)
    memory = deque(maxlen=int(1e5))
    
    agent = {
        'qnetwork_local': qnetwork_local,
        'qnetwork_target': qnetwork_target,
        'optimizer': optimizer,
        'memory': memory,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 1e-3,
        'device': device
    }
    
    return agent 