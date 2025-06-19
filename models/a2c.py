import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A2CNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2CNet, self).__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x)

def a2c_update(agent, memory, gamma=0.99):
    states = torch.FloatTensor(np.array(memory['states']))
    actions = torch.LongTensor(np.array(memory['actions']))
    rewards = torch.FloatTensor(np.array(memory['rewards']))
    next_states = torch.FloatTensor(np.array(memory['next_states']))
    dones = torch.FloatTensor(np.array(memory['dones']))

    policy_logits, state_values = agent['model'](states)
    dist = torch.distributions.Categorical(logits=policy_logits)
    log_probs = dist.log_prob(actions)
    _, next_state_values = agent['model'](next_states)
    returns = rewards + gamma * next_state_values.squeeze() * (1 - dones)
    advantages = returns - state_values.squeeze()

    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = advantages.pow(2).mean()
    loss = policy_loss + 0.5 * value_loss

    agent['optimizer'].zero_grad()
    loss.backward()
    agent['optimizer'].step()

def create_a2c_agent(state_dim, action_dim):
    model = A2CNet(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = {
        'model': model,
        'optimizer': optimizer
    }
    return agent 