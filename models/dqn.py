import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Functional Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def dqn_agent_step(agent, state, action, reward, next_state, done):
    agent['memory'].append((state, action, reward, next_state, done))
    if len(agent['memory']) > agent['batch_size']:
        experiences = random.sample(agent['memory'], agent['batch_size'])
        dqn_learn(agent, experiences)

def dqn_learn(agent, experiences):
    states, actions, rewards, next_states, dones = zip(*experiences)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    Q_targets_next = agent['qnetwork_target'](next_states).detach().max(1)[0].unsqueeze(1)
    Q_targets = rewards + (agent['gamma'] * Q_targets_next * (1 - dones))
    Q_expected = agent['qnetwork_local'](states).gather(1, actions)

    loss = nn.MSELoss()(Q_expected, Q_targets)
    agent['optimizer'].zero_grad()
    loss.backward()
    agent['optimizer'].step()

    # Soft update
    for target_param, local_param in zip(agent['qnetwork_target'].parameters(), agent['qnetwork_local'].parameters()):
        target_param.data.copy_(agent['tau'] * local_param.data + (1.0 - agent['tau']) * target_param.data)

def create_dqn_agent(state_size, action_size, seed=0):
    qnetwork_local = QNetwork(state_size, action_size, seed)
    qnetwork_target = QNetwork(state_size, action_size, seed)
    optimizer = optim.Adam(qnetwork_local.parameters(), lr=5e-4)
    memory = deque(maxlen=10000)
    agent = {
        'qnetwork_local': qnetwork_local,
        'qnetwork_target': qnetwork_target,
        'optimizer': optimizer,
        'memory': memory,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 1e-3
    }
    return agent 