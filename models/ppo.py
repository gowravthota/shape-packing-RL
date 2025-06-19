import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x)

def ppo_update(agent, memory, eps_clip=0.2, gamma=0.99):
    states = torch.FloatTensor(np.array(memory['states']))
    actions = torch.LongTensor(np.array(memory['actions']))
    rewards = torch.FloatTensor(np.array(memory['rewards']))
    old_logprobs = torch.FloatTensor(np.array(memory['logprobs']))
    returns = compute_returns(rewards, gamma)

    for _ in range(agent['K_epochs']):
        logprobs, state_values = [], []
        for state, action in zip(states, actions):
            policy_logits, value = agent['model'](state)
            dist = torch.distributions.Categorical(logits=policy_logits)
            logprobs.append(dist.log_prob(action))
            state_values.append(value)
        logprobs = torch.stack(logprobs)
        state_values = torch.stack(state_values).squeeze()

        advantages = returns - state_values.detach()
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - state_values).pow(2).mean()

        agent['optimizer'].zero_grad()
        loss.backward()
        agent['optimizer'].step()

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.FloatTensor(returns)

def create_ppo_agent(state_dim, action_dim):
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    agent = {
        'model': model,
        'optimizer': optimizer,
        'K_epochs': 4
    }
    return agent 