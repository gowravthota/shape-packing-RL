import math
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

from constraints import Shape, Container

def _patched_contains(self, shape: Shape) -> bool:
    left, top, right, bottom = shape.bounding_box()
    return (
        left >= self.x and right <= self.x + self.width and top >= self.y and bottom <= self.y + self.height
    )

Container.contains = _patched_contains

# ────────────────────────────────────────────────────────────────────────────────
# Gym environment that treats the 2‑D packing task as a discrete placement game.
# ────────────────────────────────────────────────────────────────────────────────

GRID_CELLS = 10  # resolution of the placement grid (10×10 → 100 possible positions)
CONTAINER_W = 100
CONTAINER_H = 100
CELL_SIZE = CONTAINER_W / GRID_CELLS  # square cells → CELL_SIZE×CELL_SIZE pixels

class ContainerEnv(gym.Env):
    """Simplified container‑packing environment with a discrete action space.

    Each action chooses a grid cell in which we *attempt* to drop a square shape
    of side length *CELL_SIZE*. The environment rewards successful placements
    (+1) and penalises invalid placements (‑1). The episode terminates once the
    grid is full or after *max_steps* actions, whichever comes first.
    """

    metadata = {"render.modes": []}

    def __init__(self, max_steps: int | None = None):
        super().__init__()
        self.max_steps = max_steps or GRID_CELLS * GRID_CELLS
        self.action_space = spaces.Discrete(GRID_CELLS * GRID_CELLS)  # choose a single cell
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(GRID_CELLS, GRID_CELLS),
            dtype=np.int8,
        )
        self._container: Container | None = None
        self._grid: np.ndarray | None = None
        self._steps = 0

    # Helper converting a discrete action index → (x, y) pixel coordinates.
    def _idx_to_xy(self, idx: int) -> Tuple[float, float]:
        row, col = divmod(idx, GRID_CELLS)
        x = col * CELL_SIZE + CELL_SIZE / 2.0
        y = row * CELL_SIZE + CELL_SIZE / 2.0
        return x, y

    def reset(self, *, seed: int | None = None, options=None):  # noqa: D401
        super().reset(seed=seed)
        self._container = Container(0, 0, CONTAINER_W, CONTAINER_H)
        self._grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.int8)
        self._steps = 0
        return self._grid.copy(), {}

    def step(self, action: int):  # noqa: D401
        assert self._grid is not None and self._container is not None, "Call reset() first!"
        self._steps += 1
        row, col = divmod(action, GRID_CELLS)

        reward = 0.0
        done = False
        info = {}

        if self._grid[row, col] == 1:
            # Cell already occupied → invalid action.
            reward = -1.0
        else:
            x, y = self._idx_to_xy(action)
            shape = Shape(x, y, size=CELL_SIZE)
            if self._container.add_shape(shape):
                # Successful placement.
                self._grid[row, col] = 1
                reward = 1.0
            else:
                reward = -1.0  # Out of bounds.

        # Termination: grid completely filled OR too many steps.
        filled = int(self._grid.sum())
        if filled == GRID_CELLS * GRID_CELLS or self._steps >= self.max_steps:
            done = True
            info["filled_cells"] = filled

        return self._grid.copy(), reward, done, False, info

# ────────────────────────────────────────────────────────────────────────────────
# Deep Q‑Learning implementation (vanilla DQN) — small, from‑scratch PyTorch.
# ────────────────────────────────────────────────────────────────────────────────

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        flat = GRID_CELLS * GRID_CELLS
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, GRID_CELLS * GRID_CELLS),
        )

    def forward(self, x):  # noqa: D401
        return self.layers(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):  # noqa: D401
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size: int):  # noqa: D401
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sp, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=dev),
            torch.tensor(a, dtype=torch.int64, device=dev),
            torch.tensor(r, dtype=torch.float32, device=dev),
            torch.tensor(np.array(sp), dtype=torch.float32, device=dev),
            torch.tensor(d, dtype=torch.bool, device=dev),
        )

    def __len__(self):  # noqa: D401
        return len(self.buffer)

# Training hyper‑parameters.
NUM_EPISODES = 5_000
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 15_000  # steps
TARGET_UPDATE = 500  # steps
LEARNING_RATE = 1e-3

# Environment & networks.
env = ContainerEnv()
policy_net = DQN().to(dev)
target_net = DQN().to(dev)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay = ReplayBuffer()

step_count = 0

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False

    while not done:
        # ε‑greedy action selection.
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-step_count / EPS_DECAY)
        if random.random() < eps_threshold:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                qvals = policy_net(torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0))
                action = int(qvals.argmax().item())

        next_state, reward, done, _, _ = env.step(action)
        replay.push(state, action, reward, next_state, done)
        state = next_state
        step_count += 1

        # Learn after enough samples.
        if len(replay) >= BATCH_SIZE:
            s_batch, a_batch, r_batch, sp_batch, d_batch = replay.sample(BATCH_SIZE)
            q_pred = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target_net(sp_batch).max(1)[0]
                q_target = r_batch + GAMMA * q_next * (~d_batch)
            loss = nn.functional.mse_loss(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Periodically update target network.
        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 250 == 0:
        print(f"Episode {episode+1:>4}/{NUM_EPISODES}  |  ε ≈ {eps_threshold:.3f}  |  steps: {step_count}")

print("Training complete! Saving model → dqn_container.pt")

torch.save(policy_net.state_dict(), "dqn_container.pt")

# Optional quick evaluation run.
state, _ = env.reset(seed=42)
done = False
while not done:
    with torch.no_grad():
        action = int(policy_net(torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)).argmax().item())
    state, _, done, _, info = env.step(action)

print(f"Filled {info.get('filled_cells', 0)} of {GRID_CELLS * GRID_CELLS} cells.")
