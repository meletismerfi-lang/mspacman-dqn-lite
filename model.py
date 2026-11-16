"""Standalone definition of the exported Ms. Pac-Man Q-network (no Stable-Baselines3 needed).

Input : uint8 tensor (N, 4, 84, 84) — 4 stacked grayscale frames, raw 0-255
Output: float32 tensor (N, 9) — Q-values per ALE action
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ACTIONS = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]


class LiteQNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.q_head = nn.Linear(512, 9)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0
        return self.q_head(self.linear(self.cnn(x)))


def load_torchscript(path: str | Path | None = None):
    """Load the traced model: one file, no class definition required."""
    path = Path(path or Path(__file__).parent / "mspacman_qnet.pt")
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


def load_weights(path: str | Path | None = None) -> LiteQNet:
    """Load from a state dict (safe if you prefer explicit classes over TorchScript)."""
    path = Path(path or Path(__file__).parent / "mspacman_qnet_state_dict.pt")
    model = LiteQNet()
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model


def n_params(model: nn.Module | None = None) -> int:
    model = model or LiteQNet()
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = load_torchscript()
    print("torchscript ok · params:", f"{n_params():,}")
    out = m(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
    print("output shape:", tuple(out.shape))
