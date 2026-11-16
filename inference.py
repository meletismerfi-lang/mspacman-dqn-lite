"""Minimal inference: preprocess frames -> Q-values -> action.

  python inference.py                      # runs on the bundled sample observation
  python inference.py --frames 4 --seed 7  # needs ale-py + opencv if you want live frames
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from model import ACTIONS, load_torchscript


def preprocess(frame_rgb: np.ndarray, size: int = 84) -> np.ndarray:
    """RGB frame (210,160,3) -> 84x84 grayscale. Uses cv2 when available, else a 2x box filter."""
    try:
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        gray = frame_rgb[..., :3].mean(axis=2)
        h, w = gray.shape
        fy, fx = h // size, w // size
        cropped = gray[: fy * size, : fx * size].astype(np.float32)
        return cropped.reshape(size, fy, size, fx).mean(axis=(1, 3)).astype(np.uint8)


def act(model, stack: deque, frame_rgb: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    if frame_rgb is not None:
        stack.append(preprocess(frame_rgb))
    obs = np.stack(stack)[None].astype(np.uint8)          # (1, 4, 84, 84)
    with torch.no_grad():
        q = model(torch.from_numpy(obs))[0]
    return int(q.argmax()), q.numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    model = load_torchscript(args.model)
    sample = Path(__file__).parent / "sample_obs.npy"
    if sample.exists():
        obs = np.load(sample)                             # (1, 4, 84, 84) uint8
        with torch.no_grad():
            q = model(torch.from_numpy(obs))[0].numpy()
        print("bundled sample observation")
        for name, v in zip(ACTIONS, q):
            print(f"  {name:10s} {v:8.3f}")
        print("greedy action:", ACTIONS[int(q.argmax())])
        return

    try:
        import ale_py  # noqa: F401
        import gymnasium as gym
        gym.register_envs(ale_py)
    except ImportError:
        raise SystemExit("no sample_obs.npy and ale-py not installed — nothing to run on")

    env = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0, render_mode="rgb_array")
    frame, _ = env.reset(seed=args.seed)
    stack: deque = deque(maxlen=4)
    for _ in range(4):
        stack.append(preprocess(frame))
    total, steps = 0.0, 0
    while True:
        a, _ = act(model, stack)
        frame, r, term, trunc, _ = env.step(a)
        stack.append(preprocess(frame))
        total += float(r)
        steps += 1
        if term or trunc:
            break
    print(f"episode: score {total:.0f} over {steps} steps (4-frame actions)")
    env.close()


if __name__ == "__main__":
    main()
