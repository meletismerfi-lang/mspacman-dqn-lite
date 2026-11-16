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


def preprocess(frame_rgb: np.ndarray, prev_gray: np.ndarray | None = None, size: int = 84) -> np.ndarray:
    """RGB frame (210,160,3) -> 84x84 grayscale, max-pooled over the last two frames.

    The max-pool matters: the trainer used a MaxAndSkip-style wrapper (max of consecutive
    frame pairs), so skipping it changes what the network sees and costs ~700 points.
    """
    try:
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            gray = np.maximum(gray, prev_gray)
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA), gray
    except ImportError:
        gray = frame_rgb[..., :3].mean(axis=2).astype(np.uint8)
        if prev_gray is not None:
            gray = np.maximum(gray, prev_gray)
        h, w = gray.shape
        fy, fx = h // size, w // size
        cropped = gray[: fy * size, : fx * size].astype(np.float32)
        small = cropped.reshape(size, fy, size, fx).mean(axis=(1, 3)).astype(np.uint8)
        return small, gray


def act(model, stack: deque, frame_rgb: np.ndarray | None = None, prev_gray: np.ndarray | None = None):
    """Append a (max-pooled, downsampled) frame if given, then return (action, q_values, gray)."""
    if frame_rgb is not None:
        small, prev_gray = preprocess(frame_rgb, prev_gray)
        stack.append(small)
    obs = np.stack(stack)[None].astype(np.uint8)          # (1, 4, 84, 84)
    with torch.no_grad():
        q = model(torch.from_numpy(obs))[0]
    return int(q.argmax()), q.numpy(), prev_gray


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--play", action="store_true", help="play real ALE episodes (needs ale-py)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--render", action="store_true", help="show a pygame window while playing")
    args = p.parse_args()

    model = load_torchscript(args.model)
    sample = Path(__file__).parent / "sample_obs.npy"
    if sample.exists() and not args.play:
        obs = np.load(sample)                             # (1, 4, 84, 84) uint8
        with torch.no_grad():
            q = model(torch.from_numpy(obs))[0].numpy()
        print("bundled sample observation")
        for name, v in zip(ACTIONS, q):
            print(f"  {name:10s} {v:8.3f}")
        print("greedy action:", ACTIONS[int(q.argmax())])
        print("(use --play to run the actual game)")
        return

    try:
        import ale_py  # noqa: F401
        import gymnasium as gym
        gym.register_envs(ale_py)
    except ImportError:
        raise SystemExit("ale-py not installed — pip install ale-py (or drop --play) for live play")

    env = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0, render_mode="rgb_array")
    window = None
    if args.render:
        import pygame
        pygame.init()

    scores = []
    for ep in range(args.episodes):
        frame, _ = env.reset(seed=args.seed + ep)
        stack: deque = deque(maxlen=4)
        prev_gray = None
        for _ in range(4):
            small, prev_gray = preprocess(frame, prev_gray)
            stack.append(small)
        total, steps = 0.0, 0
        while True:
            a, _, prev_gray = act(model, stack)
            frame, r, term, trunc, _ = env.step(a)
            small, prev_gray = preprocess(frame, prev_gray)
            stack.append(small)
            total += float(r)
            steps += 1
            if args.render:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                surf = pygame.transform.scale(surf, (480, 630))
                if window is None:
                    window = pygame.display.set_mode(surf.get_size())
                    pygame.display.set_caption("Ms. Pac-Man — exported DQN")
                pygame.event.pump()
                window.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.delay(8)
            if term or trunc:
                break
        scores.append(total)
        print(f"episode {ep + 1}: score {total:.0f} over {steps} steps")
    print(f"mean score: {sum(scores) / len(scores):.0f}")
    env.close()
    if args.render:
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
