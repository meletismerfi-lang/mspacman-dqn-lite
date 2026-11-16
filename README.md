# Ms. Pac-Man — lightweight DQN inference model (1.69M params)

A Double-DQN agent trained from pixels on Atari Ms. Pac-Man (ALE). This repo ships only the
**online Q-network** — the target network and the trainer are not included, so the artifact is
a single 6.5 MB file.

Trained on an RTX 5060 Laptop (WSL2 + EnvPool): 4,000,000 environment steps, ~33 minutes,
2,041 steps/s.

## Scores (full game, 3 lives, greedy policy, 4 seeds)

| seed | score |
|---|---|
| 7 | 2,210 |
| 8 | 1,820 |
| 9 | 1,820 |
| 10 | 2,760 |

Mean **2,152**; the agent clears most of the maze and eats ghosts (see the frame in `docs/`).
For scale: random play in Ms. Pac-Man scores roughly 200-300.

## Files

| file | what |
|---|---|
| `mspacman_qnet.pt` | TorchScript, self-contained — load with `torch.jit.load`, no class needed |
| `mspacman_qnet_state_dict.pt` | raw weights, for `LiteQNet` in `model.py` |
| `model.py` | the network definition (NatureCNN trunk + 9-way head) and loaders |
| `inference.py` | frames -> Q-values -> action, runs on the bundled sample |
| `sample_obs.npy` | one real preprocessed observation, `(1, 4, 84, 84)` uint8 |
| `model_info.json` | machine-readable model card |

## Architecture

```
input  uint8 (N, 4, 84, 84)         4 stacked grayscale frames, 0-255
  Conv2d(4 -> 32, 8x8, stride 4) + ReLU      77,984 conv params total
  Conv2d(32 -> 64, 4x4, stride 2) + ReLU
  Conv2d(64 -> 64, 3x3, stride 1) + ReLU
  Flatten -> 3136
  Linear(3136 -> 512) + ReLU              1,606,144 params
  Linear(512 -> 9)                            4,617 params
output float32 (N, 9) Q-values
```

**1,688,745 parameters.** 95% of them sit in the single 3136→512 dense layer; the convolutions
are only 78k. That ratio is why mixed-precision and `torch.compile` gave single-digit speedups
during training — the model is memory-bound, not FLOP-bound.

## The preprocessing contract (this is what people get wrong)

The policy was trained on grayscale, 84x84, 4-frame stacks with a 4-frame action repeat. Feed it
RGB, 210x160 frames, or a different stack order and it plays badly — not because the weights are
wrong, but because the input distribution is.

```
1. emulator runs with repeat_action_probability = 0, frame skip 4 (one env step = 4 frames)
2. each new frame: RGB(210,160,3) -> grayscale -> resize 84x84 (INTER_AREA)
3. append to a deque(maxlen=4); stack oldest->newest  -> (4, 84, 84)
4. to the model: (1, 4, 84, 84) uint8, unnormalised (the model divides by 255 internally)
5. take argmax over the 9 Q-values, repeat that action for the next 4 frames
```

Action order: `NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT`.

## Use it

```python
import numpy as np, torch
from model import load_torchscript, ACTIONS

model = load_torchscript("mspacman_qnet.pt")
obs = np.load("sample_obs.npy")                     # (1, 4, 84, 84) uint8
with torch.no_grad():
    q = model(torch.from_numpy(obs))[0]
print(ACTIONS[int(q.argmax())], q)
```

```bash
python inference.py      # prints the Q-values for the bundled observation
```

## Training settings (for provenance)

DQN, CnnPolicy, 32 parallel EnvPool envs, 4-frame skip, reward = raw score delta (no clipping),
gamma 0.99, lr 1e-4 linear decay, batch 32 × 32 envs, 200k-transition replay kept in GPU VRAM,
target update every 1,000 steps, exploration 1.0 -> 0.01 over the first 10%.

## Limitations

- One game, one seed. Not a general Atari agent.
- Deterministic greedy policy: it can loop if the maze state repeats exactly.
- Trained with episodic-life wrappers off in playback, so scores are full 3-life games; eval runs
  that terminate on each lost life will report much lower numbers (~700).
