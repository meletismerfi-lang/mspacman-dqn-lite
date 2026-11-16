#!/usr/bin/env bash
# One-shot setup for the Ms. Pac-Man DQN run on a RunPod (or any CUDA Linux) box.
# Pins the exact stack verified locally: torch cu12x, SB3 2.3.2, gymnasium 0.29.1.
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== python =="
python3 --version || { echo "need python3"; exit 1; }

pip install -q --upgrade pip
# torch stays whatever the image ships (CUDA-matched); only add what we need
pip install -q "stable-baselines3==2.3.2" "gymnasium==0.29.1" "ale-py==0.10.1" envpool opencv-python tensorboard

echo "== versions =="
python3 - <<'PY'
import torch, stable_baselines3 as sb3, gymnasium, envpool
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("sb3", sb3.__version__, "| gymnasium", gymnasium.__version__, "| envpool", envpool.__version__)
PY

echo "== smoke test (2 min): 40k steps, 32 envs =="
python3 dqn_pacman.py --envs 32 --steps 40000 --batch-size 32 --gradient-steps 2 \
    --learning-starts 5000 --ckpt-every 100000 --resume checkpoint_4m.zip 2>&1 | tail -6

echo
echo "smoke test done. If that printed a steps/s line, launch the real run:"
echo "  nohup python3 dqn_pacman.py --envs 256 --steps 20000000 --batch-size 32 \\"
echo "      --gradient-steps 2 --buffer-size 500000 --ckpt-every 1000000 \\"
echo "      --resume checkpoint_4m.zip > logs_run.txt 2>&1 &"
