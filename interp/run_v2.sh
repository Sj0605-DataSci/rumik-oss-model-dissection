#!/usr/bin/env bash
# v2 chain on interp/acts_v2 (2500 clips, accent x language crossed):
# train SAE L19 -> splice + labels -> steering sweep -> WavLM speaker probe (CPU)
# Run from rumik-oss-model-dissection/.  Every GPU stage runs under a 60G cgroup cap.
set -euo pipefail
ACTS=interp/acts_v2
SAE=interp/sae_v2_L19_x16_k64
STEER=interp/steer_v2_L19
CAP="systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 env HF_MODULES_CACHE=$HOME/.cache/hf_modules_user PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "=== [1/4] train SAE ==="
$CAP .venv/bin/python -u interp/train_sae.py --acts $ACTS --layer 19 --epochs 6 --out $SAE 2>&1 | tee $SAE.train.log

echo "=== [2/4] splice + feature labels ==="
$CAP .venv/bin/python -u interp/analyze_sae.py --sae $SAE --acts $ACTS 2>&1 | tee $SAE.analyze.log

echo "=== [3/4] steering sweep ==="
$CAP .venv/bin/python -u interp/steer.py --sae $SAE --acts $ACTS --out $STEER \
  --alphas 0 0.1 0.2 0.3 0.4 0.5 --seeds 3 2>&1 | tee $STEER.log

echo "=== [4/4] speaker probe (CPU) ==="
cd interp && CUDA_VISIBLE_DEVICES= nice -n 10 env HF_HUB_OFFLINE=1 ../.venv/bin/python -u speaker_probe.py \
  --ref ../$ACTS --steer ../$STEER/to_zoya ../$STEER/to_aisha ../$STEER/to_siya ../$STEER/pace_fast ../$STEER/sad ../$STEER/excited \
  2>&1 | tr '\r' '\n' | grep -E "\[probe\]|α=" | tee ../$STEER.probe.log
echo "=== v2 chain done ==="
