#!/usr/bin/env bash
# Stage A (vLLM NVFP4 generation, in docker) -> Stage B (bf16 activation capture).
# Usage: bash interp/run_pipeline.sh [N]   (run from rumik-oss-model-dissection/)
set -euo pipefail
N=${1:-500}
STAGE=${2:-all}   # all | B  (B = skip vLLM, reuse gen_ids.jsonl)
OUT=${3:-interp/acts}
GRID=${4:-v1}
SEED=${5:-0}
mkdir -p $OUT
if [ "$STAGE" = all ]; then

echo "=== Stage A: vLLM generation ($N clips) ==="
docker run --rm --gpus all \
  -v $PWD/rumik-oss-1-nvfp4-vllm:/models/rumik-oss-1-nvfp4-vllm:ro \
  -v $HOME/.cache/huggingface/hub:$HOME/.cache/huggingface/hub:ro \
  -v $PWD/vllm-plugin:/workspace/vllm-plugin \
  -v $PWD/interp:/work/interp \
  --entrypoint bash vllm/vllm-openai:v0.20.0 \
  -c "pip install -q -e /workspace/vllm-plugin && VLLM_PLUGINS=register_tinyaya python3 -u /work/interp/vllm_gen.py --n $N --seed $SEED --grid $GRID --out /work/$OUT"

# docker writes as root -> reclaim ownership of outputs
docker run --rm -v $PWD/interp:/work/interp --entrypoint chown vllm/vllm-openai:v0.20.0 -R $(id -u):$(id -g) /work/interp

fi

echo "=== Stage B: bf16 activation capture ==="
# hard cap: if it overruns, the kernel kills THIS process instead of freezing the DGX
systemd-run --user --scope -p MemoryMax=60G -p MemorySwapMax=0 \
  env HF_MODULES_CACHE=$HOME/.cache/hf_modules_user PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u interp/capture_acts.py --n $N --seed $SEED --grid $GRID --out $OUT 2>&1 | tee $OUT.capture.log
