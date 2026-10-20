#!/usr/bin/env bash
# Faster-JiT: JiT-B/16 on ImageNet 256x256 (upstream recipe) with the measured optimization switches.
# Usage:
#   export IMAGENET_PATH=/path/to/imagenet   # contains train/<class>/*.JPEG
#   export OUTPUT_DIR=/path/to/output        # checkpoints and logs; also used for --resume
#   bash scripts/train_faster_jit_b16.sh [extra main_jit.py arguments]
# NPROC (default 8, as upstream) sets the GPU count. Keep 128 images per GPU: the fused SwiGLU path is
# guarded to that shape and falls back to the upstream code otherwise. The 1.5618x figure was measured on 2 GPUs.
# Online FID evaluation was not part of the measured or validated path; pass --online_eval explicitly if needed.
set -euo pipefail
: "${IMAGENET_PATH:?Set IMAGENET_PATH to the ImageNet root that contains train/}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the output and checkpoint directory}"
NPROC=${NPROC:-8}
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROC must be a positive integer.' >&2; exit 2; }
[[ -d "$IMAGENET_PATH/train" ]] || { echo "Missing $IMAGENET_PATH/train" >&2; exit 2; }
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source scripts/configs/faster.env
torchrun --nproc_per_node="$NPROC" --nnodes=1 --node_rank=0 \
  main_jit.py \
  --model JiT-B/16 \
  --proj_dropout 0.0 \
  --P_mean -0.8 --P_std 0.8 \
  --img_size 256 --noise_scale 1.0 \
  --batch_size 128 --blr 5e-5 \
  --epochs 600 --warmup_epochs 5 \
  --gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
  --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR" \
  --data_path "$IMAGENET_PATH" "$@"
