source activate softvq
CONFIG=${1:-configs/vfmtok-ll-256.yaml}
WEIGHTS=${2:-experiments/tokenizer/exp003-vfmtok-ll-256/checkpoints/0250000.pth}
OUTDIR=${3:-results/vfmtok}

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/eval.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --distributed false \
  --save-png true \
  --sample-dir "$OUTDIR/png" \
  --npz-count 50000 \
  --fid true
