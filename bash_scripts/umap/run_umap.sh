source activate softvq
CONFIG=${1:-configs/vfmtok-ll-256.yaml}
WEIGHTS=${2:-weights/semtok64.pth}
OUTDIR=${3:-results/semtok64/semtok/}

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/visualize/do_umap_semanticTok.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --plot-path "$OUTDIR" \
  --skip-save

huggingface-cli download DZJ181u2u/checkpoints \
 --local-dir ./weights \
 --include "1stage-1.0-0.0.pth"