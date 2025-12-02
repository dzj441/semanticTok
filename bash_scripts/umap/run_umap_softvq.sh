source activate softvq
CONFIG=${1:-configs/softvq-l-64.yaml}
WEIGHTS=${2:-./weights/softvq/}
OUTDIR=${3:-results/soft/tok/}

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/visualize/do_umap_softvq.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --plot-path "$OUTDIR" \
  --skip-save