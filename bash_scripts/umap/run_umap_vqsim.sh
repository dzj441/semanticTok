source activate softvq
CONFIG=${1:-configs/simvq-bl-128-eval.yaml}
WEIGHTS=${2:-weights/0250000.pth}
OUTDIR=${3:-results/simsoft/tok/}

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/visualize/do_umap_simvq.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --plot-path "$OUTDIR" \
  --skip-save