source activate softvq
CONFIG=${1:-configs/vfmtok-ll-256.yaml}
WEIGHTS=${2:-./weights/vfmtok-tokenizer.pt}
OUTDIR=${3:-results/vfmtok/vfmtok/}

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/visualize/do_umap_vfmtok.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --plot-path "$OUTDIR" \
  --skip-save