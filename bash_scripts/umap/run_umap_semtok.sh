source activate softvq
CONFIG=${1:-configs/vfmtok-ll-256.yaml}
WEIGHTS=${2:-weights/1stage-1.0-0.0.pth}
OUTDIR=${3:-results/1stage-1.0cos-0.0l2/semtok/}


CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference/visualize/do_umap_semTok.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --device "cuda:0" \
  --plot-path "$OUTDIR" \
  --skip-save