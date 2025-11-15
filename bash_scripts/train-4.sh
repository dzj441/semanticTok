source activate softvq
WANDB_MODE=offline torchrun --nproc_per_node=4 train/train_tokenizer.py --config configs/vfmtok-ll-256.yaml