conda activate softvq
WANDB_MODE=offline torchrun --nproc_per_node=1 train/train_tokenizer.py --config configs/vfmtok-sem-direct.yaml