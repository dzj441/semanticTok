import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
import sys
from PIL import Image
import numpy as np
import argparse
import itertools
from utils.misc import str2bool
import ruamel.yaml as yaml
from utils.data import ImageFolderWithFilename, center_crop_arr
from modelling.tokenizer import AEModel

def main(args):
    
    
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # load vae
    vae = AEModel.from_pretrained(f"MAETok/{args.vae_name}")
    vae = torch.compile(vae)
    for param in vae.parameters():
        param.requires_grad = False
    vae = vae.to(device).eval()   
             
    # Create folder to save samples:
    folder_name = f"{args.vae_name}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()


    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform)
    print(dataset)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )    

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    

    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    for x, _, paths in loader:
        
        x = x.to(device, non_blocking=True)
        x_flip = torch.flip(x, dims=[3])
        
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            zq, _, _ = vae.encode(x)
            zq_flip, _, _ = vae.encode(x_flip)

        for i, path in enumerate(paths):
            save_path = os.path.join(sample_folder_dir, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, path=path, zq=zq[i].cpu().numpy(), zq_flip=zq_flip[i].cpu().numpy())

        torch.cuda.synchronize()            
    print("Done.")

    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--vae_name", type=str, default="maetok-b-128")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    #fFirst parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)