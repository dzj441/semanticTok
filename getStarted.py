import torch
from PIL import Image
import numpy as np
import requests
import sys
import os
sys.path.append('../')
from modelling.tokenizer import SoftVQModel
from matplotlib import pyplot as plt


def save_image(image, filename, title=''):
    os.makedirs('figures', exist_ok=True)
    plt.figure()
    plt.imshow(torch.clip((image * 0.5 + 0.5) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    save_path = os.path.join('figures', filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {save_path}")

vae = SoftVQModel.from_pretrained("./SoftVQVAE/softvq-l-64")
vae = vae.eval()

if torch.cuda.is_available():
    vae = vae.cuda()
    device = torch.device('cuda')
    device_type = 'cuda'
else:
    device = torch.device('cpu')
    device_type = 'cpu'

# load an image
img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
img = Image.open(requests.get(img_url, stream=True).raw)
img = img.resize((256, 256))
img = np.array(img) / 255.

assert img.shape == (256, 256, 3)

# normalize by ImageNet mean and std
img = img - 0.5
img = img / 0.5

save_image(torch.tensor(img), 'original.png', title='Original Image')

input_img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

with torch.amp.autocast(device_type=device_type), torch.no_grad():
    recon_img, _, _ = vae(input_img)

recon_img = recon_img[0].cpu().permute(1, 2, 0).numpy()

# 保存重建图像
save_image(torch.tensor(recon_img), 'reconstruction.png', title='Reconstructed Image')
