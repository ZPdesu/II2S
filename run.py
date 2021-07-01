
from embedding import Embedding
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
from bicubic import BicubicDownSample
import torch

import numpy as np

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
# memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
# os.system('rm tmp')
# os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))



class Images(Dataset):
    def __init__(self, root_dir):
        self.root_path = Path(root_dir)
        # self.image_list = sorted(list(self.root_path.glob("*.JPEG")))
        self.image_list = sorted(list(self.root_path.glob("*.png")))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        return image,img_path.stem



parser = argparse.ArgumentParser(description='II2S')


# I/O arguments
parser.add_argument('--input_dir', type=str, default='input', help='input data directory')
parser.add_argument('--output_dir', type=str, default='output/images', help='output data directory')
parser.add_argument('--NPY_folder', default='output/latents', type=str)
parser.add_argument('--In_W_space', type=bool, default=True)
parser.add_argument('--L2_regularizer', action='store_true')
parser.add_argument('--l2_regularize_weight', type=float, default=0.001)


# StyleGAN2 setting
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
parser.add_argument('--channel_multiplier', type=int, default=2)
parser.add_argument('--latent', type=int, default=512)
parser.add_argument('--n_mlp', type=int, default=8)


# arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="None", help='Loss function to use')
parser.add_argument('-eps', type=float, default=0, help='Target for downscaling loss (L2)')
parser.add_argument('-tile_latent', default=False, type=bool, help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=1500, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true',help='Whether to store and save intermediate HR and LR images during optimization')




args = parser.parse_args()
kwargs = vars(args)


dataset = Images(kwargs["input_dir"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)



dataloader = DataLoader(dataset, batch_size=1)


model = Embedding(args)

toPIL = torchvision.transforms.ToPILImage()



for count, (ref_im, ref_im_name) in enumerate(dataloader):
    if (kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        int_path_HR = Path(out_path / ref_im_name[0])
        int_path_HR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR) in enumerate(model(ref_im.cuda(),ref_im_name,**kwargs)):
            toPIL(HR[0].cpu().detach().clamp(0, 1)).save(
                int_path_HR / f"{ref_im_name[0]}_{j*50:0{padding}}.png")

    else:
        for j,(HR,LR) in enumerate(model(ref_im.cuda(), ref_im_name, **kwargs)):
            # print(count)
            toPIL(HR[0].cpu().detach().clamp(0, 1)).save(out_path / f"{ref_im_name[0]}.png")

