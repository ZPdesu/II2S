import numpy as np
import PIL
import PIL.Image
import sys
import os
import glob
import scipy
import scipy.ndimage
import dlib
from drive import open_url
from pathlib import Path
import argparse
from bicubic import BicubicDownSample
import torchvision
from shape_predictor import align_face
from torch import nn

parser = argparse.ArgumentParser(description='Align_face')

parser.add_argument('-input_dir', type=str, default='/media/zhup/My Passport/Raw/Ours', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='/media/zhup/My Passport/Our_dataset', help='output directory')

parser.add_argument('-output_size', type=int, default=256, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

###############
parser.add_argument('-inter_method', type=str, default='bicubic')



args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)

print("Downloading Shape Predictor")
f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
predictor = dlib.shape_predictor(f)

for im in Path(args.input_dir).glob("*.*"):
    faces = align_face(str(im),predictor)

    for i,face in enumerate(faces):
        if(args.output_size):
            factor = 1024//args.output_size
            assert args.output_size*factor == 1024
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            # if args.inter_method == 'bicubic':
            #     D = BicubicDownSample(factor=factor)
            #     face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
            # elif args.inter_method == 'bilinear':
            #     face_tensor_lr = nn.functional.interpolate(face_tensor, size=args.output_size, mode='bilinear')[0].cpu().detach().clamp(0, 1)

            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)

        # face.save(Path(args.output_dir) / (im.stem+f"_{i}.png"))
        face.save(Path(args.output_dir) / (im.stem + f".png"))