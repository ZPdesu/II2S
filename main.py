import argparse

import torch
import numpy as np
import sys
import os
import dlib


from PIL import Image


from models.II2S import II2S


def main(args):
    ii2s = II2S(args)

    ##### Option 1: input folder
    ii2s.invert_images()

    ##### Option 2: image path
    # ii2s.invert_images('input/face/20.png')

    ##### Option 3: image path list
    # ii2s.invert_images(['input/face/35.png', 'input/face/20.png'])






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='II2S')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/face',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # loss options
    parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
    parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
    parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')

    # arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--steps', type=int, default=1300, help='Number of optimization steps')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
    parser.add_argument('--verbose', action='store_true', help='Print loss information')

    args = parser.parse_args()
    main(args)