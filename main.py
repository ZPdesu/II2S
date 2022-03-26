import argparse

import torch
import numpy as np
import sys
import os
import dlib
from PIL import Image
import json

from models.II2S import II2S
from options.face_embed_options import FaceEmbedOptions


def main(args):
    ii2s = II2S(args)

    ##### Option 1: input folder
    final_latents = ii2s.invert_images(image_path=args.input_dir, output_dir=args.output_dir,
                                       return_latents=True, align_input=True, save_output=True)

    #### Option 2: image path
    # final_latents = ii2s.invert_images(image_path='input/28.jpg', output_dir=args.output_dir,
    #                                     return_latents=True, align_input=True, save_output=True)

    # ##### Option 3: image path list
    # final_latents = ii2s.invert_images(image_path=['input/28.jpg', 'input/90.jpg'], output_dir=args.output_dir,
    #                                    return_latents=True, align_input=True, save_output=True)

    # print(final_latents)




if __name__ == "__main__":

    parser = FaceEmbedOptions().parser

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input',
                        help='The directory of the images to be inverted. ')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images.')

    args = parser.parse_args()

    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)