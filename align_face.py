""" Create a cropped-image centered on a face.

"""
import os
import dlib
from pathlib import Path
import argparse
from utils.shape_predictor import align_face
from utils.model_utils import download_weight

parser = argparse.ArgumentParser(description='Align_face')

parser.add_argument('--input_dir', type=str, default='input', help='directory with unprocessed images')
parser.add_argument('--output_dir', type=str, default='input_aligned', help='output directory')
parser.add_argument('--output_size', '-s', type=int, default=1024, choices=[2 ** n for n in range(5, 11)],
                    help='size to downscale the input images to, must be power of 2')
parser.add_argument('--seed', type=int, default=127,
                    help='Random seed to use (for repeatability)')
parser.add_argument('--cache_dir', type=str, default='pretrained_models', help='cache directory for model weights')

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Shape Predictor")

predictor_weight = os.path.join(cache_dir, 'shape_predictor_68_face_landmarks.dat')
download_weight(predictor_weight)
predictor = dlib.shape_predictor(predictor_weight)


for im in Path(args.input_dir).glob("*.*"):
    face = align_face(str(im), predictor, output_size=args.output_size)
    face.save(Path(args.output_dir) / (im.stem + f".png"))

