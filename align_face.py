""" Create a cropped-image centered on a face. 

"""

import os
from tempfile import mkstemp
import dlib
from pathlib import Path
import argparse
import torchvision
import urllib.request
from shape_predictor import align_face
from torch.hub import download_url_to_file
import tqdm.auto as tq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=str,  nargs='+',
                    help='directory with unprocessed images')
parser.add_argument('--output_dir', '-o', type=str, default='.', 
                    help='output directory')
parser.add_argument('--output_size', '-s', type=int, default=256, choices=[2**n for n in range(5, 11)],
                     help='size to downscale the input images to, must be power of 2')
parser.add_argument('--seed', type=int, default=127,
                    help='Random seed to use (for repeatability)')
parser.add_argument('--cache_dir', type=str, default='cache', help='cache directory for model weights')



args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)

landmarks_file = str(cache_dir / 'face_landmarks.dat') 
landmarks_url = "https://vision.csi.miamioh.edu/data/ii2s/shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(landmarks_file):
    print("Downloading Shape Predictor")
    download_url_to_file(landmarks_url, landmarks_file)
predictor = dlib.shape_predictor(landmarks_file)

extensions = {
    'image/jpeg':'.jpg',
    'image/png': '.png',
    'image/bmp': '.bmp',
    'image/tiff': '.tif',
}

for im in args.input:
    
    if im.startswith('http'):
        dat = urllib.request.urlopen(im)
        _, im = mkstemp(extensions.get(dat.info().get_content_type(), '.jpg'))
        with open(im, 'wb') as f:
            f.write(dat.read())
        
        stem, _ = os.path.splitext(os.path.basename(im))
        faces = align_face(str(im),predictor)

        os.remove(im)
    else:
        stem, _ = os.path.splitext(os.path.basename(im))
        faces = align_face(str(im),predictor)


    for i, face in enumerate(faces):
        output = str(output_dir / stem) + f".png"
        face.save(output)
        print(f"=> {output}")