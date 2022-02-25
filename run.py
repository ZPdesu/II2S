
import sys
from tqdm.std import tqdm
from embedding import Embedding, make_embedding
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
from bicubic import BicubicDownSample
import torch
import tqdm as tq

import numpy as np
import os

from ii2s.config import cfg

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
# memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
# os.system('rm tmp')
# os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu))



class Images(Dataset):
    def __init__(self, root_dir, pattern='*.png'):
        """Construct a dataset of images to process

        Args:
            root_dir (str): The folder containing input image
            pattern (str): The pattern used to glob for images in the root folder
        """
        self.root_path = Path(root_dir)
        self.image_list = sorted(list(self.root_path.glob(pattern)))


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        return image,img_path.stem



def run_reconstruction(args, cfg):
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)


    if os.path.isfile(args.input):
        root, pattern = os.path.split(args.input)
    else:
        root, pattern = args.input, '*.png'

    dataset = Images(root, pattern)
    dataloader = DataLoader(dataset, batch_size=1)
    model = make_embedding(root, cfg)
    toPIL = torchvision.transforms.ToPILImage()

    if cfg.SAVE_INTERMEDIATE > 0:
        save_freq = cfg.SAVE_INTERMEDIATE
    else:
        save_freq = sys.maxsize

    for count, (ref_im, ref_im_name) in enumerate(tq.tqdm(dataloader, desc="Reconstructing")):
         
        for j, result in enumerate(model(ref_im.cuda(), ref_im_name), start=1):
            if j % cfg.LOG_FREQ == 0:
                tqdm.write(result['summary'])

            if j % save_freq == 0:
                int_path_HR = Path(out_path / ref_im_name[0])
                int_path_HR.mkdir(parents=True, exist_ok=True)
                image = toPIL(result['image'][0].detach().cpu().clamp(0, 1))
                latent = result['latent'][0].detach().cpu().numpy()
                image.save(int_path_HR / f"{ref_im_name[0]}_{j*50:06}.png")
                np.save(int_path_HR / f"{ref_im_name[0]}_{j*50:06}.npy", latent)
        
        image = toPIL(result['image'][0].detach().cpu().clamp(0, 1))
        latent = result['latent'][0].detach().cpu().numpy()
        image.save(out_path / f"{ref_im_name[0]}.png")
        np.save(out_path / f"{ref_im_name[0]}.npy", latent)


def dump_config(args, cfg):
    print(cfg.dump())

actions = {
    'reconstruction': run_reconstruction,
    'dump-config': dump_config
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='II2S', 
                                    epilog="Additional configuration can be done "
                                            "by editing a config file, and config "
                                            "values can be overridden by passing "
                                            "fully qualified Key-Value pairs as "
                                            "additional command line options")


    # I/O arguments
    parser.add_argument('action', choices=list(actions.keys()), help="The action to perform")
    parser.add_argument('--input', type=str, default='input', help='input image or directory')
    parser.add_argument('--output', type=str, default='output', help='output data directory')
    parser.add_argument('--NPY_folder', default='output/latents', type=str)
    parser.add_argument('--config', '-c', help="the config file to use")



    args, argv = parser.parse_known_args()

    #  Read in additional config from a file
    if args.config is not None:
        cfg.merge_from_file(args.config)

    # Override config with command line options in the format: SECTION.KEY Value
    cfg.merge_from_list(argv)

    # Do the corresponding action
    actions[args.action](args, cfg)


