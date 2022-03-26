from torch.utils.data import Dataset
from PIL import Image
import PIL
from utils import data_utils
import torchvision.transforms as transforms
import os
from utils.shape_predictor import align_face
import sys

class ImagesDataset(Dataset):

    def __init__(self, opts, image_path=None, face_predictor=None, align_input=False):

        if type(image_path) == list:
            self.image_paths = image_path
        elif os.path.isdir(image_path):
            self.image_paths = sorted(data_utils.make_dataset(image_path))
        elif os.path.isfile(image_path):
            self.image_paths = [image_path]
        else:
            sys.exit('Invalid Input')


        self.image_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.opts = opts
        self.align_input = align_input
        self.predictor = face_predictor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        im_path = self.image_paths[index]

        if self.align_input and self.predictor is not None:
            im_H = align_face(im_path, self.predictor, output_size=self.opts.size)
        else:
            im_H = Image.open(im_path).convert('RGB')
        im_L = im_H.resize((256, 256), PIL.Image.LANCZOS)
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        if self.image_transform:
            im_H = self.image_transform(im_H)
            im_L = self.image_transform(im_L)

        return im_H, im_L, im_name



