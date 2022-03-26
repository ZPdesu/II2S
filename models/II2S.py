import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset
from losses.loss import LossBuilder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import dlib
import PIL
import torchvision
from utils.model_utils import download_weight
toPIL = torchvision.transforms.ToPILImage()

class II2S(nn.Module):

    def __init__(self, opts):
        super(II2S, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.setup_loss_builder()
        self.set_up_face_predictor()


    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)

    def setup_optimizer(self):

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        latent = []
        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)

        return optimizer, latent


    def setup_dataloader(self, image_path=None, align_input=False):

        self.dataset = ImagesDataset(opts=self.opts, image_path=image_path,
                                     face_predictor=self.predictor, align_input=align_input)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print("Number of images: {}".format(len(self.dataset)))

    def setup_loss_builder(self):
        self.loss_builder = LossBuilder(self.opts)


    def set_up_face_predictor(self):
        self.predictor = None
        predictor_weight = os.path.join('pretrained_models', 'shape_predictor_68_face_landmarks.dat')
        download_weight(predictor_weight)
        self.predictor = dlib.shape_predictor(predictor_weight)


    def invert_images(self, image_path=None, output_dir=None, return_latents=False, align_input=False, save_output=True):

        final_latents =None
        if return_latents:
            final_latents = []

        self.setup_dataloader(image_path=image_path, align_input=align_input)
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')
        for ref_im_H, ref_im_L, ref_name in ibar:
            optimizer, latent = self.setup_optimizer()
            pbar = tqdm(range(self.opts.steps), desc='Embedding')
            for step in pbar:
                optimizer.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)

                gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer.step()

                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))

                if self.opts.save_intermediate and step % self.opts.save_interval==0 and save_output:
                    self.save_intermediate_results(ref_name, gen_im, latent_in, step, output_dir)

            if save_output:
                self.save_results(ref_name, gen_im, latent_in, output_dir)

            if return_latents:
                final_latents.append(latent_in)

        return final_latents


    def cal_loss(self, im_dict, latent_in):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        return loss, loss_dic


    def save_results(self, ref_name, gen_im, latent_in, output_dir):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npy')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)


    def save_intermediate_results(self, ref_name, gen_im, latent_in, step, output_dir):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        intermediate_folder = os.path.join(output_dir, ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)


    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True