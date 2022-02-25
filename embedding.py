
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from functools import partial
import numpy as np
from loss import LossBuilder
import time
from bicubic import BicubicDownSample
import os
import PIL
import torchvision
import sys

URLS={
    "stylegan2-ffhq-config-f":"https://drive.google.com/uc?id=1rKxxPm4FHnna1E-D5fBsj5gjG0sWnodr"
}


def make_embedding(input_dir,  cfg):
    """Construct an Embedding object using the configuration values

    Args:
        cfg (yacs.config.CfgNode): The configuration
    """
    return Embedding(input_dir=input_dir,
                     In_W_space=cfg.IN_W_SPACE,
                     lr=cfg.LEARNING_RATE, 
                     eps=cfg.EPS, 
                     seed=cfg.SEED,
                     model=cfg.STYLEGAN2.CKPT,
                     size=cfg.STYLEGAN2.SIZE,
                     latent=cfg.STYLEGAN2.LATENT,
                     tile_latent=cfg.TILE_LATENT,
                     channel_multiplier=cfg.STYLEGAN2.CHANNEL_MULTIPLIER,
                     n_mlp=cfg.STYLEGAN2.N_MLP,
                     opt_name=cfg.OPT_NAME,
                     steps=cfg.STEPS)

def _make_optimizer(name, latent, lr):
    """Make an optimizer to use within the embedding (internal to the Embedding class)

    Args:
        name (str): The name of the optimizer (sgd, adam, sgdm, adamax). 
        latent (list[Tensor]): The latent code to optimize
        lr (float): The learning rate for the optimizer
    Returns:
         Optimzer : An optimizer 
    """
    opt_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'sgdm': partial(torch.optim.SGD, momentum=0.9),
        'adamax': torch.optim.Adamax
    }

    opt_final = opt_dict[name](latent[0:], lr=lr)    
    return opt_final


class Embedding(torch.nn.Module):
    def __init__(self, 
                 input_dir='./input',
                 NPY_folder='./output/latents',
                 In_W_space=True,
                 lr=0.001,
                 eps=0,
                 steps=1500,
                 seed = None,
                 verbose=True,
                 model="models/stylegan2-ffhq-config-f.pt", 
                 size=1024,
                 latent=512,
                 tile_latent=False,
                 channel_multiplier=2,
                 n_mlp=8,
                 opt_name='adam',
                 L2_regularizer=0.0,
                 download=True):
        super(Embedding, self).__init__()

        self.synthesis = Generator(size, latent, n_mlp, channel_multiplier=channel_multiplier)
        self.synthesis = self.synthesis.cuda()
        self.verbose = verbose

        if download and not os.path.exists(model):
            import gdown
            gdown.download(URLS["stylegan2-ffhq-config-f"], model, quiet=not verbose)

        model_state = torch.load(model)
        self.synthesis.load_state_dict(model_state['g_ema'])
        self.latent_avg = model_state['latent_avg']

        for param in self.synthesis.parameters():
            param.requires_grad = False

        self.my_downsample = BicubicDownSample(factor=4)

        self.learning_rate = lr
        self.seed = seed
        self.eps = eps
        self.tile_latent = tile_latent
        self.L2_regularizer = L2_regularizer
        self.opt_name = opt_name
        self.steps = steps

        # Why????
        self.input_dir = input_dir
        self.NPY_folder = NPY_folder
        self.In_W_space = In_W_space

        # TODO: Shouldn't this be downlaoded like the model?  
        #       How can we assume this is in curdir?
        transformer_space = np.load('transformer_space.npz')
        self.X_mean = torch.from_numpy(transformer_space['X_mean']).float().cuda()
        self.X_comp = torch.from_numpy(transformer_space['X_comp']).float().cuda()
        self.X_stdev = torch.from_numpy(transformer_space['X_stdev']).float().cuda()


        if not self.In_W_space:
            tmp = self.latent_avg.clone().detach().cuda()
            tmp = torch.nn.LeakyReLU(negative_slope=5)(tmp) - self.X_mean
            tmp = tmp.unsqueeze(0).mm(self.X_comp.T).squeeze()
            self.W_to_normalized = tmp / self.X_stdev

        print('In_W_space', self.In_W_space)
        print('lr', self.learning_rate)


    def forward(self, ref_im, ref_im_name):
        
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True


        if (self.tile_latent):
            latent = self.latent_avg.reshape(1, 1, -1).cuda()
            latent.requires_grad = True
        else:
            latent = []
            for i in range(18):
                if self.In_W_space:
                    tmp = self.latent_avg.clone().detach().cuda()
                else:
                    tmp = self.W_to_normalized.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)


        opt_final = _make_optimizer(self.opt_name, latent, self.learning_rate)
        loss_builder_final = LossBuilder(ref_im_name, ref_im, '1.0*L2+1.0*percep', self.eps, self.input_dir).cuda()

        summary = ""
        start_t = time.time()

        if self.verbose: 
            tqdm.write("Optimizing")


        for j in range(self.steps):
            opt_final.zero_grad()

            loss_dict= {}


            if self.In_W_space:
                latent_in = torch.stack(latent).unsqueeze(0)
            else:
                latent_in = torch.nn.LeakyReLU(negative_slope=0.2)(
                    (torch.stack(latent).unsqueeze(0) * self.X_stdev).bmm(self.X_comp.unsqueeze(0)) + self.X_mean)


            #################  make noise
            noises_single = self.synthesis.make_noise()
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(1, 1, 1, 1).normal_())
            gen_im, _ = self.synthesis([latent_in], input_is_latent=True, return_latents=False, noise=noises)


            final_loss, final_dic = loss_builder_final(gen_im)
            loss_dict['L2_loss'] = final_dic['L2']
            loss_dict['Percep_loss'] = final_dic['percep']


            if self.L2_regularizer > 0:
                if self.In_W_space:
                    latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(self.X_comp.T.unsqueeze(0)) / self.X_stdev
                else:
                    latent_p_norm = torch.stack(latent).unsqueeze(0)
                l2_regularize_loss = self.L2_regularizer*(latent_p_norm.pow(2).mean())
                final_loss += l2_regularize_loss


            total_loss = final_loss
            total_loss.backward()
            opt_final.step()


            gen_im_0_1 = (gen_im + 1) / 2

            # Save best summary for log

            if self.L2_regularizer > 0:
                summary = f'BEST ({j+1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()] + [f'l2_regularize_loss: {l2_regularize_loss.detach().cpu().numpy():.4f}'])
            else:
                summary = f'BEST ({j+1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])


            # print(str(j) + '  ' + summary)
            # Save intermediate HR and LR images
            yield dict(image=gen_im_0_1,
                        lowres_image=None,
                        latent=latent_in,
                        summary=summary)

