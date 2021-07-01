
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

class Embedding(torch.nn.Module):
    def __init__(self, args, verbose=True):
        super(Embedding, self).__init__()

        self.synthesis = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).cuda()
        self.verbose = verbose
        checkpoint = torch.load(args.ckpt)
        self.synthesis.load_state_dict(checkpoint['g_ema'])
        self.latent_avg = checkpoint['latent_avg']

        for param in self.synthesis.parameters():
            param.requires_grad = False

        self.my_downsample = BicubicDownSample(factor=4)

        self.input_dir = args.input_dir

        transformer_space = np.load('transformer_space.npz')
        self.X_mean = torch.from_numpy(transformer_space['X_mean']).float().cuda()
        self.X_comp = torch.from_numpy(transformer_space['X_comp']).float().cuda()
        self.X_stdev = torch.from_numpy(transformer_space['X_stdev']).float().cuda()

        self.NPY_folder = args.NPY_folder
        self.In_W_space = args.In_W_space

        if not self.In_W_space:
            tmp = self.latent_avg.clone().detach().cuda()
            tmp = torch.nn.LeakyReLU(negative_slope=5)(tmp) - self.X_mean
            tmp = tmp.unsqueeze(0).mm(self.X_comp.T).squeeze()
            self.W_to_normalized = tmp / self.X_stdev

        print('In_W_space', self.In_W_space)
        print('lr', args.learning_rate)


    def forward(self, ref_im,
                ref_im_name,
                seed,
                loss_str,
                eps,
                tile_latent,
                opt_name,
                learning_rate,
                steps,
                lr_schedule,
                save_intermediate,
                L2_regularizer,
                l2_regularize_weight,
                **kwargs):
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True


        if (tile_latent):
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


        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        opt_final = opt_dict[opt_name](latent[0:], lr=learning_rate)

        loss_builder_final = LossBuilder(ref_im_name, ref_im, '1.0*L2+1.0*percep', eps, self.input_dir).cuda()

        summary = ""
        start_t = time.time()

        if self.verbose: print("Optimizing")


        for j in range(steps):
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


            if L2_regularizer:
                if self.In_W_space:
                    latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(self.X_comp.T.unsqueeze(0)) / self.X_stdev
                else:
                    latent_p_norm = torch.stack(latent).unsqueeze(0)
                l2_regularize_loss = l2_regularize_weight*(latent_p_norm.pow(2).mean())
                final_loss += l2_regularize_loss


            total_loss = final_loss
            total_loss.backward()
            opt_final.step()


            gen_im_0_1 = (gen_im + 1) / 2

            # Save best summary for log

            if L2_regularizer:
                summary = f'BEST ({j+1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()] + [f'l2_regularize_loss: {l2_regularize_loss.detach().cpu().numpy():.4f}'])
            else:
                summary = f'BEST ({j+1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])


            # print(str(j) + '  ' + summary)
            # Save intermediate HR and LR images
            if (save_intermediate):
                print(str(j) + '  ' + summary)
                if j%50 == 0:
                    yield (gen_im_0_1.cpu().detach().clamp(0, 1), None)


        total_t = time.time() - start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {1}'
        if self.verbose: print(summary + current_info)


        if not os.path.exists(self.NPY_folder):
            os.makedirs(self.NPY_folder)
        np.save(os.path.join(self.NPY_folder, ref_im_name[0] + '.npy'), latent_in.detach().cpu().numpy())


        yield (gen_im_0_1.clone().cpu().detach().clamp(0, 1), None)