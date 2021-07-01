import torch
from bicubic import BicubicDownSample
import torchvision
import torch.nn as nn
from torchvision import transforms
import numpy as np

import lpips

import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import PIL
from PIL import ImageFilter
import os

def rgb2gray(rgb):

    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:, :, :]
    # gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = 0.2125 * r + 0.7154 * g + 0.0721 * b
    return gray


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss



class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im_name, ref_im, loss_str, eps, input_dir):
        super(LossBuilder, self).__init__()

        self.input_dir = input_dir
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        factor=1024//im_size
        assert im_size*factor==1024
        self.D_VGG = BicubicDownSample(factor=1024 // 256)

        self.D_arc = BicubicDownSample(factor=1024 // 146)


        # self.vgg_mean = torch.from_numpy(np.array([[0.485, 0.456, 0.406]])).float().cuda().reshape(1,3,1,1)
        # self.vgg_std = torch.from_numpy(np.array([[0.229, 0.224, 0.225]])).float().cuda().reshape(1,3,1,1)

        self.vgg_mean = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])).float().cuda().reshape(1, 3, 1, 1)
        self.vgg_std = torch.from_numpy(np.array([[0.5, 0.5, 0.5]])).float().cuda().reshape(1, 3, 1, 1)

        self.ref_im = 2*ref_im - 1
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps
        self.criterionVGG = VGGLoss()
        # self.criterionMSE = torch.nn.SmoothL1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)



        #################################
        self.ref_im_name = ref_im_name
        img = PIL.Image.open(os.path.join(self.input_dir, self.ref_im_name[0]+'.png')).convert('RGB')

        # self.complete_img = 2.0 * torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda() - 1.0
        # img = img.resize((256, 256), PIL.Image.LANCZOS)


        # img = img.filter(ImageFilter.DETAIL)

        self.complete_img = 2.0 * torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda() - 1.0

        img = img.resize((256, 256), PIL.Image.LANCZOS)

        self.sharpen_img = 2.0 * torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda() - 1.0
        self.sharpen_img_0_1 = torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda()



    # def ref_arc_resize_and_crop(self, ref_im_name):
    #     img = PIL.Image.open(os.path.join(self.input_dir, ref_im_name[0] + '.png')).convert('RGB')
    #     img = img.resize((146, 146), PIL.Image.LANCZOS)
    #
    #     arc_ref_im = 2.0 * torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda() - 1.0
    #     arc_ref_im = arc_ref_im[:, [2,1,0]]
    #     arc_ref_im = arc_ref_im[:, :, 17:-17, 17:-17]
    #     return arc_ref_im
    #
    #
    #
    # def arc_resize_and_crop(self, im):
    #
    #     return self.D_arc(im)[:,:,17:-17, 17:-17]





    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        return self.criterionMSE(gen_im_lr, self.complete_img)
        # return self.criterionMSE(self.D_VGG(gen_im_lr), self.sharpen_img)

        # gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256), mode='bilinear', align_corners=True)
        # return self.criterionMSE(gen_im_lr, self.sharpen_img)
        #
        # gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256), mode='bicubic', align_corners=True)
        # return self.criterionMSE(gen_im_lr, self.sharpen_img)

    def _loss_lpips(self, gen_im_lr, ref_im, **kwargs):

        return self.percept(self.D_VGG(gen_im_lr), self.sharpen_img).sum()

        # gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256), mode='bilinear', align_corners=True)
        # return self.percept(gen_im_lr, self.sharpen_img).sum()
        # #
        # gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256), mode='bicubic', align_corners=True)
        # return self.percept(gen_im_lr, self.sharpen_img).sum()




    # def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
    #
    #     # return self.criterionMSE(self.D_VGG(gen_im_lr), self.D_VGG(ref_im))
    #     return self.criterionMSE(gen_im_lr, ref_im)
    #
    #     # return F.mse_loss(gen_im_lr, ref_im)
    #
    # def _loss_lpips(self, gen_im_lr, ref_im, **kwargs):
    #     return self.percept(self.D_VGG(gen_im_lr), self.D_VGG(ref_im)).sum()





    # def _loss_lpips(self, gen_im_lr, ref_im, **kwargs):
    #     gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256),mode='bilinear',align_corners=False)
    #     ref_im = F.interpolate(ref_im,size=(256, 256),mode='bilinear',align_corners=False)
    #     return self.percept(gen_im_lr, ref_im).sum()


    # def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
    #     gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256), mode='bilinear', align_corners=False)
    #     ref_im = F.interpolate(ref_im, size=(256, 256), mode='bilinear', align_corners=False)
    #
    #     return self.criterionMSE(gen_im_lr, ref_im)


    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return self.criterionL1(gen_im_lr, ref_im)


    # def _loss_VGG(self, gen_im_lr, ref_im, **kwargs):
    #     return self.criterionVGG(self.D_VGG(gen_im_lr), self.D_VGG(ref_im))

    def _loss_VGG(self, gen_im_lr, ref_im, **kwargs):
        ref_im = F.interpolate(ref_im, size=(256, 256), mode='bilinear', align_corners=False)
        gen_im_lr = F.interpolate(gen_im_lr, size=(256, 256), mode='bilinear', align_corners=False)
        return self.criterionVGG(gen_im_lr, ref_im)



    def _loss_arc(self, gen_im_lr, **kwargs):
        arc_gen_im_lr = self.arc_resize_and_crop(gen_im_lr[:, [2, 1, 0]])
        our_arc_vector = self.learner.model(arc_gen_im_lr)

        loss = 1.0 - torch.dot(self.arc_ref_vector[0], our_arc_vector[0])

        return loss



    def _loss_ssim(self, gen_im_lr, ref_im, **kwargs):
        new_ref_im = rgb2gray(self.D_VGG((ref_im + 1) / 2.))

        # new_ref_im = rgb2gray(self.sharpen_img_0_1)
        new_gen_im_lr = rgb2gray(self.D_VGG((gen_im_lr + 1) / 2.))



        loss = 1.0 - ssim(new_ref_im, new_gen_im_lr, data_range=1, size_average=True)

        return loss


    def _loss_msssim(self, gen_im_lr, ref_im, **kwargs):
        new_ref_im = rgb2gray(self.D_VGG((ref_im + 1) / 2.))

        # new_ref_im = rgb2gray(self.sharpen_img_0_1)
        new_gen_im_lr = rgb2gray(self.D_VGG((gen_im_lr + 1) / 2.))

        loss = 1.0 - ms_ssim(new_ref_im, new_gen_im_lr, data_range=1, size_average=True)

        return loss




    def forward(self, gen_im):
        var_dict = {
                    'gen_im_lr': gen_im,
                    'ref_im': self.ref_im,
                    }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'VGG':self._loss_VGG,
            'percep': self._loss_lpips,
            'arc': self._loss_arc,
            'ssim': self._loss_ssim,
            'ms_ssim': self._loss_msssim
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        return loss, losses
