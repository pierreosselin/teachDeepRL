import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.autograd import Variable

from einops import rearrange

class DBN(nn.Module): #https://arxiv.org/pdf/1810.04714.pdf
    def __init__(self):
        super(DBN, self).__init__()

    def forward(self, x):
        return torch.heaviside(F.sigmoid(x) - 0.5, torch.tensor([0.0]).cuda())

class Generator_Conv(nn.Module):
    def __init__(self, opt):
        super(Generator_Conv, self).__init__()

        self.init_size = opt.img_size // 4
        self.latent_dim = opt.latent_dim

        self.img_size = opt.img_size
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.binary_neuron = DBN()

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = rearrange(img, 'b c h w -> b (c h w)')
        img = F.sigmoid(img) + self.binary_neuron(img).detach() - F.sigmoid(img).detach()
        img = rearrange(img, 'b (c h w) -> b c h w', c=1, h=self.img_size, w=self.img_size)
        return img

    def generate_random(self, batch_size=1):
        z = Variable(torch.tensor(np.random.normal(0, 1, (batch_size, self.latent_dim)), dtype=torch.float)).cuda()
        return self.forward(z)



class Discriminator_Conv(nn.Module):
    def __init__(self, opt):
        super(Discriminator_Conv, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Discriminator_Conv(nn.Module):
    def __init__(self, opt):
        super(Discriminator_Conv, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 5, 2, 2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 128),
            #*discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 3
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Generator_FC(nn.Module):
    def __init__(self, opt):
        super(Generator_FC, self).__init__()

        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            #nn.Tanh()
        )

        self.binary_neuron = DBN()

    def forward(self, z):
        img = self.model(z)
        img = F.sigmoid(img) + self.binary_neuron(img).detach() - F.sigmoid(img).detach()
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator_FC(nn.Module):
    def __init__(self, opt):
        super(Discriminator_FC, self).__init__()

        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity