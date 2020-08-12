import random
import time
import mrcfile
import warnings
from PIL import Image
from pg_model import *
import criterion
import math
from skimage.measure import compare_ssim
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import torch.utils.data as udata
import torchvision.datasets as vdatasets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt


test_noisy_data = mrcfile.open('./data/test_SNR0.05_Gaussian.mrc').data
test_clean_data = mrcfile.open('./data/test_clean.mrc').data

resolution_ = 128
latent_size_ = 2
rgb_channel_ = 1
fmap_base_ = 2 ** 11
fmap_decay_ = 1.0
fmap_max_ = 2 ** 7
is_tanh_ = True
is_sigmoid_ = False
batch_size = 8
img_size = 128
g_net = Generator(resolution_, latent_size_, rgb_channel_,
                  fmap_base_, fmap_decay_, fmap_max_, is_tanh=is_tanh_).cuda()
d_net = Discriminator(resolution_, rgb_channel_,
                      fmap_base_, fmap_decay_, fmap_max_, is_sigmoid=is_sigmoid_).cuda()
e_net = Encoder(resolution_, rgb_channel_,
                fmap_base_, fmap_decay_, fmap_max_, is_sigmoid=is_sigmoid_).cuda()
net_level = 5
net_status = "stable"
net_alpha = 1
g_net.net_config = [net_level, net_status, net_alpha]
e_net.net_config = [net_level, net_status, net_alpha]
d_net.net_config = [net_level, net_status, net_alpha]
print(g_net.net_status_)
g_net.load_state_dict(torch.load('/home/hguaf/MSML/PGGAN/WGANgp_l1/Gnet_128x128.pth'))
d_net.load_state_dict(torch.load('/home/hguaf/MSML/PGGAN/WGANgp_l1/Dnet_128x128.pth'))
e_net.load_state_dict(torch.load('/home/hguaf/MSML/PGGAN/WGANgp_l1/Enet_128x128.pth'))

MSE = []
PSNR = []
ssmi = []
k = 0
for i in range(int(1500 / batch_size)):
    cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
    img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
    k = k + 1
    pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
    pcond2 = torch.from_numpy(pcond2).cuda()
    for t in range(batch_size):
        gen_img = g_net(e_net((pcond2)))[t].cpu().detach().numpy()
        gen_img = gen_img.reshape((img_size, img_size))
        # m = criterion.l1loss(gen_img, img[t])
        m = criterion.MSE(gen_img, img[t])
        ssmi.append(compare_ssim(gen_img, img[t]))
        MSE.append(m)
        PSNR.append(10 * math.log10(1 / m))

plt.imshow(gen_img, cmap='gray')
plt.savefig('1.png')
plt.imshow(img[t], cmap='gray')
plt.savefig('2.png')
print(np.mean(MSE))
print(np.mean(ssmi))
print(np.mean(PSNR))
