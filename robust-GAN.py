import time
from skimage.measure import compare_ssim
import mrcfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import criterion
import heapq
from model import sim_dcgan, sim_dcwgan, sim_resnet_gan, exper_resent_gan
import numpy as np
import argparse
import manifold_learning
from skimage.exposure import rescale_intensity
from skimage.util import random_noise
import math
import cv2
import random

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lrd', type=float, default=0.001, help='learning rate of discriminator')
parser.add_argument('--lrg', type=float, default=0.01, help='learning rate of generator')
parser.add_argument('--device', type=str, default='cuda', help='device assignment')
parser.add_argument('--arch', type=str, default='resnet_betagan', help='network architecture')
parser.add_argument('--test_name', type=str, default='test', help='name of test')
parser.add_argument('--train_name', type=str, default='train', help='name of train')
parser.add_argument('--SNR', type=float, default=0.1, help='variance of noise equal (var/255)**2')
parser.add_argument('--sim_train_size', type=int, default=50000, help='size of simulated training samples')
parser.add_argument('--exper_train_size', type=int, default=19500, help='size of exper training samples')
parser.add_argument('--test_size', type=int, default=500, help='size of training samples')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--proportion', type=float, default=0)
parser.add_argument('--robusttype', type=int, default=0)
parser.add_argument('--lambda1', type=float, default=10)
parser.add_argument('--lambda2', type=float, default=10)
parser.add_argument('--regularization', type=str, default='l1', help='regularization type')
parser.add_argument('--datatype', type=str, default='sim')
parser.add_argument('--d', type=int, default=64, help='first latent convolution filter')
opt = parser.parse_args()

# training parameters
epsilon = opt.epsilon
d = opt.d
batch_size = opt.batchSize
img_size = opt.imageSize
lrd = opt.lrd
lrg = opt.lrg
SNR = opt.SNR
robusttype = opt.robusttype
train_epoch = opt.nepoch
sim_train_size = opt.sim_train_size
exper_train_size = opt.exper_train_size
test_name = opt.test_name
train_name = opt.train_name
test_size = opt.test_size
alpha = opt.alpha
beta = opt.beta
arch = opt.arch
datatype = opt.datatype
lambda1 = opt.lambda1
lambda2 = opt.lambda2
regularization = opt.regularization
is_sigmoid = True
mname = datatype + '_' + opt.arch + '_' + opt.regularization + '_alpha' + str(alpha) + '_beta' + str(
    beta) + '_SNR' + str(SNR) + '_regulalambda' + str(lambda1) + '_robusttype' + str(robusttype) + '_proportion' + str(
    opt.proportion)

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device(opt.device)
else:
    device = torch.device(opt.device)

# set folder
if not os.path.isdir('./save_net'):
    os.mkdir('./save_net')
if not os.path.isdir('./save_net/' + mname):
    os.mkdir('./save_net/' + mname)
if not os.path.isdir('./save_img'):
    os.mkdir('./save_img')
if not os.path.isdir('./save_img/' + mname):
    os.mkdir('./save_img/' + mname)
if not os.path.isdir('./temp_img'):
    os.mkdir('./temp_img')
log = open('./save_img/' + mname + '/log.txt', 'w')  ###log file

##data
if datatype == 'sim':
    train_noisy_data = mrcfile.open('./data/' + train_name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc').data
    train_clean_data = mrcfile.open('./data/' + train_name + '_clean.mrc').data
    test_noisy_data = mrcfile.open('./data/' + test_name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc').data
    test_clean_data = mrcfile.open('./data/' + test_name + '_clean.mrc').data
else:
    noisy_data = mrcfile.open('./data/exper_20000_noisy.mrc').data
    clean_data = mrcfile.open('./data/exper_20000_clean.mrc').data
    train_noisy_data = noisy_data[0:exper_train_size]
    train_clean_data = clean_data[0:exper_train_size]
    test_noisy_data = noisy_data[exper_train_size:exper_train_size + test_size]
    test_clean_data = clean_data[exper_train_size:exper_train_size + test_size]

###network
if datatype == 'sim':
    if arch == 'resnet_betagan' or arch == 'autoencoder':
        G = sim_resnet_gan.generator(d)
        D = sim_resnet_gan.discriminator(d, is_sigmoid = True)
    elif arch == 'resnet_wgangp':
        G = sim_resnet_gan.generator(d)
        D = sim_resnet_gan.discriminator(d, is_sigmoid = False)
        
    elif arch == 'dcgan':
        G = sim_dcgan.generator(d)
        D = sim_dcgan.discriminator(d)
    elif arch == 'dcwgan':
        G = sim_dcwgan.generator(d)
        D = sim_dcwgan.discriminator(d)
else:
    if arch == 'resnet_betagan' or arch == 'autoencoder':
        G = exper_resnet_gan.generator(d)
        D = exper_resnet_gan.discriminator(d, is_sigmoid = True)
    elif arch == 'resnet_wgangp':
        G = exper_resnet_gan.generator(d)
        D = exper_resnet_gan.discriminator(d, is_sigmoid = False)
        


###train
def train():
    ###loss and init
    random_list = random.sample(range(0, int(sim_train_size / batch_size), 2),
                                int(int(sim_train_size / batch_size / 2) * opt.proportion))  ###change to noise list
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G_optimizer = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.999))
    G.cuda()
    D.cuda()
    train_MSE = []
    train_PSNR = []
    train_SSIM = []
    for epoch in range(train_epoch):
        k = 0
        epoch_start_time = time.time()
        train_range = range(0, int(sim_train_size / batch_size), 2) if datatype == 'sim' else train_range = range(
            exper_train_size)
        for i in train_range:
            cond = noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
            img = clean_data[i * batch_size: i * batch_size + batch_size, :, :]
            ## test robust
            if robusttype == 1:
                if i in random_list:
                    img = np.random.rand(batch_size, img_size, img_size)
            elif robusttype == 2:
                if i in random_list:
                    cond = np.random.rand(batch_size, img_size, img_size)
            elif robusttype == 3:
                if i in random_list:
                    cond = np.random.rand(batch_size, img_size, img_size)
                    img = np.random.rand(batch_size, img_size, img_size)
            k = k + 1
            pimg2 = np.reshape(img, (batch_size, 1, img_size, img_size))
            pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
            pimg2 = torch.from_numpy(pimg2)
            pcond2 = torch.from_numpy(pcond2)
            pimg2 = pimg2.to(device=device, dtype=dtype)
            pcond2 = pcond2.to(device=device, dtype=dtype)

            ##wgan
            ###d_step
            if arch[:-3] == 'wgan':
                if k % 2 == 0:
                    D.zero_grad()
                    D_result = D(pimg2)
                    D_result = D_result.squeeze()
                    G_result = G(pcond2)
                    D_train_loss = -torch.mean(D_result) + torch.mean(D(G_result))
                    D_train_loss.backward(retain_graph=True)
                    D_optimizer.step()

                    if opt.arch == 'wgangp':
                        alpha = torch.rand(pimg2.size(0), 1, 1, 1).to(device)
                        interpolates = (alpha * pimg2 + (1 - alpha) * G_result).requires_grad_(True)
                        d_interpolates = D(interpolates)
                        gradients = torch.autograd.grad(
                            outputs=d_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones_like(d_interpolates),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
                        gradients = gradients.view(gradients.size(0), -1)
                        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
                        gradient_penalty.backward()
                        D_train_loss += gradient_penalty.item()

                    # Clip weights of discriminator
                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                #  G_step

                G.zero_grad()
                G_result = G(pcond2)
                D_result = D(G_result).squeeze()
                if regularization == 'l2':
                    loss = torch.mean(torch.abs(G_result - pimg2) ** 2)
                else:
                    loss = torch.mean(torch.abs(G_result - pimg2))
                G_train_loss = -torch.mean(D_result) + lambda1 * loss
                G_train_loss.backward(retain_graph=True)
                G_optimizer.step()

            #####autoencoder
            if arch[:-3] == 'oder':
                G.zero_grad()
                G_result = G(pcond2)
                if regularization == 'l2':
                    loss = torch.mean(torch.abs(G_result - pimg2) ** 2)
                else:
                    loss = torch.mean(torch.abs(G_result - pimg2))
                loss.backward()
                G_optimizer.step()

            ####Betagan
            else:
                ###d_step
                if k % 2 == 0:
                    D.zero_grad()
                    D_real_score = D(pimg2).squeeze()
                    D_real_score = torch.clamp(D_real_score, 0 + epsilon,
                                               1 - epsilon)  ### restrict the score in the range(epsilon, 1-epsilon) to avoid collapse
                    D_real_coef = -D_real_score ** (alpha - 1) * (1 - D_real_score) ** beta
                    D_real_score.backward(D_real_coef / batch_size)

                    # D_real_loss = BCE_loss(D_result, y_real_)
                    D_fake_score = D(G(pcond2)).squeeze()
                    D_fake_score = torch.clamp(D_fake_score, 0 + epsilon, 1 - epsilon)
                    D_fake_coef = D_fake_score ** alpha * (1 - D_fake_score) ** (beta - 1)
                    D_fake_score.backward(D_fake_coef / batch_size)
                    D_optimizer.step()

                #  G_step

                G.zero_grad()
                G_result = G(pcond2)
                G_score = D(G(pcond2)).squeeze()
                G_score = torch.clamp(G_score, 0 + epsilon, 1 - epsilon)
                G_coef = -G_score ** alpha * (1 - G_score) ** (beta - 1)
                G_score.backward((G_coef) / batch_size)
                ### penalty
                if regularization == 'l2':
                    penalty = lambda1 * torch.mean(torch.abs(G_result - pimg2) ** 2)
                else:
                    penalty = lambda1 * torch.mean(torch.abs(G_result - pimg2))

                penalty.backward()
                G_optimizer.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            if k % 200 == 0:
                MSE = []
                PSNR = []
                SSIM = []
                log.write(
                    '[%d/%d] - lrd:%.5f, lrg:%.5f, lgptime: %.2f,l1loss: %.3f, l2loss: %.5f\n' % (
                        k, epoch, lrd, lrg, per_epoch_ptime,
                        torch.mean(torch.abs(G_result - pimg2)),
                        torch.mean((G_result - pimg2) ** 2)))
                for t in range(batch_size):
                    a = G_result[t].cpu().detach().numpy()
                    a = np.reshape(a, (img_size, img_size))
                    m = criterion.MSE(a, img[t])
                    MSE.append(m)
                    PSNR.append(10 * math.log10(1 / m))
                    SSIM.append(compare_ssim(a, img[t]))
                train_MSE.append(np.mean(MSE))
                train_PSNR.append(np.mean(PSNR))
                train_SSIM.append(np.mean(SSIM))
        ##save_net
        torch.save(G.state_dict(),
                   './save_net/' + mname + '/G_%d' % (epoch))
        torch.save(D.state_dict(),
                   './save_net/' + mname + '/D_%d' % (epoch))
    np.save('./save_net/' + mname + '/test_mse.npy', MSE)
    np.save('./save_net/' + mname + '/test_psnr.npy', PSNR)
    np.save('./save_net/' + mname + '/train_MSE.npy', train_MSE)
    np.save('./save_net/' + mname + '/train_PSNR.npy', train_PSNR)
    np.save('./save_net/' + mname + '/train_SSIM.npy', train_SSIM)

    log.write("train\n")
    log.write('MSE-std:%s %s\n' % (np.mean(train_MSE), np.std(train_MSE)))
    log.write('PSNR-std:%s %s\n' % (np.mean(train_PSNR), np.std(train_PSNR)))
    log.write('SSIM-std:%s %s\n' % (np.mean(train_SSIM), np.std(train_SSIM)))

    return train_MSE, train_SSIM


def test(train_MSE, train_SSIM):
    test_PSNR = []
    test_MSE = []
    test_SSIM = []
    for j in range(train_epoch):
        pretrained_net = torch.load('./save_net/' + mname + '/G_' + str(j))
        G.load_state_dict(pretrained_net)
        MSE = []
        PSNR = []
        SSIM = []
        for i in range(int(test_size / batch_size)):
            cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
            img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
            pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
            pcond2 = torch.from_numpy(pcond2)
            pcond2 = pcond2.to(device=device, dtype=dtype)
            for t in range(batch_size):
                gen_img = G(pcond2)[t].cpu().detach().numpy()
                gen_img = gen_img.reshape((img_size, img_size))
                m = criterion.MSE(gen_img, img[t])
                MSE.append(m)
                PSNR.append(10 * math.log10(1 / m))
                SSIM.append(compare_ssim(gen_img, img[t]))
            ###save denoise fig
            if i % 20 == 0:
                plt.imshow(gen_img, cmap='gray')
                plt.savefig('./save_img/' + mname + '/denoise_%d.png' % i)
                clean_img = np.reshape(img[t], (img_size, img_size))
                plt.imshow(clean_img, cmap='gray')
                plt.savefig('./save_img/' + mname + '/clean_%d.png' % i)

        test_MSE.append(np.mean(MSE))
        test_PSNR.append(np.mean(PSNR))
        test_SSIM.append(np.mean(SSIM))

    log.write("test\n")
    log.write('MSE-std:%s %s\n' % (np.mean(test_MSE), np.std(test_MSE)))
    log.write('PSNR-std:%s %s\n' % (np.mean(test_PSNR), np.std(test_PSNR)))
    log.write('SSIM-std:%s %s\n' % (np.mean(test_SSIM), np.std(test_SSIM)))

    log.write('5th mse, psnr ssim\n')
    minlist = list(map(train_MSE.index, heapq.nsmallest(5, train_MSE)))
    maxlist = list(map(train_SSIM.index, heapq.nlargest(5, train_SSIM)))
    fifth = []
    for i in minlist:
        fifth.append(test_MSE[i])
    log.write('MSE-std:%s %s\n' % (np.mean(fifth), np.std(fifth)))
    fifth = []
    for i in minlist:
        fifth.append(test_PSNR[i])
    log.write('PSNR-std:%s %s\n' % (np.mean(fifth), np.std(fifth)))
    fifth = []
    for i in maxlist:
        fifth.append(test_SSIM[i])
    log.write('SSIM-std:%s %s\n' % (np.mean(fifth), np.std(fifth)))
    log.close()

    ###plt 5th image
    plt.title('train and test MSE')
    plt.plot(np.arange(train_epoch), train_MSE, color='r', label='train')
    plt.plot(np.arange(train_epoch), test_MSE, color='b', label='test')
    plt.legend(loc='upper right')
    plt.savefig('./save_img/' + mname + '/train_test.png')


if __name__ == '__main__':
    train_MSE, train_SSIM = train()
    test(train_MSE, train_SSIM)



