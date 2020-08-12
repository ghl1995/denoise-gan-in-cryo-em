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
from model import dcgan, wgan,  exper_conv_model, sim_resnet
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
parser.add_argument('--arch', type=str, default='betagan1', help='network architecture')
parser.add_argument('--test_name', type=str, default='test', help='name of test')
parser.add_argument('--train_name', type=str, default='train', help='name of train')
parser.add_argument('--SNR', type=float, default=0.1, help='variance of noise equal (var/255)**2')
parser.add_argument('--train_size', type=int, default=6000, help='size of training samples')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--proportion', type=float, default=0)
parser.add_argument('--type', type=int, default=2)
parser.add_argument('--lamda', type = float, default=10)
opt = parser.parse_args()

# training parameters
epsilon = opt.epsilon
batch_size = opt.batchSize
lrd = opt.lrd
lrg = opt.lrg
SNR = opt.SNR
mname = 'resnet' +  opt.arch + 'alpha' + str(opt.alpha) +'_beta' + str(opt.beta) + '_SNR' + str(SNR) + '_lamda' +str(opt.lamda)
train_epoch = opt.nepoch
size = opt.train_size
img_size = opt.imageSize
test_name = opt.test_name
train_name = opt.train_name

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


###train
def train():
    ###network
    if opt.arch == 'jsgan' or opt.arch == 'betagan1' or opt.arch == 'betagan2':
       # G = exper.generator()
       # D = dcgan.discriminator()
        G = sim_resnet.generator(64)
        D = sim_resnet.discriminator(64)

    elif opt.arch == 'wgan':
        G = wgan.generator()
        D = wgan.discriminator()
    elif opt.arch == 'wgangp':
        G = sim_resnet.generator(64)
        D = sim_resnet.discriminator(64)
    BCE_loss = nn.BCELoss()  # Binary Cross Entropy loss
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G_optimizer = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.999))
    G.cuda()
    D.cuda()

    ##data
    noisy_data = mrcfile.open('./data/' + train_name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc').data
    clean_data = mrcfile.open('./data/' + train_name + '_clean.mrc').data
    test_noisy_data = mrcfile.open('./data/' + test_name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc').data
    test_clean_data = mrcfile.open('./data/' + test_name + '_clean.mrc').data
    list = random.sample(range(0, 2500, 2), int(1250 * opt.proportion))
    D_loss = []
    G_loss = []
    mse = []
    mse1 = []
    psnr = []
    MSE1 = []
    PSNR = []
    for epoch in range(train_epoch):
        k = 0
        epoch_start_time = time.time()
        for i in range(0, int(50000 / batch_size), 2):
            cond = noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
            img = clean_data[i * batch_size: i * batch_size + batch_size, :, :]
            if opt.type == 1:
                if i in list:
                    cond = np.random.rand(batch_size, img_size, img_size)
            elif opt.type == 2:
                if i in list:
                    img = np.random.rand(batch_size, img_size, img_size)
            elif opt.type == 3:
                if i in list:
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
            if opt.arch == 'wgan' or opt.arch == 'wgangp':
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

                    # D_losses.append(D_train_loss.data[0])
                    D_loss.append(D_train_loss.item())

                    # Clip weights of discriminator
                    for p in D.parameters():
                        p.data.clamp_(-0.01, 0.01)

                #  G_step

                G.zero_grad()
                G_result = G(pcond2)
                D_result = D(G_result).squeeze()
                G_train_loss = -torch.mean(D_result)  # + 100 * torch.mean(torch.abs(G_result - pimg2))
                G_train_loss.backward(retain_graph=True)
                G_optimizer.step()

            ####jsgan
            if opt.arch == 'jsgan':
                y_real_ = torch.ones((batch_size, 5, 5))
                y_fake_ = torch.zeros((batch_size, 5, 5))
                y_real_ = y_real_.to(device=device, dtype=dtype)
                y_fake_ = y_fake_.to(device=device, dtype=dtype)
                if k % 2 == 0:
                    ###d_step
                    D.zero_grad()
                    D_result = D(pimg2)
                    D_result = D_result.squeeze()
                    D_real_loss = BCE_loss(D_result, y_real_)
                    G_result = G(pcond2)
                    D_result = D(G_result).squeeze()
                    D_fake_loss = BCE_loss(D_result, y_fake_)
                    D_train_loss = D_real_loss + D_fake_loss
                    D_train_loss.backward()
                    D_optimizer.step()
                    D_loss.append(D_train_loss.item())

                #  G_step
                G.zero_grad()
                G_result = G(pcond2)
                D_result = D(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_) + 10 * torch.mean(torch.abs(G_result - pimg2))
                G_train_loss.backward(retain_graph=True)
                G_optimizer.step()
                G_loss.append(G_train_loss.item())

            if opt.arch == 'betagan1' or opt.arch == 'betagan2':
                alpha = opt.alpha
                beta = opt.beta
                ###d_step
                if k % 2 == 0:
                    D.zero_grad()
                    D_real_score = D(pimg2).squeeze()
                    D_real_score = torch.clamp(D_real_score, 0 + epsilon, 1 - epsilon)
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
                # G_score = torch.clamp(D_result, 0 + epsilon, 1 - epsilon)
                # G_train_loss = 0.5 * BCE_loss(D_result, y_real_) + conf.L1_lambda1 * (
                # torch.mean(torch.abs(G_result - pimg2)))
                # print(G_coef)

                G_score.backward((G_coef) / batch_size)
                ###gradient penalty
                penalty = opt.lamda * torch.mean(torch.abs(G_result - pimg2))
                penalty.backward()
                G_optimizer.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            if k % 200 == 0:
                print(
                    '[%d/%d] - lrd:%.5f, lrg:%.5f, lgptime: %.2f,l1loss: %.3f, l2loss: %.5f' % (
                        k, epoch, lrd, lrg, per_epoch_ptime,
                        10 * torch.mean(torch.abs(G_result - pimg2)),
                        torch.mean((G_result - pimg2) ** 2)))

            if k % 200 == 0:
                for t in range(batch_size):
                    a = G_result[t].cpu().detach().numpy()
                    a = np.reshape(a, (img_size, img_size))
                    m = criterion.MSE(a, img[t])
                    mse.append(m)
                    psnr.append(10 * math.log(1 / m))
                plt.imshow(a, cmap='gray')
                plt.savefig('./temp_img/' + str(epoch) + '_' + str(k / 200) + '.png')
                print('mse:%.6f' % np.mean(mse))
                mse1.append(np.mean(mse))
                # test_data = []

                '''for i in range(int(200 / batch_size)):
                    cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
                    img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
                    pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
                    pcond2 = torch.from_numpy(pcond2)
                    pcond2 = pcond2.to(device=device, dtype=dtype)
                    for t in range(batch_size):
                        gen_img = G(pcond2)[t].cpu().detach().numpy()
                        gen_img = gen_img.reshape((img_size, img_size))
                        # test_data.append(gen_img)
                        m = criterion.MSE(gen_img, img[t])
                        MSE.append(m)
                        #MSE1.append(np.mean(MSE))
                        PSNR.append(10 * math.log(1 / m))
                print('MSE:%.6f' % np.mean(MSE))'''
        MSE = []
        PSNR = []
        for i in range(int(1500 / batch_size)):
            cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
            img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
            pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
            pcond2 = torch.from_numpy(pcond2)
            pcond2 = pcond2.to(device=device, dtype=dtype)
            for t in range(batch_size):
                gen_img = G(pcond2)[t].cpu().detach().numpy()
                gen_img = gen_img.reshape((img_size, img_size))
                # test_data.append(gen_img)
                m = criterion.MSE(gen_img, img[t])
                MSE.append(m)
                # MSE1.append(np.mean(MSE))
                PSNR.append(10 * math.log(1 / m, 10))
        print('MSE:%.6f' % np.mean(MSE))
        print('PSNR:%.6f'% np.mean(PSNR))
        np.save('./save_net/' + mname + '/train_mse.npy', mse)
        np.save('./save_net/' + mname + '/train_psnr.npy', psnr)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        torch.save(G.state_dict(),
                   './save_net/' + mname + '/G_%d' % (epoch))
        torch.save(D.state_dict(),
                   './save_net/' + mname + '/D_%d' % (epoch))

        # print('MSE:%.6f' % m)
        # manifold_learning.plot_tsne(epoch, test_data, mname)
        np.save('./save_net/' + mname + '/test_mse.npy', MSE)
        np.save('./save_net/' + mname + '/test_psnr.npy', PSNR)
    np.save('./save_net/' + mname + '/G_loss.npy', G_loss)
    np.save('./save_net/' + mname + '/D_loss.npy', D_loss)


##test
'''def ttest():
    ##data
    test_noisy_data = mrcfile.open('./data/' + test_name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc').data
    test_clean_data = mrcfile.open('./data/' + test_name + '_clean.mrc').data
    G = sim_resnet.generator(64).cuda()
    pretrained_net = torch.load(
        './save_net/' + mname + '/G_' + str(
            19))  # torch.load('D:\cryo_em\code\denoise_code\denoise_gan\save_net\gan\model9\G_19')
    G.load_state_dict(pretrained_net)
    MSE = []
    PSNR = []
    ssim = []
    img1 = []
    k = 0
    for i in range(int(1500 / batch_size)):
        cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
        img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
        k = k + 1
        pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
        pcond2 = torch.from_numpy(pcond2)
        pcond2 = pcond2.to(device=device, dtype=dtype)
        for t in range(batch_size):
            gen_img = G(pcond2)[t].cpu().detach().numpy()
            gen_img = gen_img.reshape((img_size, img_size))
            #m = criterion.l1loss(gen_img, img[t])
            ssim.append(compare_ssim(gen_img, img[t]))
            m = criterion.MSE( gen_img, img[t])
            MSE.append(m)
            PSNR.append(10 * math.log10(1 / m))
        if i % 20 == 0:
            plt.imshow(gen_img, cmap = 'gray')
            plt.savefig('./save_img/'+ mname +'/denoise_%d.png' %i)
            clean_img = np.reshape(img[t], (img_size, img_size))
            plt.imshow(clean_img, cmap = 'gray')
            plt.savefig('./save_img/'+ mname + '/clean_%d.png' %i)

    print(np.mean(MSE))
    print(np.std(MSE))
    print(np.mean(PSNR))
    print(np.std(PSNR))
    print(np.mean(ssim))
    print(np.std(ssim))
    # np.save('./save_net/' + mname + '/test_mse.npy', MSE)
    # np.save('./save_net/' + mname + '/test_psnr.npy', PSNR)'''

def ttest():
    ##data
    if not os.path.isdir('./save_img/' + mname + '/denoise_150/'):
       os.mkdir('./save_img/' + mname + '/denoise_150/')
    path = './data/150_clean/'
    listdir = os.listdir(path)
    G = sim_resnet.generator(64).cuda()
    pretrained_net = torch.load(
        './save_net/' + mname + '/G_' + str(
            19))  # torch.load('D:\cryo_em\code\denoise_code\denoise_gan\save_net\gan\model9\G_19')
    G.load_state_dict(pretrained_net)
    MSE = []
    PSNR = []
    ssim = []
    img1 = []
    k = 0
    for i in np.sort(listdir):
        batch_size = 1
        img = cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE)
        img.shape
        img = rescale_intensity(1.0 * img, out_range=(0, 1)).astype(np.float32)
        cond= random_noise(img, mode='gaussian',
                           var=img.var() / 0.1)
        k = k + 1
        pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
        pcond2 = torch.from_numpy(pcond2)
        pcond2 = pcond2.to(device=device, dtype=dtype)
        gen_img = G(pcond2)[0].cpu().detach().numpy()
        gen_img = gen_img.reshape((img_size, img_size))
            #m = criterion.l1loss(gen_img, img[t])
        plt.axis('off')
        plt.imshow(gen_img, cmap='gray')
        plt.savefig('./save_img/' + mname + '/denoise_150/' + i, bbox_inches='tight', pad_inches=0.0)
        ssim.append(compare_ssim(gen_img, img))
        m = criterion.MSE( gen_img, img)
        MSE.append(m)
        PSNR.append(10 * math.log10(1 / m))


    print(np.mean(MSE))
    print(np.std(MSE))
    print(np.mean(PSNR))
    print(np.std(PSNR))
    print(np.mean(ssim))
    print(np.std(ssim))
if __name__ == '__main__':
    #train()
    ttest()


