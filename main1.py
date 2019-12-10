import time
import mrcfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import criterion
from model import dcgan, wgan
import numpy as np
import argparse
import manifold_learning
import math

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lrd', type=float, default=0.001, help='learning rate of discriminator')
parser.add_argument('--lrg', type=float, default=0.01, help='learning rate of generator')
parser.add_argument('--device', type=str, default='cuda', help='device assignment')
parser.add_argument('--arch', type=str, default='jsgan', help='network architecture')
parser.add_argument('--test_name', type=str, default='test', help='name of test')
parser.add_argument('--train_name', type=str, default='train', help='name of train')
parser.add_argument('--SNR', type=float, default=0.1, help='variance of noise equal (var/255)**2')
parser.add_argument('--train_size', type=int, default=50000, help='size of training samples*2')
parser.add_argument('--test_size', type=int, default=1500, help='size of test samples*2')
parser.add_argument('--alpha', type=float, default=-0.5, help='aplha parameter in betagan')
parser.add_argument('--beta', type=float, default=-0.5, help='beta parameter in betagan')
parser.add_argument('--epsilon', type=float, default=0.01, help='avoid the model collapse in betagan')
parser.add_argument('--save_pic_time', type =float, default=200, help='time to show image')

opt = parser.parse_args()

# Parameters
epsilon = opt.epsilon
batch_size = opt.batchSize
lrd = opt.lrd
lrg = opt.lrg
SNR = opt.SNR
mname = opt.arch + 'SNR' + str(SNR)
train_epoch = opt.nepoch
size = opt.train_size
img_size = opt.imageSize
test_name = opt.test_name
train_name = opt.train_name
train_size = opt.train_size
test_size = opt.test_size
save_pic_time = opt.save_pic_time
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
        G = dcgan.generator()
        D = dcgan.discriminator()
    elif opt.arch == 'wgan':
        G = wgan.generator()
        D = wgan.discriminator()
    elif opt.arch == 'wgangp':
        G = wgan.generator()
        D = wgan.discriminator()
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
    print(criterion.MSE(noisy_data[0], clean_data[0]))
    print(np.max(test_noisy_data[0]))
    print(np.max(test_clean_data[0]))
    D_loss = []
    G_loss = []
    mse = []
    psnr = []
    for epoch in range(train_epoch):
        k = 0
        epoch_start_time = time.time()
        for i in range(0, int(train_size / batch_size), 2):
            cond = noisy_data[i * batch_size: i * batch_size + batch_size, :, :]  # noisy image as one input
            img = clean_data[i * batch_size: i * batch_size + batch_size, :, :]  # clean image as the other input
            pimg2 = np.reshape(img, (batch_size, 1, img_size, img_size))
            pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
            pimg2 = torch.from_numpy(pimg2).to(device=device, dtype=dtype)
            pcond2 = torch.from_numpy(pcond2).to(device=device, dtype=dtype)
            k = k + 1
            ##wgan
            if opt.arch == 'wgan' or opt.arch == 'wgangp':
                # d step, updata twice g steps in each d step
                if k % 2 == 0:
                    D.zero_grad()
                    D_result = D(pimg2)
                    D_result = D_result.squeeze()
                    G_result = G(pcond2)
                    D_train_loss = -torch.mean(D_result) + torch.mean(D(G_result))
                    D_train_loss.backward(retain_graph=True)
                    D_optimizer.step()
                    if opt.arch == 'wgangp':  # add gradient penalty
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
                G.zero_grad()
                G_result = G(pcond2)
                D_result = D(G_result).squeeze()
                G_train_loss = BCE_loss(D_result, y_real_) + 10 * torch.mean(torch.abs(G_result - pimg2))
                G_train_loss.backward(retain_graph=True)
                G_optimizer.step()
                G_loss.append(G_train_loss.item())

            # betagan
            if opt.arch == 'betagan1' or opt.arch == 'betagan2':
                alpha = opt.alpha
                beta = opt.beta
                if k % 2 == 0:
                    D.zero_grad()
                    D_real_score = D(pimg2).squeeze()
                    D_real_score = torch.clamp(D_real_score, 0 + epsilon, 1 - epsilon)
                    D_real_coef = -D_real_score ** (alpha - 1) * (1 - D_real_score) ** beta
                    D_real_score.backward(D_real_coef / batch_size)
                    D_fake_score = D(G(pcond2)).squeeze()
                    D_fake_score = torch.clamp(D_fake_score, 0 + epsilon, 1 - epsilon)
                    D_fake_coef = D_fake_score ** alpha * (1 - D_fake_score) ** (beta - 1)
                    D_fake_score.backward(D_fake_coef / batch_size)
                    D_optimizer.step()
                G.zero_grad()
                G_result = G(pcond2)
                G_score = D(G(pcond2)).squeeze()
                G_score = torch.clamp(G_score, 0 + epsilon, 1 - epsilon)
                G_coef = -G_score ** alpha * (1 - G_score) ** (beta - 1)
                G_score.backward((G_coef) / batch_size)
                penalty = 10 * torch.mean(torch.abs(G_result - pimg2))
                penalty.backward()
                G_optimizer.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time  #run time
            ###output results
            if k % int(save_pic_time/10) == 0:
                print(
                    '[%d/%d] - lrd:%.5f, lrg:%.5f, lgptime: %.2f,l1loss: %.3f, l2loss: %.5f' % (
                        k, epoch, lrd, lrg, per_epoch_ptime,
                        10 * torch.mean(torch.abs(G_result - pimg2)),
                        torch.mean((G_result - pimg2) ** 2)))
            ####save image to show
            if k % save_pic_time == 0:
                for t in range(batch_size):
                    a = G_result[t].cpu().detach().numpy()
                    a = np.reshape(a, (img_size, img_size))
                    m = criterion.MSE(a, img[t])
                    mse.append(m)
                    psnr.append(10 * math.log(1 / m))
                plt.imshow(a, cmap='gray')
                plt.savefig('./temp_img/' + str(epoch) + '_' + str(k / 200) + '.png')
        np.save('./save_net/' + mname + '/train_mse.npy', mse)
        np.save('./save_net/' + mname + '/train_psnr.npy', psnr)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        torch.save(G.state_dict(),
                   './save_net/' + mname + '/G_%d' % (epoch))
        torch.save(D.state_dict(),
                   './save_net/' + mname + '/D_%d' % (epoch))
    np.save('./save_net/' + mname + '/G_loss.npy', G_loss)
    np.save('./save_net/' + mname + '/D_loss.npy', D_loss)


##test
def test():
    ##test data
    test_noisy_data = mrcfile.open('./data/' + test_name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc').data
    test_clean_data = mrcfile.open('./data/' + test_name + '_clean.mrc').data
    G = dcgan.generator(64).cuda()
    pretrained_net = torch.load(
        './save_net/' + mname + '/G_' + str(
            17))  # torch.load('D:\cryo_em\code\denoise_code\denoise_gan\save_net\gan\model9\G_19')
    G.load_state_dict(pretrained_net)
    MSE = []
    PSNR = []
    k = 0
    for i in range(int(test_size / batch_size)):
        cond = test_noisy_data[i * batch_size: i * batch_size + batch_size, :, :]
        img = test_clean_data[i * batch_size: i * batch_size + batch_size, :, :]
        k = k + 1
        pcond2 = np.reshape(cond, (batch_size, 1, img_size, img_size))
        pcond2 = torch.from_numpy(pcond2)
        pcond2 = pcond2.to(device=device, dtype=dtype)
        for t in range(batch_size):
            gen_img = G(pcond2)[t].cpu().detach().numpy()
            gen_img = gen_img.reshape((img_size, img_size))
            m = criterion.MSE(gen_img, img[t])
            MSE.append(m)
            PSNR.append(10 * math.log10(1 / m))
    np.save('./save_net/' + mname + '/test_mse.npy', MSE)
    np.save('./save_net/' + mname + '/test_psnr.npy', PSNR)


if __name__ == '__main__':
    train()
    test()
