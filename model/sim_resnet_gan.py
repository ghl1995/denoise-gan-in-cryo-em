import torch
from model.padding_same import Conv2d_same_padding
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def make_layer_G(filter):
    layer = nn.Sequential(
        nn.Conv2d(filter, filter, 3, 1, 1),
        nn.BatchNorm2d(filter),
    )
    return layer


def make_layer_D(filter):
    layer = nn.Sequential(
        nn.Conv2d(filter, filter, 3, 1, 1),
        nn.BatchNorm2d(filter),
    )
    return layer


class generator(nn.Module):
    def __init__(self, d):
        super(generator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.layere1a = make_layer_G(64)
        self.layere1b = make_layer_G(64)
        self.layere1c = make_layer_G(64)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.layere2a = make_layer_G(128)
        self.layere2b = make_layer_G(128)
        self.layere2c = make_layer_G(128)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.layere3a = make_layer_G(256)
        self.layere3b = make_layer_G(256)
        self.layere3c = make_layer_G(256)
        self.conv10 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.conv10_bn = nn.BatchNorm2d(d * 2)
        self.layerd4a = make_layer_G(128)
        self.layerd4b = make_layer_G(128)
        self.layerd4c = make_layer_G(128)
        self.conv11 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.conv11_bn = nn.BatchNorm2d(d)
        self.layerd5a = make_layer_G(64)
        self.layerd5b = make_layer_G(64)
        self.layerd5c = make_layer_G(64)
        self.conv12 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.conv12_bn = nn.BatchNorm2d(1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, img):
        # encoder
        e1 = self.conv1_bn(self.conv1(img))
        e10 = F.relu(e1)
        e1a = F.relu(torch.add(self.layere1a(e10), 1, e10))
        e1b = F.relu(torch.add(self.layere1b(e1a), 1, e1a))
        e1c = F.relu(torch.add(self.layere1c(e1b), 1, e1b))

        e2 = self.conv2_bn(self.conv2(e1c))
        e20 = F.relu(e2)
        e2a = F.relu(torch.add(self.layere2a(e20), 1, e20))
        e2b = F.relu(torch.add(self.layere2b(e2a), 1, e2a))
        e2c = F.relu(torch.add(self.layere2c(e2b), 1, e2b))

        e3 = self.conv3_bn(self.conv3(e2c))
        e30 = F.relu(e3)
        e3a = F.relu(torch.add(self.layere3a(e30), 1, e30))
        e3b = F.relu(torch.add(self.layere3b(e3a), 1, e3a))
        e3c = F.relu(torch.add(self.layere3c(e3b), 1, e3b))





        # decoder


        d40 = F.relu(torch.add(self.conv10_bn(self.conv10(e3c)), 1, e2))
        d4a = F.relu(torch.add(self.layerd4a(d40), 1, d40))
        d4b = F.relu(torch.add(self.layerd4b(d4a), 1, d4a))
        d4c = F.relu(torch.add(self.layerd4c(d4b), 1, d4b))

        d50 = F.relu(torch.add(self.conv11_bn(self.conv11(d4c)), 1, e1))
        d5a = F.relu(torch.add(self.layerd5a(d50), 1, d50))
        d5b = F.relu(torch.add(self.layerd5b(d5a), 1, d5a))
        d5c = F.relu(torch.add(self.layerd5c(d5b), 1, d5b))

        d60 = self.conv12_bn(self.conv12(d5c))
        return F.tanh(d60)


class discriminator(nn.Module):
    # initializers
    def __init__(self, d):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.layerf1a = make_layer_D(64)
        self.layerf1b = make_layer_D(64)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.layerf2a = make_layer_D(128)
        self.layerf2b = make_layer_D(128)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.layerf3a = make_layer_D(256)
        self.layerf3b = make_layer_D(256)
        self.layerf3c = make_layer_D(256)
        self.conv6 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, cond, is_sigmoid):
        x = cond
        f10 = F.relu(self.conv1_bn(self.conv1(x)))  # 128
        f1a = F.relu(torch.add(self.layerf1a(f10), 1, f10))
        f1b = F.relu(torch.add(self.layerf1b(f1a), 1, f1a))
        f20 = F.relu(self.conv2_bn(self.conv2(f1b)))  # 64
        f2a = F.relu(torch.add(self.layerf2a(f20), 1, f20))
        f2b = F.relu(torch.add(self.layerf2b(f2a), 1, f2a))
        f30 = F.relu(self.conv3_bn(self.conv3(f2b)))  # 32
        f3a = F.relu(torch.add(self.layerf3a(f30), 1, f30))
        f3b = F.relu(torch.add(self.layerf3b(f3a), 1, f3a))
        f3c = F.relu(torch.add(self.layerf3c(f3b), 1, f3b))
        f60 = self.conv6(f3c)  # 26
        if is_sigmoid:
            f60 = F.sigmoid (f60)
            

        return f60
