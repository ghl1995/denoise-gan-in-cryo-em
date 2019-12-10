import torch.nn as nn
import torch.nn.functional as F
import torch


def batch_norm(x, filter):
    # scope = x.size()[0]
    m = nn.BatchNorm2d(filter)
    return m(x)


def conv2d(input, filter, kernel, stride, pad):
    m = nn.Conv2d(filter, filter, kernel, stride, pad)
    return m(input)


def Identity_block_for_D(X, filter):
    X_shortcut = X
    X1 = F.relu(batch_norm(conv2d(X, filter, kernel=1, stride=1, pad=0), filter))
    X2 = F.relu(batch_norm(conv2d(X1, filter, kernel=3, stride=1, pad=1), filter))
    X3 = batch_norm(conv2d(X2, filter, kernel=1, stride=1, pad=0))
    X4 = torch.add(X_shortcut, 1, X3)
    X5 = F.relu(X4)
    return X5


def deconv2d(input, filter, kernel, stride, pad):
    m = nn.ConvTranspose2d(filter, filter, kernel, stride, pad)
    return m(input)


def Identity_block_for_G(X, filter):
    X_shortcut = X
    X1 = F.relu(batch_norm(conv2d(X, filter, kernel=1, stride=1, pad=0), filter))
    X2 = F.relu(batch_norm(conv2d(X1, filter, kernel=3, stride=1, pad=1), filter))
    X3 = batch_norm(conv2d(X2, filter, kernel=1, stride=1, pad=0))
    X4 = torch.add(X_shortcut, 1, X3)
    X5 = F.relu(X4)
    return X5
