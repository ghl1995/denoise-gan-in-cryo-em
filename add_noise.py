import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.util import random_noise
import mrcfile
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--SNR', type=float, default=0.05, help='siganl noise ration')
parser.add_argument('--name', type=str, default='test', help='train or test')
opt = parser.parse_args()

data = mrcfile.open('./data/150_clean.mrc').data
with mrcfile.new('./data/150' + opt.name + '_SNR' + str(opt.SNR) + '_Gaussian.mrc',
                 overwrite=True) as mrc:
    mrc.set_data(np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32))
    for k in range(data.shape[0]):
        im = data[k]
        im = rescale_intensity(1.0 * im, out_range=(0, 1))
        im1 = random_noise(im, mode='gaussian',
                           var=im.var() / opt.SNR)
        mrc.data[k, :, :] = im1
        print(k)
plt.imshow(im1, cmap='gray')
plt.show()
print(np.var(im) / np.var(im1 - im))
