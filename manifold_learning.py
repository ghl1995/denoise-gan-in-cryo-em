#############data pre-processing: get the angles of each image and rotation and mask, translate into png file
# import cv2
import os
import numpy as np
import cv2
import mrcfile
from time import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.segmentation import random_walker


# root_dir ='./'
# trajlist_file="%s/noisy_images.list"%(root_dir)

def plot_tsne(epoch, data1, mname):
    if not os.path.isdir('./results/' + mname):
        os.mkdir('./results/' + mname)
    data = []
    for i in range(1500):
        image = data1[i]
        # image = binary.findGoldMask(image)
        data.append(image.flatten())
    class_list = np.load('./data/test_namelist.npy')
    data_binary = np.array(data)
    class_list = np.array(class_list) + 1

    ##Binary

    # starting manifold embedding
    n_neighbors = 25
    n_components = 2
    SNR = 0.05
    colors_jet = ['k', 'r', 'g', 'b', 'y']

    method = 't-SNE'
    t0 = time()
    Y = manifold.TSNE(n_components=n_components).fit_transform(data_binary)
    print(Y.shape)
    np.savetxt('./results/'+ mname + "/epoch_%d_denoise_SNR_%f_images_%s_%dNN_%d_component_eigs" % (epoch, SNR, method, n_neighbors, n_components), Y)
    t1 = time()
    print(" t-SNE: %.2g sec" % (t1 - t0))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca()
    for i in range(1, 6):  # because the list id is from 1,five conformations
        index = np.where(class_list == i)[0]
        plt.scatter(Y[index, 0], Y[index, 1], label="state %d" % i, color=colors_jet[i - 1], s=30)
    plt.title(" t-SNE (%.2g sec)" % (t1 - t0))
    plt.savefig(
        "./results/"+ mname + "/epoch_%d_denoise_SNR_%f_images_%s_%dNN_%d_component.png" % (epoch, SNR, method, n_neighbors, n_components))
    plt.close()

    method = 'ISOMAP'
    t0 = time()
    Y = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(data_binary)
    print(np.shape(Y))
    np.savetxt('./results/'+ mname + '/epoch_%d_denoise_SNR_%f_images_%s_%dNN_%d_component_eigs' % (epoch, SNR, method, n_neighbors, n_components), Y)
    t1 = time()
    print("Isomap: %.2g sec" % (t1 - t0))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca()
    # plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
    for i in range(1, 6):  # because the list id is from 1,five conformations
        index = np.where(class_list == i)[0]
        plt.scatter(Y[index, 0], Y[index, 1], label="state %d" % i, color=colors_jet[i - 1], s=30)
    plt.title("Isomap (%.2g sec)" % (t1 - t0))
    plt.savefig(
        "./results/"+ mname + "/epoch_%d_denoise_SNR_%f_images_%s_%dNN_%d_component.png" % (
        epoch, SNR, method, n_neighbors, n_components))
    plt.close()

'''method = 'MDS'
t0 = time()
Y = manifold.MDS(n_components=n_components).fit_transform(data_binary)
np.savetxt('SNR_%f_images_%s_%dNN_%d_component_eigs' % (SNR, method, n_neighbors, n_components), Y)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
fig = plt.figure(figsize=(15, 15))
ax = fig.gca()
for i in range(1, 6):  # because the list id is from 1,five conformations
    index = np.where(class_list == i)[0]
    plt.scatter(Y[index, 0], Y[index, 1], label="state %d" % i, color=colors_jet[i - 1], s=30)
plt.title("MDS (%.2g sec)" % (t1 - t0))
plt.savefig("SNR_%f_images_%s_%dNN_%d_component.png" % (SNR, method, n_neighbors, n_components))
plt.close()

method = 'SpectralEmbedding'
t0 = time()
Y = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors,
                               affinity='nearest_neighbors').fit_transform(data_binary)
np.savetxt('SNR_%f_images_%s_%dNN_%d_component_eigs' % (SNR, method, n_neighbors, n_components), Y)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
fig = plt.figure(figsize=(15, 15))
ax = fig.gca()
for i in range(1, 6):  # because the list id is from 1,five conformations
    index = np.where(class_list == i)[0]
    plt.scatter(Y[index, 0], Y[index, 1], label="state %d" % i, color=colors_jet[i - 1], s=30)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
plt.savefig("SNR_%f_images_%s_%dNN_%d_component.png" % (SNR, method, n_neighbors, n_components))
plt.close()

method = 'PCA'


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.savefig('SNR_%f_noisy_eigenfaces.png')


t0 = time()
pca1 = PCA(n_components=2)
Y = pca1.fit_transform(data_binary)
eigenfaces = pca1.components_.reshape((2, width, height))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
fig = plt.figure(figsize=(15, 15))
ax = fig.gca()
# plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
t1 = time()
np.savetxt('SNR_%f_images_%s_%dNN_%d_component_eigs' % (SNR, method, n_neighbors, n_components), Y)
print("pca: %.2g sec" % (t1 - t0))
for i in range(1, 6):  # because the list id is from 1,five conformations
    index = np.where(class_list == i)[0]
    plt.scatter(Y[index, 0], Y[index, 1], label="state %d" % i, color=colors_jet[i - 1], s=30)
plt.title("PCA (%.2g sec)" % (t1 - t0))
plt.savefig("SNR_%f_images_%s_%dNN_%d_component.png" % (SNR, method, n_neighbors, n_components))'''
