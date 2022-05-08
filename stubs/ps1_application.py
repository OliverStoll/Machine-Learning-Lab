""" sheet1_implementation.py

PUT YOUR NAMES HERE:
<FIRST_NAME><LAST_NAME>
<FIRST_NAME><LAST_NAME>


Write the functions
- usps
- outliers
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
import numpy as np
import scipy.io as sio
import ps1_implementation as imp
import matplotlib.pyplot as plt


def usps():
    ''' performs the usps analysis for assignment 5'''
    def add_noise(img, noise_level):
        # add noise to the image, with the given noise level, img is a numpy vector of length 256
        return img + np.random.normal(0, noise_level, img.shape)

    noises = [0, .2, .5, 'outlier']
    denoising_components = 25

    # load the data from mat file
    data = sio.loadmat('../data/usps.mat')

    # read metadata to determine shapes
    # print(len(data['data_labels']))
    # print(len(data['data_patterns']))
    # print(len(data['data_patterns'][0]))

    # extract the data labels and the data itself from the data dictionary
    labels = data['data_labels']
    data_raw = data['data_patterns']  # 2007 images of size 256 (16x16)

    data_raw = data_raw.T

    for noise in noises:
        if noise == 'outlier':  # outliers noise
            data_noised = [img for img in data_raw]
            # add noise to 5 images in data raw
            for i in range(5):
                data_noised[i] = add_noise(data_noised[i], 1)
        else:
            # add noise to the data
            data_noised = [add_noise(img, noise) for img in data_raw]

        # perform PCA on the data
        data_pca = imp.PCA(data_noised)

        # extract all principal values
        principle_values = data_pca.D
        # extract the first 25 principal values
        principle_values_25 = principle_values[:25]
        # plot the principle values as a barplot
        # create figure
        plt.figure(figsize=(9, 15))
        # plt.tight_layout()
        plt.suptitle('Visualisation for noise level: ' + str(noise))
        plt.subplot(3, 1, 1)
        plt.bar(range(len(principle_values)), principle_values)
        plt.title('All principle values')
        plt.xlabel('Principle value')
        plt.ylabel('Value')
        #plt.show()
        # plot the principle values as a barplot
        # plt.figure()
        plt.subplot(3, 1, 2)
        plt.bar(range(len(principle_values_25)), principle_values_25)
        plt.title('25 most significant principle values')
        plt.xlabel('Principle value')
        plt.ylabel('Value')
        #plt.show()

        # extract the first 5 principle components
        plt.subplot(3, 1, 3)
        principle_components = data_pca.U[:, :5]
        # plot the first 5 principle components
        plt.imshow(principle_components, cmap='gray', aspect='auto')
        plt.title('Five most relevant principle components')
        plt.xlabel('Principle component')
        plt.ylabel('Vector')

        # plot the first 10 images in data_noised
        plt.figure(figsize=(10, 7.5))
        for i in range(10):
            plt.subplot(3, 10, i + 1)
            plt.imshow(data_raw[i].reshape(16, 16), cmap='gray', aspect='auto')
            # hide ticks
            plt.xticks([])
            plt.yticks([])
        for i in range(10):
            plt.subplot(3, 10, i + 11)
            plt.imshow(data_noised[i].reshape(16, 16), cmap='gray', aspect='auto')
            # hide ticks
            plt.xticks([])
            plt.yticks([])
        for i in range(10):
            plt.subplot(3, 10, i + 21)
            # denoise the images in data noised using PCA.denoise
            data_denoised = data_pca.denoise(data_noised, denoising_components)
            plt.imshow(data_denoised[i].reshape(16, 16), cmap='gray', aspect='auto')
            # hide ticks
            plt.xticks([])
            plt.yticks([])

        plt.suptitle("First 10 images for noise level " + str(noise) + "\nshowing originals, noisy images and denoised images")
        plt.show()



def outliers_calc():
    ''' outlier analysis for assignment 6'''
    # np.savez_compressed('outliers.npz', var1=var1, var2=var2, ...)


def outliers_disp():
    ''' display the boxplots'''
    # results = np.load('outliers.npz')


def lle_visualize(dataset='flatroll'):
    ''' visualization of LLE for assignment 7'''


def lle_noise():
    ''' LLE under noise for assignment 8'''    


if __name__ == '__main__':
    usps()