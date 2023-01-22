""" sheet1_implementation.py

PUT YOUR NAMES HERE:
Oliver Stoll
Anton Hopmann


Write the functions
- usps
- outliers
- lle
Write your implementations in the given functions src!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class PCA():
    """
    PCA class for performing PCA on a given dataset.
    """
    def __init__(self, Xtrain):
        # center the data in Xtrain
        self.C = np.mean(Xtrain, axis=0)
        self.Xtrain = Xtrain - self.C

        # compute the covariance matrix
        self.cov = np.cov(self.Xtrain.T)

        # compute the eigenvalues and eigenvectors
        import scipy.linalg as la
        self.eigvals, self.eigvecs = la.eig(self.cov)

        # get only the real eigenvalues and eigenvectors
        self.eigvals = self.eigvals.real
        self.eigvecs = self.eigvecs.real

        # sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(self.eigvals)[::-1]
        self.D = self.eigvals[idx]
        self.U = self.eigvecs[:, idx]

    def project(self, Xtest, m):
        # calculate the projection of Xtest on the m-dimensional space
        # using the principle components self.U
        return np.dot(Xtest - self.C, self.U[:, :m])

    def denoise(self, Xtest, m):
        # denoise the data in Xtest by projecting it on the m-dimensional space
        # and then back to the original space
        projection = self.project(Xtest, m)
        return np.dot(projection, self.U[:, :m].T) + self.C


def gammaidx(X, k):
    # calculate the gamma index for the data in X
    # using k nearest neighbors
    gamma = np.empty(len(X))

    for i in range(len(X)):
        # calculate distances between X[i] and neighbors
        distances = np.linalg.norm(X - X[i], axis=-1)
        # sort the distances
        idx = np.argsort(distances)
        # take the indexes from the k nearest neighbors, average the distances
        gamma[i] = np.mean(distances[idx[1:k+1]])
    return gamma


def auc(y_true, y_pred, plot=False):
    # debug set plot to true
    # plot=True
    # import and plot the ROC curve using sklearn metrics
    from sklearn.metrics import roc_curve
    # calculate the false positive rates and true positive rates for the curve to be plotted
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)

    # plot the curve if desired
    if plot:
        plt.plot(false_positive_rate, true_positive_rate)
        plt.show()

    # calculate the AUC from the false positive rates and true positive rates
    auc = np.trapz(true_positive_rate, false_positive_rate)

    # return the area under the curve
    return auc


def lle(X, m, n_rule, param, tol=1e-2):
    # print 'Step 1: Finding the nearest neighbours by rule ' + n_rule
    # ...
    # print 'Step 2: local reconstruction weights'
    # ...
    # print 'Step 3: compute embedding'
    # ...
    pass


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
        data_pca = PCA(Xtrain=data_noised)

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