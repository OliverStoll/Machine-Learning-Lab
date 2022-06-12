""" sheet1_implementation.py

PUT YOUR NAMES HERE:
Oliver Stoll
Anton Hopmann


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2021
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


class PCA():
    def __init__(self, Xtrain):
        # center the data in Xtrain
        self.C = np.mean(Xtrain, axis=0)
        self.Xtrain = Xtrain - self.C

        # compute the covariance matrix
        self.cov = np.cov(self.Xtrain.T)

        # compute the eigenvalues and eigenvectors
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
