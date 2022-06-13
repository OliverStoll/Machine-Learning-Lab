""" ps3_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from sklearn.model_selection import KFold, RepeatedKFold
from mpl_toolkits.mplot3d import Axes3D


def zero_one_loss(y_true, y_pred):
    ''' Loss function that calculates percentage of correctly predicted signs'''
        return np.average(y_true == np.sign(y_pred))


def mean_absolute_error(y_true, y_pred):
    """ Loss function that return the average of the absolutes of the differences between label and prediction """

    assert len(y_true) == len(y_pred)
    errors = []
    for index in range(len(y_true)):
        abs_error = np.abs(y_true - y_pred)
        errors.append(abs_error)
    return np.mean(errors)


def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5):
    """ Creates a class 'method' for cross validation """

    X = np.array(X)
    y = np.array(y)
    avg_error = 0
    param_options = [param_list for param_list in params.values()]

    optimal_loss = 999999999
    optimal_params_combi = None
    num_param_combinations = len(list(it.product(*param_options)))
    counter = 0
    all_losses_for_single_param_combi = []

    # iterate over all parameter combinations and find the optimal one (by cross-validation)
    for param_combi_unnamed in it.product(*param_options):
        counter += 1
        param_combi_losses = []
        param_combination = {}
        for i, name in enumerate(params.keys()):
            param_name = name
            param_combination[param_name] = param_combi_unnamed[i]

        for repetion in range(nrepetitions):
            # divide x in 10 random partitions of the same size
            kf = KFold(n_splits=nfolds)
            for train_ix, test_ix in kf.split(X):
                # get the values and labels for training and testing
                X_train, y_train = X[train_ix], y[train_ix]
                X_test, y_test = X[test_ix], y[test_ix]

                # train the model using the training data and get predictions about the test data
                model = method()
                model.fit(X=X_train, y=y_train, **param_combination)
                y_pred = model.predict(X=X_test)

                # evaluate the predictions against the labels for the test data
                loss = loss_function(y_true=y_test, y_pred=y_pred)
                param_combi_losses.append(loss)
                all_losses_for_single_param_combi.append(loss)

        avg_loss = np.mean(param_combi_losses)
        print(f"{param_combination}: {avg_loss} ... completed {counter}/{num_param_combinations}")  # print log

        if avg_loss < optimal_loss:
            optimal_params_combi = param_combination
            optimal_loss = avg_loss

    # finally train model on optimal param combi
    model = method()
    model.cvloss = optimal_loss
    if num_param_combinations == 1:
        model.cvloss = np.mean(all_losses_for_single_param_combi)
    model.fit(X=X, y=y, **optimal_params_combi)

    return model

class krr:
    def __init__(self, kernel, kernelparameter, regularization):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization
    def fit(self, Xtrain, ytrain):
        n,d = Xtrain.shape
        K = kernelmatrix(X = Xtrain, kernelparameter = self.kernelparameter)
        self.K = K
        if self.regularization == 0:
            self.regularization = one_out_crossval(data = Xtrain, labels = ytrain, K = self.K)
        inverse = scipy.linalg.inv(K + self.regularization * np.eye(n))
        alpha = inverse @ ytrain
        self.w = Xtrain.T @ alpha


    def predict(self, Xtest):
        return(Xtest @ self.w)

def kernelmatrix(X, kernel, kernelparameter):
    n,d = X.shape
    K = np.zeros((n,n))
    for i,x in enumerate(X):
        for j,y in enumerate(X):
            K[i][j] = kernelcalc(x = x,y = y,kernel = kernel,kernelparameter = kernelparameter)
    return(K)


def kernelcalc(x,y,kernel, kernelparameter):
    if kernel == 'linear':
        return(np.dot(x,y))
    elif kernel == 'polynomial':
        return((np.dot(x,y) + 1 ) ** kernelparameter)
    elif kernel == 'gaussian':
        a = (scipy.linalg.norm(x-y, ord = 2)**2 )/(2* (kernelparameter**2))
        return(np.exp(-a))
    else:
        raise Exception('kernel not implemented')

def one_out_crossval(data, labels, K):
    eigvals = scipy.linalg.eigvals(data.T @ data)
    mean = np.mean(eigvals)
    candidates1 = mean + np.exp(np.linspace(-10, 10, 21))
    candidates2 = mean - np.exp(np.linspace(-10, 10, 21))
    candidates = np.concatenate([candidates1, np.zeros(1), candidates2[::-1]])
    errors = np.zeros(len(candidates))
    L,U = eig_decomp(K)
    for index, candidate in enumerate(candidates):
        error = one_out_err(labels = labels, C=candidate, L = L, U = U)
        errors[index] = error
    optindex = np.argmin(errors)
    return candidates[optindex]
def one_out_err(labels, C, L, U):
    diag = 1/(L + C)
    diagmat = np.diag(diag)
    S = U @ L @ diagmat @ U.T
    Sy = S @ labels
    fraction = (labels - Sy)/(1-diag)
    return(np.average(fraction))

'''compute eigen decomposition of symmetric matrix A, i.e. A = U @ L @ U.T'''
def eig_decomp(A):
    eigvals,eigvecs = scipy.linalg.eigh(A)
    L, U = np.diag(eigvals), eigvecs
    return(L,U)

def load_data():
    data = {}
    for testset in ['banana', 'diabetis', 'flare-solar', 'image', 'ringnorm']:
        data[testset] = {}
        for type in ['xtrain', 'xtest', 'ytrain', 'ytest']:
            pathsuffix = testset + '-' + type + '.dat'
            path = 'C:/Users/funto/PycharmProjects/MLLAB/data/U04_' + pathsuffix
            data[testset][type] = np.loadtxt(path)
    return(data)


if __name__ == "__main__":
    X = list(range(20))
    y = [0] * 20
    params = {'a': [0, 1, 2], 'b': [3, 4]}
    cv(X=X, y=y, method=None, params=params)
