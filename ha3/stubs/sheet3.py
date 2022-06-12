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
    ''' your header here!
    '''


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


class krr():
    ''' your header here!
    '''

    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' your header here!
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        return self

    def predict(self, X):
        ''' your header here!
        '''
        return self


if __name__ == "__main__":
    X = list(range(20))
    y = [0] * 20
    params = {'a': [0, 1, 2], 'b': [3, 4]}
    cv(X=X, y=y, method=None, params=params)
