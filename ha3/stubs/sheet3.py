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
import pickle
import time
import scipy.io as sio
import pylab as pl
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold
from mpl_toolkits.mplot3d import Axes3D
import scipy


def zero_one_loss(y_true, y_pred):
    ''' Loss function that calculates percentage of correctly predicted signs'''
    return np.average(y_true == np.sign(y_pred))


def mean_squared_error(y_true, y_pred):
    # compute the mean squared error
    return np.mean((y_true - y_pred) ** 2)


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
    param_options = [param_list for param_list in params.values()]
    num_param_combinations = len(list(it.product(*param_options)))

    counter = 0
    optimal_loss = 999999999
    optimal_params_combi = None
    all_losses_for_single_param_combi = []

    # iterate over all parameter combinations and find the optimal one (by cross-validation)
    for param_combi_unnamed in it.product(*param_options):
        start_time = time.time()        # start timer
        counter += 1
        param_combi_losses = []
        param_combination = {}
        for i, name in enumerate(params.keys()):
            param_name = name
            param_combination[param_name] = param_combi_unnamed[i]

        for repetion in range(nrepetitions):
            # divide x in nfolds random partitions of the same size
            kf = KFold(n_splits=nfolds)
            for train_ix, test_ix in kf.split(X):
                # get the values and labels for training and testing
                X_train, y_train = X[train_ix], y[train_ix]
                X_test, y_test = X[test_ix], y[test_ix]

                # train the model using the training data and get predictions about the test data
                model = method(**param_combination)
                model.fit(X=X_train, y=y_train)
                y_pred = model.predict(X_test)

                # evaluate the predictions against the labels for the test data
                loss = loss_function(y_true=y_test, y_pred=y_pred)
                param_combi_losses.append(loss)
                all_losses_for_single_param_combi.append(loss)

        avg_loss = np.mean(param_combi_losses)
        time_diff = time.time() - start_time
        eta = time_diff * (num_param_combinations - counter)
        print(f"{param_combination}: {avg_loss:.6f} ... completed {counter}/{num_param_combinations}  ETA: {eta:.1f}s")

        if avg_loss < optimal_loss:
            optimal_params_combi = param_combination
            optimal_loss = avg_loss

    # finally train model on optimal param combi
    model = method(**optimal_params_combi)
    model.cvloss = optimal_loss
    if num_param_combinations == 1:
        model.cvloss = np.mean(all_losses_for_single_param_combi)
    model.fit(X=X, y=y)

    return model


class krr:
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y):
        n, d = X.shape
        K = kernelmatrix(X=X, kernelparameter=self.kernelparameter, kernel=self.kernel)
        self.K = K
        if self.regularization == 0:
            self.regularization = one_out_crossval(data=X, labels=y, K=self.K)
        inverse = scipy.linalg.inv(K + self.regularization * np.eye(n))
        alpha = inverse @ y
        self.w = X.T @ alpha

    def predict(self, Xtest):
        return (Xtest @ self.w)


def kernelmatrix(X, kernel, kernelparameter):
    n, d = X.shape
    K = np.zeros((n, n))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            K[i][j] = kernelcalc(x=x, y=y, kernel=kernel, kernelparameter=kernelparameter)
    return (K)


def kernelcalc(x, y, kernel, kernelparameter):
    if kernel == 'linear':
        return (np.dot(x, y))
    elif kernel == 'polynomial':
        return ((np.dot(x, y) + 1) ** kernelparameter)
    elif kernel == 'gaussian':
        a = (scipy.linalg.norm(x - y, ord=2) ** 2) / (2 * (kernelparameter ** 2))
        return (np.exp(-a))
    else:
        raise Exception('kernel not implemented')


def one_out_crossval(data, labels, K):
    eigvals = scipy.linalg.eigvals(data.T @ data)
    mean = np.mean(eigvals)
    candidates1 = mean + np.exp(np.linspace(-10, 10, 20))
    candidates2 = mean - np.exp(np.linspace(-10, 10, 20))
    candidates = np.concatenate([candidates1, candidates2[::-1]])
    errors = np.zeros(len(candidates))
    L, U = eig_decomp(K)
    for index, candidate in enumerate(candidates):
        error = one_out_err(labels=labels, C=candidate, L=L, U=U)
        errors[index] = error
    optindex = np.argmin(errors)
    return candidates[optindex]


def one_out_err(labels, C, L, U):
    diag = 1 / (L + C)
    diagmat = np.diag(diag)
    S = U @ L @ diagmat @ U.T
    Sy = S @ labels
    fraction = (labels - Sy) / (1 - diag)
    return (np.average(fraction))


def eig_decomp(A):
    '''compute eigen decomposition of symmetric matrix A, i.e. A = U @ L @ U.T'''
    eigvals, eigvecs = scipy.linalg.eigh(A)
    L, U = np.diag(eigvals), eigvecs
    return (L, U)


##############################################################################


class krr_application:
    """ Class that is used to perform Assignment 4 """
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_dict = self._load_data()
        self.results_dict = {}
        self.num_cv_repetitions = 1  # TODO: change this to 5

        # TODO: What are the parameters we need to test??
        self.kernel_list = ['linear', 'polynomial', 'gaussian']
        self.kernelparameter_list = [1, 2, 3]
        self.regularization_list = [0, 0.1, 1, 10]

        # save the parameter options as dictionary
        self.parameters_dict = {'kernel': self.kernel_list,
                                'kernelparameter': self.kernelparameter_list,
                                'regularization': self.regularization_list}

    def _load_data(self):
        """ Load the data from the data path, as a dictonary of dictionaries"""
        data = {}
        for testset in ['banana', 'diabetis', 'flare-solar', 'image', 'ringnorm']:
            data[testset] = {}
            for type in ['xtrain', 'xtest', 'ytrain', 'ytest']:
                full_path = f'{self.data_path}/U04_{testset}-{type}.dat'
                data[testset][type] = sio.loadmat(full_path)  # transpose to have the correct shape
        return (data)

    def search_for_optimal_parameters(self, loss_function=zero_one_loss):
        """
        Function that actually searches for the parameters that yield the best performance

        The loss function parameter can be set to mean_squared_error (function without brackets as arg) for 4d
        """
        for test_set in self.data_dict:
            self.results_dict[test_set] = {}

            # get the trainings data & labels
            x_train, y_train = self.data_dict[test_set]['xtrain'], self.data_dict[test_set]['ytrain']

            # perform cross validation over the general parameter options
            print("Performing cross validation for test set:", test_set)
            optimal_model = cv(X=x_train, y=y_train, method=krr, params=self.parameters_dict,
                               loss_function=loss_function, nrepetitions=self.num_cv_repetitions)

            # store the results in the results dictionary
            self.results_dict[test_set]['cvloss'] = optimal_model.cvloss
            self.results_dict[test_set]['kernel'] = optimal_model.kernel
            self.results_dict[test_set]['kernelparameter'] = optimal_model.kernelparameter
            self.results_dict[test_set]['regularization'] = optimal_model.regularization

            # get the predictions from the test set
            self.results_dict[test_set]['y_pred'] = optimal_model.predict(self.data_dict[test_set]['xtest'])

        # finally, save results dict to file
        with open(f'results.p', 'wb') as f:
            pickle.dump(self.results_dict, f)

    def plot_roc_curve(self, biases):
        """
        Function that takes a set of biases and calculates the TPR and FPR.
        After this, the ROC curve is plotted by calling cv with the optimal parameters from results.

        This can somehow be done by using roc_fun as a loss function and calculater of tpr and fpr at the same time.
        """


def roc_fun(y_true, y_pred):
    """ """




class assignment_3:
    """ Class that is used to perform Assignment 3 """
    def __init__(self, data_path="data/qm7.mat"):
        self.data = sio.loadmat(data_path)
        self.data_len = len(self.data['X'])
        self.train_split = 5000

        # get eigenvalues as basic data
        self.eigenvalues_sums = None
        self.eigenvalues = None
        self._get_eigenvalues_from_data()

        # fixed data
        self.labels = self.data['T'].reshape(-1)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self._fix_dataset()

        # cross validation
        self.regulization_params = np.logspace(-7, 0, 5)
        self.width_quantiles = [0.01, 0.99]  # [0.01, 0.1, 0.5, 0.9, 0.99]
        self.width_quantiles_values = None

        # results
        self.optimal_width_param = None
        self.optimal_regulization_param = None

    def run(self):
        """ Run the assignment """
        self.plot_distances()

    def _get_eigenvalues_from_data(self):
        """ From the data (and there the list of matrices), get the eigenvectors """

        eig_vals, eig_vals_sums = [], []
        for matrix in np.real(self.data['X']):
            eig_val = np.real(scipy.linalg.eigvals(matrix))
            eig_val_sum = np.sum(eig_val)
            eig_vals.append(eig_val)
            eig_vals_sums.append(eig_val_sum)

        # get the sorted indices of the eigenvalues
        sorted_indices = np.argsort(eig_vals_sums).tolist()[::-1]

        # sort the eigenvalues by the sorted indices
        self.eigenvalues = [eig_vals[i] for i in sorted_indices]
        self.eigenvalues_sums = [eig_vals_sums[i] for i in sorted_indices]

    def _fix_dataset(self):
        """ Shuffle the data, split it into training and test set 5000/2165 and fix it """

        # get the shuffled indices
        shuffled_indices = np.random.permutation(self.data_len)
        train_indices = shuffled_indices[:self.train_split]
        test_indices = shuffled_indices[self.train_split:]

        # convert eigenvalues and labels to numpy arrays
        self.eigenvalues = np.array(self.eigenvalues)
        self.labels = np.array(self.labels)

        # fix the data
        self.X_train, self.y_train = self.eigenvalues[train_indices], self.labels[train_indices]
        self.X_test, self.y_test = self.eigenvalues[test_indices], self.labels[test_indices]

    def plot_distances(self):
        """ Plot the distances between all pairs of eigenvectors, and their respective labels """

        sums_x = []
        sums_y = []
        for i in range(self.data_len):
            for j in range(self.data_len):
                sum_x = np.abs(self.eigenvalues_sums[i] - self.eigenvalues_sums[j])
                sum_y = np.abs(self.labels[i] - self.labels[j])
                sums_x.append(sum_x)
                sums_y.append(sum_y)

        # plot the distances
        plt.figure(figsize=(12, 12))
        plt.scatter(sums_x, sums_y, s=1)
        plt.show()

    def apply_cv(self):
        """
        Apply cross validation on 2500 random training samples and fix them.
        Also report the optimal parameters.

        For:
          - the width parameter of the gaussian kernel
          - the regularization parameter (log between 10^-7 and 10^0)
        """

        # get the shuffled indices from the training set
        shuffled_indices = np.random.permutation(self.train_split)[:2500]

        X = self.X_train[shuffled_indices]
        y = self.y_train[shuffled_indices]
        X_sums = np.sum(X, axis=1)

        # get quantiles of the eigenvalues_sums
        self.width_quantiles_values = np.quantile(X_sums, self.width_quantiles, axis=0)
        parameters = {}

        # get the optimal width parameter by cross validation
        result_model = cv(X=X, y=y, )



        # self.optimal_width_param =
        # self.optimal_regulization_param =
        pass

    def plot_mae(self):
        """ Plot the mean absolute error for the test set as a function of the number of training samples (n) """

        pass

    def plot_scatter(self):
        """
        Plot the scatter plot of points (y_i, ^y_i) with train and test data in two different colors

        This is done for three different models that: underfit, fit well and overfit.
        """

        pass



if __name__ == '__main__':

    # krr_application(data_path='data').search_for_optimal_parameters()
    assignment_3().apply_cv()
