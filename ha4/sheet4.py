""" ps4_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import solvers
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
import time
from torch.nn import Module, Parameter, ParameterList
from torch.optim import SGD
import scipy.io as sio
from copy import deepcopy


class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None

    def fit(self, X, Y):
        n, d = X.shape

        K = buildKernel(X.T, kernel=self.kernel, kernelparameter=self.kernelparameter)

        one = np.ones(n)
        cee = np.full(n, fill_value=self.C)
        P = ((K * Y).T * Y).T
        q = -one
        G = np.concatenate([-np.eye(n), np.eye(n)], axis=0)
        h = np.concatenate([np.zeros(n), cee], axis=0)
        A = Y.reshape(1, n)  # hint: this has to be a row vector
        b = 0  # hint: this has to be a scalar

        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        solvers.options['show_progress'] = False

        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()
        indexes_sv = np.where(alpha > 1e-5)[0]

        self.indexes_sv = indexes_sv
        self.alpha_sv = alpha[indexes_sv]
        self.X_sv = X[indexes_sv]
        self.Y_sv = Y[indexes_sv]
        self.K_sv = buildKernel(self.X_sv.T, kernel=self.kernel, kernelparameter=self.kernelparameter)

        indexes_bias = np.where(self.alpha_sv < self.C - 1e-5)[0]
        alpha_bias = self.alpha_sv[indexes_bias]
        X_bias = self.X_sv[indexes_bias]
        Y_bias = self.Y_sv[indexes_bias]
        if len(list(indexes_bias)) > 0:
            '''calculate bias from vectors on margin'''

            K_bias = buildKernel(X_bias.T, self.X_sv.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
            vector_sv = self.alpha_sv * self.Y_sv
            func_vals = K_bias @ vector_sv
            biases = func_vals - Y_bias
            self.b = np.mean(biases)
        elif len(list(indexes_sv)) > 0:
            '''calculate bias from support vectors'''
            print('no vectors on margin, using support vectors')
            vector_sv = self.alpha_sv * self.Y_sv
            func_vals = self.K_sv @ vector_sv
            biases = func_vals - self.Y_sv
            self.b = np.mean(biases)
        else:
            print('no support vectors')
            self.b = 0




        '''test biases on margin points if possible'''

        #results_bias = self.predict(X_bias)
        #print('biastest=', Y_bias * results_bias)

    def predict(self, X):
        K = buildKernel(self.X_sv.T, X.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
        vec = self.alpha_sv * self.Y_sv
        prebias = K.T @ vec
        return prebias - self.b


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1. / (1. / 2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)


def plot_boundary_2d(X, Y, model, title=''):

    plt.plot()

    if hasattr(model, 'X_sv'):
        plt.scatter(x=model.X_sv[:, 0], y=model.X_sv[:, 1], c='r', marker='x', alpha=1, s=50, label='support vectors')
    positive_indices = np.where(np.sign(Y) == 1)[0]

    # if neural net, we need to switch -1 and 0 in the labels
    negative_indices = np.where(Y == 0)[0] if 0 in Y else np.where(np.sign(Y) == -1)[0]

    # X in positive and negative class
    X_positive = X[positive_indices]
    X_negative = X[negative_indices]

    if len(X_positive) > 0:
        plt.scatter(x=X_positive[:, 0], y=X_positive[:, 1], label='class1', c='b', marker='o', alpha=.6)
    if len(X_negative) > 0:
        plt.scatter(x=X_negative[:, 0], y=X_negative[:, 1], label='class2', c='g', marker='o', alpha=.6)

    '''draw contour line'''

    grid_density = 1000

    xmax = np.max(X[:, 0])
    xmin = np.min(X[:, 0])
    ymax = np.max(X[:, 1])
    ymin = np.min(X[:, 1])
    xvals = np.linspace(xmin, xmax, grid_density)
    yvals = np.linspace(ymin, ymax, grid_density)
    x, y = np.meshgrid(xvals, yvals)
    points = np.array([x.flatten(), y.flatten()]).T
    predictions = model.predict(points)
    n = grid_density ** 2
    if len(predictions.shape) == 2:
        targets = np.zeros(n)
        for i in range(n):
            if predictions[i][0] > predictions[i][1]:
                targets[i] = 1
            else:
                targets[i] = -1
    else:
        targets = np.sign(predictions)

    plt.contourf(x, y, targets.reshape(x.shape), levels=0, alpha=.3)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.title(f'{title}')
    plt.legend(loc='upper left')
    plt.show()
    print("Done")


def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X ** 2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2 * np.dot(X.T, X)
    else:
        X2 = sum(X ** 2, 0)[:, np.newaxis]
        Y2 = sum(Y ** 2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2 * np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if type(Y) is bool and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K ** kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter ** 2))
    else:
        raise Exception('unspecified kernel')
    return K


class neural_network(Module):
    def __init__(self, layers=[2, 100, 2], scale=.1, p=.1, lr=.1, lam=.1):
        super().__init__()
        self.weights = ParameterList([Parameter(scale * torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([Parameter(scale * torch.randn(n)) for n in layers[1:]])

        self.p = p  # dropout rate
        self.lr = lr  # learning rate
        self.lam = lam  # weight decay coefficient
        self.train = False

    def relu(self, X, W, b):
        # print("Relu Dimensions: ", W.shape, b.shape)
        if self.train:
            Z = torch.matmul(X, W) + b
            # apply relu
            Z[Z < 0] = 0
            # apply dropout bernoulli
            dropout_mask = torch.rand(Z.shape) < self.p
            Z[dropout_mask] = 0
        else:
            Z = torch.matmul(X, W) * (1 - self.p)  # scale down by p, for testing
            Z = Z + b
            Z[Z < 0] = 0
        return Z

    def softmax(self, X, W, b):
        Z = torch.matmul(X, W) + b
        pred = torch.exp(Z) / torch.sum(torch.exp(Z), dim=1, keepdims=True)
        return pred

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            X = self.relu(X, W, b)
        X = self.softmax(X, self.weights[-1], self.biases[-1])
        return X

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        # implement cross entropy loss
        assert ypred.shape == ytrue.shape
        sum = 0
        for i in range(ypred.shape[0]):
            for c in range(ypred.shape[1]):
                sum += ytrue[i, c] * torch.log(ypred[i, c])
        return -sum / ypred.shape[0]

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.floor(.9 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()


class Assignment_4():

    def __init__(self):
        # load .npz data
        data = np.load('data/easy_2d.npz')
        self.X_train, self.y_train = data['X_tr'].T, data['Y_tr']
        self.X_test, self.y_test = data['X_te'].T, data['Y_te']
        self.optimal_kernelparameter = None
        self.optimal_model = None
        self.optimal_C = None

    def find_optimal_parameters(self):
        # find optimal parameters for gaussian kernel
        params = {'kernel': ['gaussian'],
                  'kernelparameter': np.linspace(0.1, 3, 5),
                  'C': np.linspace(0.1, 3, 5)}
        optimal_model = cv(X=self.X_train, y=self.y_train, loss_function=zero_one_loss, method=svm_qp, params=params,
                           nrepetitions=1)
        print("Optimal parameters found [kernel, c]", optimal_model.kernelparameter, optimal_model.C)
        plot_boundary_2d(self.X_test, self.y_test, optimal_model)
        print("DONE")
        self.optimal_model = optimal_model

    def train_overfit_underfit(self):
        models = {
            'underfit_model_skl': svm_sklearn(kernel='gaussian', kernelparameter=100, C=1),
            'underfit_model': svm_qp(kernel='gaussian', kernelparameter=100, C=1),
            'overfit_model_skl': svm_sklearn(kernel='gaussian', kernelparameter=0.1, C=0.001),
            'overfit_model': svm_qp(kernel='gaussian', kernelparameter=0.1, C=0.001)
        }
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)

            plot_boundary_2d(self.X_test, self.y_test, model, title=name)

    def plot_roc(self):
        """ Plot ROC curve for varying bias parameter b of SVM """

        if self.optimal_model is None:
            self.optimal_model = svm_qp(kernel='gaussian', kernelparameter=1, C=1.5)
            self.optimal_model.fit(self.X_train, self.y_train)
        b = np.linspace(-2, 2, 10000)
        fpr, tpr = [], []
        model = self.optimal_model
        total_p = np.sum(self.y_test == 1)
        total_n = np.sum(self.y_test == -1)
        for b_ in b:
            model.b = b_
            y_pred = model.predict(self.X_test)
            y_pred = np.sign(y_pred)
            # compute the number of true positives and false positives
            tp = np.sum((y_pred == 1) & (self.y_test == 1))
            fp = np.sum((y_pred == 1) & (self.y_test == -1))
            tpr += [tp / total_p]
            fpr += [fp / total_n]

        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve for SVM over different bias parameters')
        plt.show()


class Assignment_5():
    def __init__(self, model=svm_qp):
        self.data_dict = dict(np.load('data/iris.npz'))
        self.X = self.data_dict['X'].T
        self.Y = self.data_dict['Y'].T
        self.loss = []

    def lin_test(self, model=svm_qp):
        n, d = self.X.shape
        '''for each class check for loinear separability from other two classes'''
        for clas in range(1, 4):
            self.Y_mod = np.zeros(n)
            self.Y_mod[self.Y != clas] = -1
            self.Y_mod[self.Y == clas] = 1
            plt.figure(figsize=(15,15))
            for index,kernel in enumerate(['linear','polynomial', 'gaussian']):
                for kernelparameter in np.linspace(1,5,5):
                    kernelparameter = int(kernelparameter)
                    losses = []
                    subplot_num = int((index * 5) + kernelparameter)
                    plt.subplot(3, 5, subplot_num)
                    for C in np.linspace(1, 10, 10):
                        self.model = model(kernel=kernel, kernelparameter=kernelparameter, C=C)
                        self.model.fit(self.X, self.Y_mod)
                        predictions = self.model.predict(self.X)
                        loss = zero_one_loss(self.Y_mod, predictions)
                        losses.append(loss)
                    plt.plot(np.linspace(1, 10, 10), losses, c='r')
                    # draw dotted line on 0 loss
                    plt.plot(np.linspace(1, 10, 10), np.zeros(10), c='k', linestyle='--', alpha = .5)
                    plt.xlabel('C')
                    plt.ylabel('loss')
                    plt.ylim(-.2, 1)
                    plt.title(f'{str(kernel)} , {kernelparameter}')
            plt.suptitle(f'Classification Results for Class {clas}',fontsize=30)
            plt.show()


class Assignment_6():
    def __init__(self):
        # load matlab data
        data = sio.loadmat('data/usps.mat')
        self.X = data['data_patterns'].T
        # reshape every vector in self.X to 16x16 img
        self.X_2d = self.X.reshape(self.X.shape[0], 16, 16)
        self.Y = data['data_labels'].T
        self.Y_zero_one = deepcopy(self.Y)
        self.Y_zero_one[self.Y == -1] = 0
        # convert labels from one-hot to single int
        self.Y_int = np.argmax(self.Y, axis=1)
        self.p = 0.1
        self.lr = 0.02
        self.lam = 0.01

    def plot_25_images(self, labels, labels_true=None):
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.X_2d[i], cmap='gray')
            plt.title(labels[i], color='red' if labels_true is not None and labels_true[i] == labels[i] else 'black')
            plt.axis('off')
        plt.show()

    def svm_cross_validation(self):

        # create log file
        with open('svm_cross_validation.txt', 'w'):
            pass

        # iterate over all label classes 0-9
        for label in range(10):
            print('\nTESTING DIGIT:', label)
            # set label to 1 if it is the current label, else -1 as integer
            y = (self.Y_int == label).astype(int)
            y[y == 0] = -1

            # cross validation
            params = {'kernel': ['linear', 'polynomial', 'gaussian'],
                      'kernelparameter': np.linspace(1, 3, 3)}
            optimal_model = cv(X=self.X, y=y, method=svm_qp, loss_function=zero_one_loss, params=params, nrepetitions=1,
                               nfolds=5)
            predictions = optimal_model.predict(self.X)
            predictions = np.sign(predictions)
            self.plot_25_images(predictions, labels_true=self.Y_int)

            # write and print logs
            print("Optimal parameters found [kernel, kernel_param]", optimal_model.kernel,
                  optimal_model.kernelparameter, )
            print("Test Error:", optimal_model.cvloss)
            with open('svm_cross_validation.txt', 'a') as f:
                f.write(f"{label} {optimal_model.kernel} {optimal_model.kernelparameter} {optimal_model.cvloss}\n")

    def nn_cross_validation(self, nsteps=250, n_params=4):

        # create log file
        with open('nn_cross_validation.txt', 'w'):
            pass

        params = {'layers': [[256, 100, 10]],
                  'p': np.linspace(0.1, 0.3, n_params),
                  'lam': np.logspace(-3, -0.5, n_params),
                  'lr': np.logspace(-3, -0.5, n_params)}

        optimal_model = cv(X=self.X, y=self.Y_zero_one, method=neural_network, loss_function=nn_zero_one_loss,
                           params=params, nrepetitions=1, nfolds=5, nsteps=nsteps)

        print("Optimal parameters found [layers, p, lam, lr]", len(optimal_model.weights), optimal_model.p,
              optimal_model.lam, optimal_model.lr)
        print("Test Error:", optimal_model.cvloss)
        self.optimal_model = optimal_model
        with open('nn_cross_validation.txt', 'a') as f:
            f.write(
                f"{optimal_model.weights} {optimal_model.p} {optimal_model.lam} {optimal_model.lr} {optimal_model.cvloss}\n")

    def plot_support_vectors(self):
        """ For every class (digit), plot the support vectors of nn and svm """
        pass

    def plot_nn_weight_vectors(self):
        """ Plot 100 weight vectors of the first layer of the neural net (grayscale) """

        for nsteps in [5, 50, 250]:
            model = neural_network(layers=[256, 100, 10], p=self.p, lr=self.lr, lam=self.lam)
            model.fit(X=self.X, y=self.Y_zero_one, nsteps=nsteps, plot=True)
            weights = model.weights[0].detach().numpy()
            weights = weights.reshape(-1, 10, 10)
            plt.figure(figsize=(10, 10))
            dim = 10
            for i in range(dim * dim):
                plt.subplot(dim, dim, i + 1)
                # imshow from -1 to 1
                plt.imshow(weights[i], cmap='gray', vmin=-1, vmax=1)
                plt.axis('off')
            plt.suptitle(f"First Layer weights fitted with nsteps={nsteps}")
            plt.show()

        for lam in [0.01, 0.1, 1]:
            model = neural_network(layers=[256, 100, 10], p=self.p, lr=self.lr, lam=lam)
            model.fit(X=self.X, y=self.Y_zero_one, nsteps=250)
            weights = model.weights[0].detach().numpy()
            weights = weights.reshape(-1, 10, 10)
            plt.figure(figsize=(10, 10))
            dim = 10
            for i in range(dim * dim):
                plt.subplot(dim, dim, i + 1)
                plt.imshow(weights[i], cmap='gray', vmin=-1, vmax=1)
                plt.axis('off')
            plt.suptitle(f"First Layer weights fitted with lam={lam}")
            plt.show()


def zero_one_loss(y_true, y_pred):
    ''' Loss function that calculates percentage of correctly predicted signs'''
    output = np.sum(y_true != np.sign(y_pred))
    total = len(y_true)
    return output / total


def nn_zero_one_loss(y_true, y_pred, target=None):
    ''' Loss function that calculates percentage of correctly predicted signs'''

    # get the index of the largest value in y_pred
    y_pred_index = np.argmax(y_pred, axis=1)
    # get the index of the largest value in y_true
    y_true_index = np.argmax(y_true, axis=1)
    if target is None:
        output = np.sum(y_true_index != y_pred_index)
        output = output / len(y_true)
    else:
        output = np.sum((y_true_index != y_pred_index) & (y_true_index == target))
        # get the number of samples that are the target class
        total = np.sum((y_true_index == target))
        output = output / total
    return output


def nn_loss(y_pred, y_true):
    # implement cross entropy loss
    assert y_pred.shape == y_true.shape
    sum = 0
    for i in range(y_pred.shape[0]):
        for c in range(y_pred.shape[1]):
            sum += y_true[i, c] * np.log(y_pred[i, c])
    return -sum / y_pred.shape[0]


def cv(X, y, method, params, loss_function, nfolds=10, nrepetitions=5, bias=None, nsteps=None):
    from sklearn.model_selection import KFold
    import itertools as it
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
        start_time = time.time()  # start timer
        counter += 1
        param_combination = {}
        for i, name in enumerate(params.keys()):
            param_name = name
            param_combination[param_name] = param_combi_unnamed[i]

        param_combi_losses = []
        for repetion in range(nrepetitions):
            # divide x in nfolds random partitions of the same size
            kf = KFold(n_splits=nfolds)
            for train_ix, test_ix in kf.split(X):
                # get the values and labels for training and testing
                X_train, y_train = X[train_ix], y[train_ix]
                X_test, y_test = X[test_ix], y[test_ix]

                # train the model using the training data and get predictions about the test data
                model = method(**param_combination)
                if nsteps is not None:
                    model.fit(X_train, y_train, nsteps=nsteps)
                else:
                    model.fit(X_train, y_train)
                if bias is None:
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test, bias=bias)

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
        all_losses_for_single_param_combi = np.array(all_losses_for_single_param_combi)
        model.cvloss = np.mean(all_losses_for_single_param_combi, axis=0)
    model.fit(X, y)

    return model


if __name__ == '__main__':
    if False:
        runner = Assignment_6()
        runner.svm_cross_validation()
    runner = Assignment_4()
    runner.find_optimal_parameters()
    runner.train_overfit_underfit()
    runner.plot_roc()

    import winsound

    winsound.Beep(500, 200)

