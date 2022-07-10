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
from torch.nn import Module, Parameter, ParameterList
from torch.optim import SGD
from sheet3 import cv


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
        n,d = X.shape

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
        '''calculate bias'''
        vector_sv = self.alpha_sv * self.Y_sv
        func_vals = self.K_sv.T @ vector_sv
        biases = func_vals - self.Y_sv
        self.b = np.mean(biases)
        '''test biases on margin points'''
        indexes_bias = np.where(self.alpha_sv < self.C - 1e-5)[0]
        alpha_bias = self.alpha_sv[indexes_bias]

        X_bias = X[indexes_bias]
        Y_bias = Y[indexes_bias]
        results_bias = self.predict(X_bias)
        print('biastest=', Y_bias * results_bias)
        '''
        K_bias = buildKernel(self.X_sv.T, X_bias.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
        vector_sv = self.alpha_sv * self.Y_sv
        func_vals = K_bias.T @ vector_sv
        biases = func_vals - Y_bias
        if len(indexes_bias) > 0:
            self.b = np.mean(biases)
        else:
            self.b = 0
            '''
        '''test y_i f(x_i) = 1'''
        #print(np.dot(self.alpha_sv,self.Y_sv))
    def predict(self, X):
        K = buildKernel(self.X_sv.T,X.T, kernel=self.kernel, kernelparameter=self.kernelparameter)
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


def plot_boundary_2d(X, Y, model,title = 'whatever'):

    fig,ax = plt.subplots()
    if hasattr(model, 'X_sv'):
        x_sv = model.X_sv[:, 0]
        y_sv = model.X_sv[:, 1]
        ax.scatter(x = x_sv, y = y_sv, c='r', marker='x',alpha=1, label = 'support vectors')
    pos_inds = np.where(np.sign(Y) == 1)[0]
    '''neural net special case'''
    if 0 in Y:
        neg_inds = np.where(Y == 0)[0]
    else:
        neg_inds = np.where(np.sign(Y) == -1)[0]

    X_c1 = X[pos_inds]
    X_c2 = X[neg_inds]

    # remove all support vectors from X_c1 and X_c2
    if hasattr(model, 'X_sv'):
        X_c1 = X_c1[np.where(np.isin(X_c1, model.X_sv, invert=True))[0]]
        X_c2 = X_c2[np.where(np.isin(X_c2, model.X_sv, invert=True))[0]]

    ax.scatter(x = X_c1[:,0],y = X_c1[:,1], label = 'class1', c = 'b', marker = 'o',alpha = 1)
    ax.scatter(x = X_c2[:,0],y = X_c2[:,1], label = 'class2',c = 'g', marker = 'o',alpha = 1)



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
    n = grid_density**2
    if len(predictions.shape) ==2:
        targets = np.zeros(n)
        maxima = np.max(predictions,axis=1)
        for i in range(n):
            if predictions[i][0] > predictions[i][1]:
                targets[i] = 1
            else:
                targets[i] = -1
    else:
        targets = np.sign(predictions)

    ax.contourf(x,y, targets.reshape(x.shape), levels = 0, alpha = .3)
    ax.set_xlim([xmin ,xmax])
    ax.set_ylim([ymin,ymax])
    plt.title(f'{title}')
    ax.legend(loc = 'upper left')
    plt.show()


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
    def __init__(self, layers=[2, 100, 2], scale=.1, p=None, lr=None, lam=None):
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
        optimal_model = cv(X=self.X_train, y=self.y_train, method=svm_qp, params=params, nrepetitions=1)
        print("Optimal parameters found [kernel, c]", optimal_model.kernelparameter, optimal_model.C)
        plot_boundary_2d(self.X_test, self.y_test, optimal_model)
        print("DONE")
        self.optimal_model = optimal_model

    def train_overfit_underfit(self):
        overfit_model = svm_qp(kernel='gaussian', kernelparameter=100, C=1)
        underfit_model = svm_qp(kernel='gaussian', kernelparameter=0.1, C=0.001)
        for model in [overfit_model, underfit_model]:
            model.fit(self.X_train, self.y_train)
            plot_boundary_2d(self.X_test, self.y_test, model, title = f'{model}')

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



class assignment_5():
    def __init__(self, model = svm_qp):
        self.data_dict = dict(np.load('data/iris.npz'))
        self.X = self.data_dict['X'].T
        self.Y = self.data_dict['Y'].T
        self.model = model(kernel='polynomial', kernelparameter=3, C=5)
        self.loss = []
    def lin_test(self):
        n,d = self.X.shape
        '''for each class check for loinear separability from other two classes'''
        for clas in range(1,4):
            self.Y_mod = np.zeros(n)
            self.Y_mod[self.Y != clas] = -1
            self.Y_mod[self.Y == clas] = 1
            self.model.fit(self.X, self.Y_mod)
            predictions = self.model.predict(self.X)
            self.loss.append(np.average(np.sign(predictions) != self.Y_mod))
        print(self.loss)



if __name__ == '__main__':
    runner = Assignment_4()
    # runner.find_optimal_parameters()
    # runner.train_overfit_underfit()
    #runner.plot_roc()
    runner = assignment_5(svm_qp)
    runner.lin_test()