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
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from torch.nn import Module, Parameter, ParameterList
from torch.optim import SGD


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

        K = buildKernel(X.T)
        print(np.linalg.matrix_rank(K))
        one = np.ones(n)
        cee = np.full(n,fill_value=self.C)
        P = ((K * Y).T * Y).T
        q = -one
        #q = - one.reshape(n,1)
        G = np.concatenate([-np.eye(n),np.eye(n)], axis = 0)
        #G = np.eye(n)
        h = np.concatenate([np.zeros(n),cee], axis = 0)
        #h = cee.reshape(n,1)
        #A = np.zeros(shape=(n,n)) + Y
        A = Y.reshape(1,n) # hint: this has to be a row vector
        b = 0  # hint: this has to be a scalar

        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()
        indexes = np.where(alpha != 0)
        self.alpha_sv = alpha[indexes]
        self.X_sv = X[indexes]
        self.Y_sv = Y[indexes]
        '''calculate bias'''
        m = len(self.alpha_sv)
        self.K_sv = buildKernel(self.X_sv.T)
        vector = self.alpha_sv * self.Y_sv
        biases = self.Y_sv - self.K_sv @ vector
        self.b = np.average(biases)

    def predict(self, X):
        K = buildKernel(self.X_sv.T,X.T)
        vec = self.alpha_sv * self.Y_sv
        return(K.T @ vec - self.b)


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


def plot_boundary_2d(X, y, model):
    fig,ax = plt.subplots()
    pos_inds = np.where(np.sign(y) == 1)
    neg_inds = np.where(np.sign(y) == -1)
    X_c1 = X[pos_inds]
    X_c2 = X[neg_inds]
    ax.scatter(x = X_c1[:,0],y = X_c1[:,1], label = 'class1', c = 'b')
    ax.scatter(x = X_c2[:,0],y = X_c2[:,1], label = 'class2',c = 'r')



    '''draw contour line'''

    grid_density = 100

    xmax = np.max(X[:, 0])
    xmin = np.min(X[:, 1])
    ymax = np.max(X[:, 0])
    ymin = np.min(X[:, 1])
    xvals = np.linspace(xmin, xmax, grid_density)
    yvals = np.linspace(ymin, ymax, grid_density)
    x, y = np.meshgrid(xvals, yvals)
    points = np.array([x.flatten(), y.flatten()]).T
    targets = model.predict(points)

    ax.contourf(x,y, targets.reshape(x.shape), levels = 0, alpha = .3)
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




if __name__ == '__main__':
    X = np.array([[0,1],[0,2],[5,1],[5,4]])
    y = np.array([1,1,-1,-1])
    model = svm_qp()
    model.fit(X = X, Y = y)
    model.predict(np.array([[0,1],[1,2]]))
    #X_grid,y_grid = grid_eval(X = X, y = y,model = model, grid_density=100)

    plot_boundary_2d(X = X,y = y, model= model)