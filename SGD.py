import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.preprocessing import StandardScaler


def dataset1():
    """
    Source: https://www.kaggle.com/c/santander-customer-satisfaction/data
    Dataset about cusomer satisfaction of Santander customers
    """
    train = pd.read_csv("train.csv")
    test  = pd.read_csv("test.csv")
    train = pd.concat([train[train.TARGET == 1][:1000], train[train.TARGET == 0][:1000]], axis = 0)
    X_train = train.iloc[:, :-1]
    sample_basic = ["var3", "var15"]
    X = X_train[sample_basic].values
    y_train = train.TARGET.values
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    X = X.astype(np.float64)
    y_train = y_train.astype(np.float64)
    return X, y_train

def dataset2():

    iris = dt.load_iris()
    X = iris.data[:, :2]
    y_true = (iris.target != 0) * 1

def loss(y_hat, y_true):
    return (-y_true * np.log(y_hat) - (1 - y_true) * np.log(1 - y_hat)).mean()

    # Taking derivatives of the cost function wrt every feature
    grad = np.dot(X.transpose(), y_hat - y_true)
    return grad

def hypothesis(X, theta):

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    f_x = np.dot(X, theta)
    y_hat = sigmoid(f_x)

    return y_hat

def shuffle_set(X, y):

  r = np.random.permutation(len(y))
  return X[r], y[r]

def batch_gradient_descent(X, y_true, lr = 0.01, iter = 10000, batch_ratio = 0.2):

    loss_hist = []
    theta_hist = []
    theta = np.zeros(X.shape[1])
    batch = int(len(y_true)*batch_ratio)
    for epoch in range(iter):
        X, y = shuffle_set(X,y_true)
        X = X[:batch,:]
        y_true = y[:batch]
        y_hat = hypothesis(X, theta)

        gradient = np.dot(X.T, y_hat - y_true)
        # print(gradient)
        theta = theta - lr * gradient
        # print(theta)
        # current loss
        cost = loss(y_hat, y_true)
        loss_hist.append(cost)
        # calculating gradient & changing coefficients
        theta_hist.append(theta)
    return theta, loss_hist, theta_hist

X, y_true = dataset1()

X = StandardScaler().fit_transform(X)

theta, loss_hist, theta_hist = batch_gradient_descent(X, y_true)
# pd.Series(loss_hist).plot()
learn_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
costs = pd.DataFrame([batch_gradient_descent(X, y_true, lr = i)[1] for i in learn_rates])
costs.index = learn_rates

for i in range(5):
    costs.T.iloc[:,i].plot()
    plt.show()
