import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv/train.csv')
data = np.array(data)

m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
x_dev = data_dev[1:n]
y_dev = data_dev[0]
x_dev = x_dev / 255

data_train = data[1000:m].T
x_train = data_train[1:n]
y_train = data_train[0]
x_train = x_train / 255

def weightsAndBiases():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    res = np.exp(z) / sum(np.exp(z))
    return res

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def deriv_ReLU(z):
    return z > 0

def one_hot_encoder(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def backward_prop(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_y = one_hot_encoder(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * sum(dz2)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * sum(dz1)
    return dw1,db1,dw2,db2

def parameter_update(w1, b1, w2, b2, dw1, db1, dw2, db2, a):
    w2 = w2 - a * dw2
    #b2 = b2 - a * db2
    w1 = w1 - a * dw1
    #b1 = b1 - a * db1
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2,0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, a, iterations):
    w1, b1, w2, b2 = weightsAndBiases()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = parameter_update(w1, b1, w2, b2, dw1, db1, dw2, db2,a)
        if i % 100 == 0:
            print('Iteration: ',i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(x_train, y_train, 0.1, 500)

_, _, _, a2 = forward_prop(w1, b1, w2, b2, x_dev)
print(get_accuracy(get_predictions(a2),y_dev))