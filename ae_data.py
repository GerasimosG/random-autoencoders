import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import math
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import scipy.io
import pickle


def mnist_import(numTrain=50000, is_visualize=True, range_normalize=None, is_one_hot=True):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=is_one_hot)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Reshape
    train_data = np.reshape(train_data, [train_data.shape[0], 28, 28, 1])
    eval_data = np.reshape(eval_data, [eval_data.shape[0], 28, 28, 1])
    
    # Visualize
    if is_visualize:        
        import matplotlib.pyplot as plt
        ind = np.random.randint(train_data.shape[0])
        plt.imshow(train_data[ind,:,:,0], cmap='gray')
        plt.axis('off')
        plt.show()

    # Make X_train, X_val, X_test
    numVal = train_data.shape[0] - numTrain
    ind = np.random.permutation(train_data.shape[0])
    X_train = np.reshape(train_data[ind[0:numTrain]], [numTrain, 28**2])
    y_train = train_labels[ind[0:numTrain]]
    X_val = np.reshape(train_data[ind[numTrain:]], [numVal, 28**2])
    y_val = train_labels[ind[numTrain:]]

    numTest = eval_data.shape[0]
    X_test = np.reshape(eval_data, [numTest, 28**2])
    y_test = np.array(eval_labels, copy=True)

    dataDim = 28**2
    
    # range normalize
    X_train = do_range_normalize(X_train, range_normalize)
    X_val = do_range_normalize(X_val, range_normalize)
    X_test = do_range_normalize(X_test, range_normalize)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dataDim


def do_range_normalize(X, range_normalize=None):    
    if range_normalize is not None:
        a, b = np.min(range_normalize), np.max(range_normalize)
        c, d = np.min(X), np.max(X)
        m = (b-a)*1.0/(d-c)        
        p = b - m*d
        X = X*m + p
    return X