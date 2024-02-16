import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:, :-1]
    Y = np.where(data[:, -1] == 0, -1, 1)  # Convert 0 to -1
    ### END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    d = np.ones(N) / N # initial weights

    ### BEGIN SOLUTION
    for i in range(num_iter):
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        tree.fit(X, y, sample_weight=d)
        trees.append(tree)
        y_hat = tree.predict(X)
        err = np.sum(d[y != y_hat])/np.sum(d)
        if err != 0:
            alpha = np.log((1-err)/err)
        else:
            alpha = 1
        d = d * np.exp(alpha * (y != y_hat))
        trees_weights.append(alpha)
    ### END SOLUTION
    return trees, trees_weights

def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION
    for tree, alpha in zip(trees, trees_weights):
        y += alpha * tree.predict(X)
    y = np.sign(y)
    ### END SOLUTION
    return y
