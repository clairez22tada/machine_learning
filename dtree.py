import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col]<self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)



class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction




def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    
    return 1-(len(x[x==0])/len(x))**2- (len(x[x==1])/len(x))**2

    
def find_best_split(X, y, loss, min_samples_leaf):
    k = 11
    best = (-1, -1, loss(y)) #(feature, split, loss)
    for col in range(X.shape[1]):
        candidates = np.unique(X[:,col])
        candidates = np.random.choice(candidates,min(k,len(candidates)),replace=False)#randomly choose k
        for split in candidates:
            yl=y[X[:,col]<split]
            yr=y[X[:,col]>=split]
            if len(yl)<min_samples_leaf or len(yr)<min_samples_leaf:
                continue
            l = (len(yl)*loss(yl)+len(yr)*loss(yr))/len(y)
            if l==0:
                return (col, split, l)
            if l<best[2]:
                best = (col, split, l)
    return best     
    
    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)
        

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
            
        if len(X)<=self.min_samples_leaf or len(np.unique(X))==1:
                return self.create_leaf(y)
        col,split = find_best_split(X, y, self.loss,self.min_samples_leaf)[:2]
        if col==-1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:,col]<split,:], y[X[:,col]<split])
        rchild = self.fit_(X[X[:,col]>=split,:], y[X[:,col]>=split])
        return DecisionNode(col, split, lchild, rchild)

        

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.apply_along_axis(self.root.predict, axis=1, arr=X_test)
    

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return 1 - np.sum((self.predict(X_test) - y_test)**2)/np.sum((y_test -np.mean(y_test))**2)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))




class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return np.sum(y_test == self.predict(X_test))/len(y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, stats.mode(y,keepdims=True))
