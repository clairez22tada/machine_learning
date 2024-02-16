import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        

    def fit(self, X, y):
        """a
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.oob_indexes_per_tree = []
        self.nunique=len(np.unique(y))

        for i in range(self.n_estimators):
            # Bootstrap Sampling
            bootstrap_indexes = np.random.choice(len(X), size=len(X), replace=True)
            oob_indexes = np.array(list(set(range(len(X))) - set(bootstrap_indexes)))
            self.oob_indexes_per_tree.append(oob_indexes)

            self.trees[i].fit(X[bootstrap_indexes], y[bootstrap_indexes])

        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y, self.oob_indexes_per_tree)
        

            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = [RegressionTree621(min_samples_leaf, max_features) for i in range(n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        pred = []
        for x_test in X_test:
            y_pred = 0
            total_weight=0
            for tree in self.trees:            
                y_pred += tree.predict(np.array([x_test]))[0]*tree.root.leaf(x_test).n
                total_weight+= tree.root.leaf(x_test).n
            if total_weight != 0:
                pred.append(y_pred / total_weight)
            else:
                pred.append(0)
            
        return np.array(pred)
        
    def compute_oob_score(self, X, y, oob_indexes_per_tree):
        pred = []
        for i in range(X.shape[0]):
            x=X[i]
            y_pred = 0
            total_weight=0
            for j in range(len(self.trees)):
                if i in self.oob_indexes_per_tree[j]:
                    tree=self.trees[j]
                    y_pred += tree.predict(np.array([x]))[0]*tree.root.leaf(x).n
                    total_weight+= tree.root.leaf(x).n
            if total_weight != 0:
                pred.append(y_pred / total_weight)
            else:
                pred.append(0)
        return r2_score(y, np.array(pred))
            
    
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
        
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = [ClassifierTree621(min_samples_leaf, max_features) for i in range(n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        pred = []
        for x_test in X_test:
            class_counts=np.zeros(self.nunique)
            for tree in self.trees:
                class_counts=np.add(class_counts, tree.root.leaf(x_test).prediction)
            pred.append(np.argmax(class_counts))
            
        return np.array(pred)
    
    def compute_oob_score(self, X, y, oob_indexes_per_tree):
        pred = []
        for i in range(X.shape[0]):
            x=X[i]
            y_pred = 0
            class_counts=np.zeros(self.nunique)
            for j in range(len(self.trees)):
                tree=self.trees[j]
                if i in self.oob_indexes_per_tree[j]:
                    class_counts=np.add(class_counts, tree.root.leaf(x).prediction)
            pred.append(np.argmax(class_counts))
            
        return accuracy_score(y,  np.array(pred))


  


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
