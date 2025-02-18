import math
from xml.dom import minidom
from xml.etree import ElementTree as ET

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .c45_utils import decision, grow_tree

class C45(BaseEstimator, ClassifierMixin):
    """A C4.5 tree classifier.

    Parameters
    ----------
    attrNames : list, optional (default=None)
        The list of feature names used in printing tree during. If left default,
        attributes will be named attr0, attr1... etc
    See also
    --------
    DecisionTreeClassifier
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning
    .. [2] https://en.wikipedia.org/wiki/C4.5_algorithm
    .. [3] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.
    .. [4] J. R. Quinlain, "C4.5: Programs for Machine Learning",
           Morgan Kaufmann Publishers, 1993
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from c45 import C45
    >>> iris = load_iris()
    >>> clf = C45(attrNames=iris.feature_names)
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """
    def __init__(self, attrNames_=None):
        #if attrNames_ is not None:
        #    attrNames_ = [''.join(i for i in x if i.isalnum()).replace(' ', '_') for x in attrNames_]
        self.attrNames_ = attrNames_

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.resultType = type(y[0])
        assert(self.attrNames_ is not None)
        #if self.attrNames_ is None:
        #    self.attrNames_ = [f'attr{x}' for x in range(len(self.X_[0]))]

        assert(len(self.attrNames_) == len(self.X_[0]))

        data = [[] for i in range(len(self.attrNames_))]
        categories = []

        for i in range(len(self.X_)):
            categories.append(str(self.y_[i]))
            for j in range(len(self.attrNames_)):
                data[j].append(self.X_[i][j])
        root = ET.Element('DecisionTree')
        grow_tree(data,categories,root,self.attrNames_)
        self.tree_ = ET.tostring(root, encoding="unicode")

        self.classes_ = self.predict(X)
        return self

    def predict(self, X):
        #check_is_fitted(self, ['tree_', 'resultType', 'attrNames'])
        X = check_array(X)
        dom = minidom.parseString(self.tree_)
        root = dom.childNodes[0]
        predictions = []
        for i in range(len(X)):
            answerlist = decision(root,X[i],self.attrNames_,1)
            answerlist = sorted(answerlist.items(), key=lambda x:x[1], reverse = True )
            answer = answerlist[0][0]
            predictions.append((self.resultType)(answer))
        return predictions

    def printTree(self):
        check_is_fitted(self, ['tree_'])
        dom = minidom.parseString(self.tree_)
        print(dom.toprettyxml(newl="\r\n"))
