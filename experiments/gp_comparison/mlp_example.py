from sklearn.datasets import load_digits
from multilayer_perceptron  import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn import grid_search

# contrive the "exclusive or" problem
X = np.array([[0,0], [1,0], [0,1], [1,1]])
y = np.array([0, 1, 1, 0])

# MLP training performance
mlp = MultilayerPerceptronRegressor(n_hidden=60, max_iter=2000, alpha=0.001)
params = {"n_hidden":range(10,100, 10), "max_iter":range(100,10001,100), "alpha":[0.001,0.01,0.1,1,2,5,10]}
clf = grid_search.GridSearchCV(mlp, params)

clf.fit(X, y)

print "Training Score = ", clf.score(X, y)
print "Predicted labels = ", clf.predict(X)
print "True labels = ", y