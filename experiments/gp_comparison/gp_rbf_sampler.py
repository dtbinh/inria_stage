import sys
from sklearn import grid_search

__author__ = 'omohamme'
from data_parser_mathew import data_parser

import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.metrics import r2_score

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcess

from multilayer_perceptron  import MultilayerPerceptronRegressor


def rbf_sampler_exp():
    fourier_score_list = []
    for i in range(100):
        data_train, targets_train = data_parser(num_in_samples=150)
        data_test, targets_test = data_parser(num_in_samples=1000)

        feature_map_fourier = RBFSampler(gamma=.2, random_state=10)
        # feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
        fourier_approx_gp = pipeline.Pipeline([("feature_map", feature_map_fourier), ("GP", GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget = 3.00e-12))])

        fourier_approx_gp.set_params(feature_map__n_components=2)
        fourier_approx_gp.fit(data_train, targets_train)
        fourier_score = fourier_approx_gp.score(data_test, targets_test)

        # print "Start of fitting RBF"
        # feature_map_fourier.fit(data_train, targets_train)
        # x = feature_map_fourier.transform(data_train)
        #
        # print feature_map_fourier
        # print "data_train = ", len(data_train)
        # print "x = ", len(x)

        print fourier_score
        fourier_score_list.append(fourier_score)

        plt.figure(figsize=(15.0, 11.0))
        plt.boxplot(fourier_score_list)
        plt.ylabel("R2 Score", fontsize=20)
        plt.xlabel("Number of sampling points", fontsize=20)
        plt.show()

def pca_analysis():
    for D in range(1,13):
        data_train, targets_train = data_parser(num_in_samples=20000)
        pca = PCA(n_components=D)
        pca.fit(data_train)
        # print(pca.explained_variance_ratio_)
        print(pca.score_samples(data_train))

def svm_gp():
    final_results = {}
    # for dimension in [2, 3, 4, 5, 6, 12]:
    for dimension in [6]:
        print "Current Dimension = ", dimension
        X, y = data_parser(num_in_samples=15000, file_name="all_outputs_"+str(dimension)+"d.txt")
        gp_data, gp_target = data_parser(num_in_samples=10, file_name="all_outputs_"+str(dimension)+"d.txt")
        data_test, targets_test = data_parser(num_in_samples=1000, file_name="all_outputs_"+str(dimension)+"d.txt")
        # Fit regression model
        svm_non_linear = svm.NuSVR()
        # svm_non_linear = svm.SVR(C=10, kernel='poly', epsilon=0.02, degree=5, tol=1e-5)
        y_non_linear = svm_non_linear.fit(X, y)

        gp_data_base = {}
        for i in range(len(gp_target)):
            gp_data_base[gp_target[i]] = gp_data[i]

        toolbar_width = 100
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        for i in range(100):
            gp = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100,
                                 nugget=3.00e-14)
            # Pick a random sample
            gp.fit(gp_data_base.values(), gp_data_base.keys())
            random_data, random_target = data_parser(num_in_samples=1, file_name="all_outputs_"+str(dimension)+"d.txt")
            gp_data_base[random_target[0]] = random_data[0]

            # Get the score for this point from
            # model_0 = svm_non_linear.predict(random_data) + gp.predict(random_data)
            # model_1 = svm_non_linear.predict(random_data) * gp.predict(random_data)
            # y_pred, MSE = gp.predict(data_test, eval_MSE=True)
            # print MSE
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("\n")

        # model_0 = svm_non_linear.predict(data_test) + gp.predict(data_test)
        final_results[dimension] = {"model_0": svm_non_linear.predict(data_test),
                                    "model_1": svm_non_linear.predict(data_test) + gp.predict(data_test, eval_MSE=True)[1],
                                    "model_2": -1 * svm_non_linear.predict(data_test) * gp.predict(data_test)}
        model_0 = svm_non_linear.predict(data_test)
        model_1 = svm_non_linear.predict(data_test) + gp.predict(data_test, eval_MSE=True)[1]
        model_2 = -1 * svm_non_linear.predict(data_test) * gp.predict(data_test)
        model_3 = (svm_non_linear.predict(data_test) + gp.predict(data_test)) * 0.5
        print "model_0 score = ", repr(r2_score(targets_test, model_0))
        print "model_1 score = ", repr(r2_score(targets_test, model_1))
        print "model_2 score = ", repr(r2_score(targets_test, model_2))
        print "model_3 score = ", repr(r2_score(targets_test, model_3))
        print "---------------------------"

def nn_gp():
    for dimension in [6]:
        print "Current Dimension = ", dimension
        X, y = data_parser(num_in_samples=18000, file_name="all_outputs_"+str(dimension)+"d.txt")
        X= np.array(X).astype(np.float)
        y= np.array(y).astype(np.float)
        data_test, targets_test = data_parser(num_in_samples=1000, file_name="all_outputs_"+str(dimension)+"d.txt")

        parameters = dict(algorithm = ['l-bfgs', 'sgd'],
                      hidden_layer_sizes = [60, 100],
                      activation = ['logistic', 'tanh', 'relu'],
                      alpha = [0.00001],
                      max_iter = [2000],
                      learning_rate = ['invscaling'])

        # clf = MultilayerPerceptronRegressor(algorithm='l-bfgs', hidden_layer_sizes=60, activation='tanh', alpha=0.00001, max_iter=2000, learning_rate='invscaling') # l-bfgs sgd
        mlp = MultilayerPerceptronRegressor()

        clf = grid_search.GridSearchCV(mlp, parameters)
        clf.fit(X, y)

        data_test = np.array(data_test).astype(np.float)
        targets_test = np.array(targets_test).astype(np.float)

        # mlp.fit(X, y)
        # y_compute = mlp.predict(X)
        # mlp_score = mlp.score(data_test, targets_test)
        grid_score = clf.score(data_test, targets_test)
        print grid_score
        print(grid_search.best_estimator_)
# rbf_sampler_exp()
# pca_analysis()
# svm_gp()
nn_gp()