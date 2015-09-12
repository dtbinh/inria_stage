__author__ = 'omohamme'
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import linear_model
import os
import random
import multiprocessing
import time
import copy
import matplotlib.pyplot as plt
import math
import json
from scipy import stats

start = 0.0
stop = 1
dimension = 12 #default value

def zdt1_problem(x):
    assert (type(x) == list)

    input_x = x
    f1 = input_x[0]
    g = 1.0
    for item in x[1:]:
        g += 9.0 / (len(x) - 1) * item
    h = 1.0 - math.sqrt(f1 / g)
    f2 = g * h

    return f1, f2

def zdt2_problem(x):
    assert (type(x) == list)

    input_x = x
    f1 = input_x[0]
    g = 1.0
    for item in x[1:]:
        g += 9.0 / (len(x) - 1) * item
    h = 1.0 - ((f1 / g)**2)
    f2 = g * h

    return f1, f2

def uniform_random_sampling():
    global start, stop, dimension
    results = []
    for i in range(dimension):
        random_number = start + (random.random() * (stop - start))
        results.append(str(random_number))
    return results

def generate_random_test_samples(number_of_test_samples, sampling_function):
    test_points = []
    for i in range(number_of_test_samples):
        # generated_parameters = uniform_random_sampling()
        generated_parameters = sampling_function()
        generated_parameters = map(float, generated_parameters)
        test_points.append(generated_parameters)
    return test_points

class problem_gp():
    def __init__(self, num_in_samples=2, sampling_function=uniform_random_sampling, problem_function = zdt1_problem):
        # self.gp = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget = 3.00e-14)
        # self.gp = linear_model.LinearRegression()
        self.gp = svm.SVR()
        self.results_hash = {}
        self.problem = problem_function
        self.model_problem_r2_score = 0
        self.observations = []
        self.samples = []
        self.num_in_samples = num_in_samples
        self.sampling_function = sampling_function
        self.model_problem_results = {}
        self.generateX_AND_getY_for_training()

    def generateX_AND_getY_for_training(self):
        for i in range(self.num_in_samples):
            # generated_parameters = uniform_random_sampling()
            generated_parameters = self.sampling_function()
            results = self.problem(map(float, generated_parameters)) [1] # I will only learn one objective
            # results = self.problem(map(float, generated_parameters)) [0] # I will only learn one objective
            self.results_hash[",".join(generated_parameters)] = results

        for item in self.results_hash:
            self.samples.append(map(float, item.split(",")))
            self.observations.append(self.results_hash[item])

    def fit_gp(self):
        assert (len(self.samples) > 0)
        assert (len(self.observations) > 0)
        self.gp.fit(self.samples, self.observations)

    def test_gp(self, test_pairs):
        self.model_problem_r2_score = self.gp.score(X=test_pairs[0], y=test_pairs[1])
        return self.model_problem_r2_score

class problem_testing():
    def __init__(self, num_test_samples=50, sampling_function=uniform_random_sampling, problem_function=zdt1_problem):
        self.test_pairs = []
        self.num_test_samples = num_test_samples
        self.problem = problem_function
        self.sampling_function = sampling_function
        self.test_points = []
        self.real_problem_results = {}
        self.generate_test_samples()

    def generate_test_samples(self):
        X = []
        y = []
        for i in range(self.num_test_samples):
            generated_parameters = self.sampling_function()
            results = self.problem(map(float, generated_parameters))[1] # I will only learn one objective
            self.test_points.append(generated_parameters)
            X.append(map(float, generated_parameters))
            self.real_problem_results[",".join(generated_parameters)] = results
            y.append(results)

        self.test_pairs = [X, y]

    def return_test_points(self):
        return self.test_points

    def return_test_pairs(self):
        return self.test_pairs

    def compute_r2_score(self, model_results):
        r2_score = 0
        for test_input in model_results:
            r2_score += (model_results[test_input] - self.real_problem_results[test_input])**2
        return r2_score / self.num_test_samples

def experiment(parameters_list):
    problem_object = parameters_list[0]
    test_object = parameters_list[1]
    problem_object.fit_gp()
    model_results = problem_object.test_gp(test_pairs=test_object.return_test_pairs())
    print model_results
    del problem_object
    del test_object
    return model_results

t0 = time.time()
load_from_file = False
# learning_algorithms = {"SVM_Regression":}
for dimension in range(8, 13):
    print "Current Dimension = ", dimension
    if not load_from_file:
        test_methods = problem_testing(num_test_samples=1000)
        prediction_results = []
        samples_numbers_range = [10, 25, 50, 100, 200, 500]
        # samples_numbers_range = [10, 25]
        final_prediction_values = {}
        gp_kernels = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']

        for i in samples_numbers_range: # Fit the models with different points. See how the accuracy changes
            print "current sample = ", i
            inputs_list = []
            prediction_results = []
            for j in range(100):
                problem = problem_gp(num_in_samples=i)
                inputs_list.append((copy.deepcopy(problem), copy.deepcopy(test_methods)))
                # inputs_list = (copy.deepcopy(problem), copy.deepcopy(test_methods))
                # prediction_results.append(experiment(inputs_list))
                del problem

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
            prediction_results = pool.map(experiment, inputs_list)

            final_prediction_values[i] = prediction_results
            with open("result_"+str(dimension)+"D.json", 'w') as fp:
                json.dump(final_prediction_values, fp)
        del test_methods
    else:
        with open("result_"+str(dimension)+"D.json", 'r') as fp:
            final_prediction_values = json.loads(fp.readline())

    current_path = os.getcwd()
    plt.figure(figsize=(15.0, 11.0))
    plt.boxplot(final_prediction_values.values())
    plt.xticks(range(1, len(final_prediction_values.keys()) + 1), final_prediction_values.keys(), rotation=45, fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylabel("R2 score", fontsize=20)
    plt.xlabel("Number of sampling points", fontsize=20)
    plt.savefig("results_benchmark_"+str(dimension)+"D_SVR_obj_1.png")
    os.chdir(current_path)

    # stat_file = open("Stat_tests_benchmark_"+str(dimension)+"D_linearmodel_obj_0.txt", "w")
    # seen_pairs = []
    # for number_of_samples in final_prediction_values:
    #     for number_of_samples2 in final_prediction_values:
    #         if (number_of_samples != number_of_samples2) and ((number_of_samples, number_of_samples2) not in seen_pairs):
    #             seen_pairs.append((number_of_samples, number_of_samples2))
    #             seen_pairs.append((number_of_samples2, number_of_samples))
    #             statistical_significance = stats.wilcoxon(final_prediction_values[number_of_samples], final_prediction_values[number_of_samples2])
    #             print >> stat_file, number_of_samples, " VS ", number_of_samples2, " -->", statistical_significance
    #             print >> stat_file, number_of_samples, " median = ", np.median(final_prediction_values[number_of_samples])
    #             print >> stat_file, number_of_samples2, " median = ", np.median(final_prediction_values[number_of_samples2])
    #             print >> stat_file, "----------------------------------------------------------"

t1 = time.time() - t0
print "Total regression time = ", t1