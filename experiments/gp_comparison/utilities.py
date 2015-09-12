__author__ = 'omohamme'

import gc
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import r2_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcess
from multilayer_perceptron import MultilayerPerceptronRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
import os
import random
import string
import multiprocessing
import time
import copy
import json
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys
import random
from PyGMO.problem import base
from PyGMO import *
from multiprocessing import Process, Queue, Value, Array
import uuid # In order to generate a unique ID
import math
import itertools

def clear_screen():
    os.system("reset")

def bf_sampling(dimensions=2, min_value=0, max_value=math.pi, step=0.04):
    """
    This function make a brute force sampling for the whole solution space. This is important in order to
    produce graphs similar to the one in "crossing the reality" paper (it is not useful for learning in general).
    For more details about this brilliant python implementation, refer to
    https://docs.python.org/2/library/itertools.html#itertools.product
    and
    https://docs.python.org/2/library/itertools.html#itertools.repeat
    :param dimensions: Dimensions of the problem (I assume all dimensions have the same range and probability distribution).
    :param min_value: Min range value
    :param max_value: Max range value
    :param step: Step to sample each dimension
    :return: A list of samples.
    """
    samples_list = []
    one_dim_points = np.arange(min_value, max_value, step)
    for sample in itertools.product(one_dim_points, repeat=dimensions):
        samples_list.append(sample)

    print len(samples_list)
    return samples_list

def lhs_sampling_core(num_samples=3, dimensions=4, min_value=0, max_value=math.pi):
    """
    This is a corrected implementation for LHS sampling based on the book 'Design and analysis of computer experiments',
    page 128. I assume here that the all dimensions have similar continuous range.
    :param num_samples: This is the number of samples required.
    :param dimensions: This is the problem dimensionality.
    :param min_value: The start value of the range for any dimension.
    :param max_value: The end value of the range for any dimension.
    :return: A list of point lists, sampled according to the LHS technique.
    """
    # Generate a list of lists. Each of these sub-lists represent one column from the nXd matrix (one dimension).
    # Each sub-list is a random permutation of {1,2,...,num_samples}
    nd_matrix = []
    for i in range(dimensions):
        random_list = list(np.random.permutation(num_samples) + 1) # Generate a random list of values {1,2,...,num_samples}
        nd_matrix.append(random_list)

    # Now construct the nXd matrix according to the book.
    nd_matrix = np.matrix(nd_matrix).transpose()
    # Then, get you samples according to equations in page 128. These samples are from range [0, 1]
    samples_set = []
    for sample_index in range(num_samples):
        lhs_gen_point = []
        for dim_index in range(dimensions):
            # random_U = random.uniform(min_value, max_value)
            random_U = random.random()
            lhs_gen_point.append( (nd_matrix[sample_index, dim_index] - 1 + random_U) / num_samples )
        samples_set.append(lhs_gen_point)

    # Now scale the matrix to the range we want [min_value, max_value]
    samples_set = np.matrix(samples_set)
    samples_set = min_value + (samples_set * (max_value - min_value))
    samples_set = samples_set.tolist()

    return samples_set

def robot_sampling_multiple_points(sampling_points=[], controller_dim=6, robot_condition="good", objective_list=[1]):
    """
    :param sampling_points:
    :param controller_dim:
    :param robot_condition:
    :return:
    """
    assert len(sampling_points) > 0
    # First, establish the hash data structure for each points
    hash_data_structure = []
    for sample in sampling_points:
        hash_data_structure.append(dict(controller_dim=controller_dim,
                              sampling_points=sample,
                              robot_condition=robot_condition))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
    robot_results = pool.map(robot_sampling_one_point, hash_data_structure)
    if len(objective_list) == 1:
        """
        This toggling in the objectives values is due to a problem I made in mop_dev_v2. However, it should work fine.
        """
        if objective_list[0] == 1:
            robot_results = [result[0] for result in robot_results]
            return sampling_points, [robot_results]
        else:
            robot_results = [result[1] for result in robot_results]
            return sampling_points, [robot_results]
    else:
        return sampling_points, [result[0] for result in robot_results], [result[1] for result in robot_results]

def robot_sampling_one_point(parameter_info):
    """
    This function takes one point to be tested on the physical robot, and return a list of tuples. Each
    position in the tuple corresponds to the one of the objective functions.
    :return:
    """
    sampling_points = parameter_info["sampling_points"]
    controller_dim = parameter_info["controller_dim"]
    robot_condition = parameter_info["robot_condition"]

    test_id = str(uuid.uuid1()).replace("-", "")
    parameters = str(controller_dim)+" "+test_id+" "+" ".join(map(str, sampling_points))

    command = "python ./test_case_service.py " + parameters
    os.system(command)
    output_file = "output_"+test_id+".txt"
    with open(output_file, "r") as file:
        results_data = file.readlines()[0].split("   ")
        results_data[0] = -1 * float(results_data[0])
        results_data[1] = -1 * (1.0 - float(results_data[1]))
    command = "rm -rf "+output_file
    os.system(command)
    return results_data[1], results_data[0]

def lhs_sampling_robot(num_samples=3, min_value=0, max_value=math.pi, controller_dim=6, robot_condition="good", objective_list=[1]):
    sample_points = lhs_sampling_core(num_samples=num_samples, dimensions=controller_dim, min_value=min_value, max_value=max_value)
    sample_points, results = robot_sampling_multiple_points(sampling_points=sample_points, controller_dim=controller_dim, robot_condition=robot_condition, objective_list=objective_list)
    return sample_points, results

def pareto_frontier(Xs, Ys, maxX = False, maxY = False):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

# def data_parser(num_samples=5, file_name="all_outputs_12d_good.txt", objective_num = 1):
def data_parser(num_samples=3, min_value=0, max_value=math.pi, controller_dim=6, robot_condition="good", objective_list=[1]):
    """
    This function parses a data base file that contains samples and outputs for different objective functions,
    and return a tuple of the list of samples and a list of lists of outputs.
    :param num_samples: If -1, it means to get the whole data in this file.
    :param file_name:
    :param objective_num:
    :return:
    """
    objective_num = objective_list[0]
    file_name = "all_outputs_" + str(controller_dim) + "d_" + robot_condition + ".txt"
    # file_name = "all_outputs_" + str(controller_dim) + "d_V2Controller_" + robot_condition + ".txt"
    samples = []
    observations = []
    try:
        with open(file_name, "r") as file:
            file_lines = file.read().splitlines()
            random.shuffle(file_lines) # This new line targets to shuffle the data.
            # This should make the sample more representative and fair.
    except ValueError:
        print "There is no such data file ..... "
        exit()

    if num_samples != -1:
        random_indices = random.sample(range(0, len(file_lines)), num_samples)
        for i in range(num_samples):
            # Get a random index
            random_index = random_indices[i]
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = map(float, input_points)
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            # output_points = map(float, output_points)[target] # Get the target we want.
            output_points = map(float, output_points)[objective_num]# Get the target we want and deduce the mean from it
            samples.append(input_points)
            observations.append(output_points)
    else:
        for i in range(len(file_lines)):
            # Get a random index
            random_index = i
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = map(float, input_points)
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            # output_points = map(float, output_points)[target] # Get the target we want.
            output_points = map(float, output_points)[objective_num]# Get the target we want and deduce the mean from it
            samples.append(input_points)
            observations.append(output_points)

    # Just cleaning - Probably unimportant operation.
    del file_lines
    return samples, [observations]

class toolbar():
    def __init__(self, toolbar_width=100):
        self.toolbar_width = toolbar_width

    def init_toolbar(self):
        sys.stdout.write("[%s]" % (" " * self.toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.toolbar_width+1)) # return to start of line, after '['

    def progress_toolbar(self):
        sys.stdout.write("-")
        sys.stdout.flush()

    def end_toolbar(self):
        sys.stdout.write("\n")

class robot_learning():
    def __init__(self, num_in_samples=5,
                 robot_condition="good",
                 dimension=2,
                 learninig_model=linear_model.LinearRegression(),
                 objectives_list=[1],
                 sampling_function=data_parser):
        """
        Shape of inputs:
        - num_in_samples: This is the number of samples used for training the learning model.
        - robot_condition: For now, we have two conditions, "good" and "damaged". Later, I will increase more categoried of damaged robots.
        - dimension: This is the controller dimensionality.
        - learninig_model: This the learning model that will be used for this problem.
        """
        # Sanity checks
        #######################
        # Initialization steps
        #######################
        # Initialize a learning model for each objective
        self.clf = []
        self.observations = []
        self.samples = []
        for i in range(len(objectives_list)):
            self.clf.append(copy.deepcopy(learninig_model))
            self.observations.append([])

        self.objectives = objectives_list
        self.dimension = dimension
        self.robot_condition = robot_condition

        self.mean_value = []
        self.model_problem_r2_score = []
        self.num_in_samples = num_in_samples
        self.model_robot_results = {}
        self.results_hash = {}
        self.sampling_function = sampling_function
    
    def get_training_data(self): # This function now samples from the files
        self.samples, self.observations = self.sampling_function(num_samples=self.num_in_samples,
                                                                 controller_dim=self.dimension, robot_condition=self.robot_condition, objective_list=self.objectives)


    def feed_training_data(self, inputs, outputs):
        """
        This method is to enable to take data from outside directly.
        """
        self.samples = inputs[:]
        self.observations[0] = outputs

    def get_observations_mean(self, observation_points):
        modified_observations = []
        for obj_obs in observation_points:
            obj_obs = np.array(obj_obs).astype(float)
            modified_observations.append(obj_obs - obj_obs.mean())
            self.mean_value.append(obj_obs.mean())
        return modified_observations

    def score_model(self, test_points, test_results):
        for objective in range(len(self.objectives)):
            modified_test_results = np.array(test_results[objective]).astype(float)
            self.model_problem_r2_score.append(self.clf[objective].score(X=test_points, y=modified_test_results))
        return self.model_problem_r2_score

    def fit_learning_model(self, message = 1):
        """
        message here is a demo variable, made only to make is easier to make parallel process.
        """
        assert (len(self.samples) > 0)
        self.observations = self.get_observations_mean(self.observations)
        for objective in range(len(self.objectives)):
            assert (len(self.observations[objective]) == len(self.samples))
            self.clf[objective].fit(np.array(self.samples).astype(float), np.array(self.observations[objective]).astype(float))

    def predict(self, test_points, eval_MSE=False): #TODO: Extend this function to multiple objectives. It works for now with a single objective.
        # predictions = [[] for i in range(len(self.objectives))]
        if eval_MSE:
            # predictions.append(np.array(self.clf[0].predict(test_points, eval_MSE=True)).astype(float))
            predictions = np.array(self.clf[0].predict(test_points, eval_MSE=eval_MSE)).astype(float)
        else:
            # predictions.append(np.array(self.clf[0].predict(test_points)).astype(float))
            predictions = np.array(self.clf[0].predict(test_points)).astype(float)

        return predictions

    def __str__(self):
        return "Robot learning problem with objectives = ", self.objectives, ", and ", self.dimension, " dimensions"

class robot_test_samples():
    def __init__(self, num_test_samples=5,
                 robot_condition="good",
                 dimension=2,
                 objectives_list=[1]):
        self.num_test_samples = num_test_samples
        self.test_points = []
        self.test_results = [[] for i in range(len(objectives_list))]
        self.objectives = objectives_list
        self.dimension = dimension
        self.robot_condition = robot_condition
        self.mean_value = []

    def get_test_samples(self):
        with open("all_outputs_"+str(self.dimension)+"d_"+self.robot_condition+".txt", "r") as file:
        # with open("all_outputs_"+str(self.dimension)+"d_V2Controller.txt", "r") as file:
        # with open("all_outputs_"+str(self.dimension)+"d_V4Controller_"+self.robot_condition+".txt", "r") as file:
            file_lines = file.readlines()
            random.shuffle(file_lines)

        random_indices = random.sample(range(0, len(file_lines)), self.num_test_samples)
        for i in range(self.num_test_samples):
            # Get a random index
            random_index = random_indices[i]
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = map(float, input_points)
            for objective in range(len(self.objectives)):
                output_points = line.split("--->")[1].replace(" ", "").split(",")
                output_points = map(float, output_points)[self.objectives[objective]]
                self.test_results[objective].append(output_points)
            self.test_points.append(input_points)
        self.test_results = self.get_observations_mean(self.test_results)

    def get_observations_mean(self, observation_points):
        modified_observations = []
        for obj_obs in observation_points:
            obj_obs = np.array(obj_obs).astype(float)
            modified_observations.append(obj_obs - obj_obs.mean())
            self.mean_value.append(obj_obs.mean())
        # print "self.mean_value = ", self.mean_value
        return modified_observations

    def return_test_pairs(self):
        return self.test_points, self.test_results

class robot_mop_problem(base): # For NSGA-2
    """
    This is the robot problem base
    """
    def __init__(self, dim=12, speed_model =robot_learning, height_model = robot_learning):
        # We call the base constructor as 'dim' dimensional problem, with 0 integer parts and 2 objectives.
        self.dim = dim
        self.start = 0.0
        self.stop = 3.14
        super(robot_mop_problem, self).__init__(dim, 0, 2) # What was that?
        self.set_bounds(self.start, self.stop)
        self.robot_learning_speed = speed_model
        self.robot_learning_height = height_model
        print "Initialization is COMPLETE :D"

    # Reimplement the virtual method that defines the objective function
    def _objfun_impl(self, x):
        robot_speed = self.robot_learning_speed.predict(x)[0] + self.robot_learning_speed.mean_value[0]
        robot_height = self.robot_learning_height.predict(x)[0] + self.robot_learning_height.mean_value[0]
        # print "robot_speed = ", robot_speed, ", robot_height = ", robot_height
        return robot_speed, robot_height

    # Add some output to __repr__
    def human_readable_extra(self):
        return "\n\tRobot test - MOP"

def fit_one_model(model):
    model.fit_learning_model()
    return model

class robot_multiple_regressors():
    """
    The idea of this is something like a bagging regression.
    Since GP has a very bad performance in higher dimensions, so I will fit many
    GPs with different random points. Then, in the prediction phase, I predict over alll
    these GPs, and I select the value for the GP with the lowest MSE for the predicted point.
    Many other policies can be explored.
    Also (this is mainly for EHVI), policies for re-fitting this complex model can be explored.
    """
    def __init__(self, number_of_estimators=10, num_in_samples=100, robot_condition="good", dimension=2, learninig_model=linear_model.LinearRegression(),objectives_list=[1]):
        self.number_of_estimators = number_of_estimators
        self.num_in_samples = num_in_samples
        self.robot_condition = robot_condition
        self.dimension = dimension
        self.learninig_model = learninig_model
        self.objectives_list = objectives_list
        self.models = []
        for i in range(self.number_of_estimators):
            new_model = robot_learning(num_in_samples=num_in_samples, robot_condition=robot_condition, dimension=dimension,
                                       learninig_model=copy.deepcopy(learninig_model), objectives_list=objectives_list)
            self.models.append(copy.deepcopy(new_model))

    def initialization(self):
        # First, get the data for the models (different data set for each model).
        input_samples = []
        output_samples = []
        file_name = "all_outputs_"+str(self.dimension)+"d_"+self.robot_condition+".txt"
        input_data, output_data = data_parser(num_samples=self.num_in_samples*self.number_of_estimators, min_value=0, max_value=math.pi,
                                                        controller_dim=self.dimension, robot_condition=self.robot_condition,
                                                        objective_list=self.objectives_list) # Todo: Expand this to handle multple objectives
        input_samples = input_data
        output_samples = output_data

        for i in range(len(self.models)):
            index_start = i * self.num_in_samples
            index_stop = ((i + 1) * self.num_in_samples)
            self.models[i].feed_training_data(inputs=input_samples[index_start:index_stop], outputs=output_samples[0][index_start:index_stop])
        print "Number of models = ", len(self.models)
        # for i in range(self.number_of_estimators):
        #     self.models[i].fit_learning_model()
        #     print i
        # print "------------------------------------------------------------------------"

    def fit(self):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
        self.models = pool.map(fit_one_model, self.models)
        # print "-------------------------------------------"

    def prediction_one_point(self, test_point):
        """
        Get the prediction from the model for one point only with the least MSE score
        :return: the result of prediction one point.
        """
        best_mse_score = 100
        best_predicition = 0
        for model in self.models:
            prediction, mse = model.predict(test_point, eval_MSE=True)
            if mse < best_mse_score:
                best_mse_score = mse
                best_predicition = prediction
        return best_predicition

    def predict(self, test_points):
        """
        This API will make the prediction for a list of points.
        :param test_points: These are the points we want to predict their values.
        :return: The predictions for test_points
        """
        predictions = []
        for point in test_points:
            predictions.append(self.prediction_one_point(test_point=point))
        return predictions

class robot_multiple_regressors_clustering():
    """
    The idea of this is something like a bagging regression.
    Since GP has a very bad performance in higher dimensions, so I will fit many
    GPs with different random points. Then, in the prediction phase, I predict over alll
    these GPs, and I select the value for the GP with the lowest MSE for the predicted point.
    Many other policies can be explored.
    Also (this is mainly for EHVI), policies for re-fitting this complex model can be explored.
    """
    def __init__(self, number_of_estimators=10, num_in_samples=100, robot_condition="good", dimension=2, learninig_model=linear_model.LinearRegression(),objectives_list=[1]):
        self.number_of_estimators = number_of_estimators
        self.num_in_samples = num_in_samples
        self.robot_condition = robot_condition
        self.dimension = dimension
        self.learninig_model = learninig_model
        self.objectives_list = objectives_list
        self.models = []
        for i in range(self.number_of_estimators):
            new_model = robot_learning(num_in_samples=num_in_samples, robot_condition=robot_condition, dimension=dimension,
                                       learninig_model=copy.deepcopy(learninig_model), objectives_list=objectives_list)
            self.models.append(copy.deepcopy(new_model))

        self.k_means = KMeans(n_clusters=self.number_of_estimators, n_jobs=-1)

    def initialization(self):
        # First, get the data for the models (different data set for each model).
        input_data, output_data = data_parser(num_samples=self.num_in_samples*self.number_of_estimators, min_value=0, max_value=math.pi,
                                                        controller_dim=self.dimension, robot_condition=self.robot_condition,
                                                        objective_list=self.objectives_list) # Todo: Expand this to handle multple objectives
        input_samples = input_data
        output_samples = output_data

        ######################################################################################
        # We need to use the clustering method instead of the commented part.
        ######################################################################################
        # for i in range(len(self.models)):
        #     index_start = i * self.num_in_samples
        #     index_stop = ((i + 1) * self.num_in_samples)
        #     self.models[i].feed_training_data(inputs=input_samples[index_start:index_stop], outputs=output_samples[0][index_start:index_stop])

        # 1. Cluster the input data into 'self.number_of_estimators' clusters
        self.k_means.fit(input_samples)
        values = self.k_means.cluster_centers_.squeeze()
        labels = self.k_means.labels_

        # 2. For each cluster, take the data points assigned to this cluster and fit it to a GP.
        for cluster in range(self.number_of_estimators):
            indices = [i for i, j in enumerate(labels) if j == cluster] # Will get the indices for all the points attached to this cluster.
            data_points_for_this_cluster = [input_samples[i] for i in indices] # This will get the input data points attached to this cluster
            output_points_for_this_cluster = [output_samples[0][i] for i in indices] # This will get the output data points attached to this cluster
            self.models[cluster].feed_training_data(inputs=data_points_for_this_cluster, outputs=output_points_for_this_cluster)
        ###########################################
        print "Clustering method - Number of models = ", len(self.models)

    def fit(self):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
        self.models = pool.map(fit_one_model, self.models)
        # print "-------------------------------------------"

    def prediction_one_point(self, test_point):
        """
        Get the prediction from the model for one point only with the least MSE score
        :return: the result of prediction one point.
        """
        # best_mse_score = 100
        # best_predicition = 0
        # for model in self.models:
        #     prediction, mse = model.predict(test_point, eval_MSE=True)
        #     if mse < best_mse_score:
        #         best_mse_score = mse
        #         best_predicition = prediction
        best_cluster = self.k_means.predict(test_point)
        best_predicition, mse = self.models[best_cluster].predict(test_point, eval_MSE=True)
        return best_predicition

    def predict(self, test_points):
        """
        This API will make the prediction for a list of points.
        :param test_points: These are the points we want to predict their values.
        :return: The predictions for test_points
        """
        predictions = []
        for point in test_points:
            predictions.append(self.prediction_one_point(test_point=point))
        return predictions

class robot_multiple_regressors_ClusterCentroids():
    """
    The idea of this is something like a bagging regression.
    Since GP has a very bad performance in higher dimensions, so I will fit many
    GPs with different random points. Then, in the prediction phase, I predict over alll
    these GPs, and I select the value for the GP with the lowest MSE for the predicted point.
    Many other policies can be explored.
    Also (this is mainly for EHVI), policies for re-fitting this complex model can be explored.
    """
    def __init__(self, num_in_samples=100, robot_condition="good", dimension=2, learninig_model=linear_model.LinearRegression(),objectives_list=[1]):
        self.num_in_samples = num_in_samples
        self.robot_condition = robot_condition
        self.dimension = dimension
        self.learninig_model = learninig_model
        self.objectives_list = objectives_list
        self.models = robot_learning(num_in_samples=num_in_samples, robot_condition=robot_condition, dimension=dimension, learninig_model=learninig_model, objectives_list=objectives_list)

        # self.k_means = KMeans(n_clusters=self.num_in_samples, n_jobs=-1, max_iter=200, precompute_distances=True)
        self.k_means = KMeans(n_clusters=self.num_in_samples, n_jobs=-1, max_iter=30, precompute_distances=True)

    def initialization(self):
        # First, get the data for the models (different data set for each model).
        input_samples = []
        output_samples = []
        file_name = "all_outputs_"+str(self.dimension)+"d_"+self.robot_condition+".txt"

        input_data, output_data = data_parser(num_samples=self.num_in_samples*10, min_value=0, max_value=math.pi,
                                                        controller_dim=self.dimension, robot_condition=self.robot_condition,
                                                        objective_list=self.objectives_list) # Todo: Expand this to handle multple objectives # Todo: Expand this to handle multiple objectives
        input_samples = input_data
        output_samples = output_data

        ######################################################################################
        # In this method, I will use the centroids of the clusters as my training data.
        # I need to get the outputs for this centroids, and use this in order to train the gaussian process.
        ######################################################################################
        # 1. Cluster the input data into 'self.num_in_samples' clusters
        self.k_means.fit(input_samples)
        values = self.k_means.cluster_centers_.squeeze()
        labels = self.k_means.labels_

        # 2. For each centroid of each cluster, we need to evaluate it first on the robot to get its actual objective optimization values. -- Really Super shit!
        # Another idea is to select the nearest point to
        data_points_for_this_cluster = []
        output_points_for_this_cluster = []
        for cluster_number in range(self.num_in_samples): # For each cluster centroids, try to find th nearest point for this centroid from the data
            current_cluster_indices = [i for i, j in enumerate(labels) if j == cluster_number]
            current_cluster_points = [input_samples[i] for i in current_cluster_indices]
            # current_cluster_points.append(values[cluster_number])
            nn_classifier = NearestNeighbors(n_neighbors=2)
            nn_classifier.fit(current_cluster_points)
            nearest_point_index = nn_classifier.kneighbors(values[cluster_number], return_distance=False, n_neighbors=1)[0][0]
            data_points_for_this_cluster.append(current_cluster_points[nearest_point_index])

        for point in data_points_for_this_cluster:
            for input_point_index in range(len(input_samples)):
                if input_samples[input_point_index] == point:
                    output_points_for_this_cluster.append(output_samples[0][input_point_index])
                    break

        self.models.feed_training_data(inputs=data_points_for_this_cluster, outputs=output_points_for_this_cluster)
        # print "data_points_for_this_cluster = ", data_points_for_this_cluster
        # print "output_points_for_this_cluster = ", output_points_for_this_cluster
        ###########################################
        print "Centroids of Clusters method"

    def fit(self):
        self.models.fit_learning_model()

    def predict(self, test_points):
        """
        This API will make the prediction for a list of points.
        :param test_points: These are the points we want to predict their values.
        :return: The predictions for test_points
        """
        # predictions = []
        # predictions.append(self.models.predict(test_points, eval_MSE=True))
        return self.models.predict(test_points, eval_MSE=False)
# Quick testing.
# TODO: In this test, working with more than one objective doesn't work properly. One objective only is OK. I need to check this issue.
# gp = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
# test_method = robot_test_samples(num_test_samples=10, objectives_list=[0, 1])
# test_method.get_test_samples()
# test_points, test_results = test_method.return_test_pairs()
# print "Test points "
# print test_points
# print "Test results "
# for i in test_results:
#     print i
#     print "----------------------------------------------------------"
# robot_problem = robot_learning(num_in_samples=10, objectives_list=[0, 1], learninig_model=copy.deepcopy(gp))
# robot_problem.get_training_data()
# robot_problem.fit_learning_model()
# print robot_problem.score_model(test_points=test_points, test_results=test_results)