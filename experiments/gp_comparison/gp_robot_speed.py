__author__ = 'omohamme'

__author__ = 'omohamme'
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from multilayer_perceptron import MultilayerPerceptronRegressor
from sklearn import svm
from sklearn.svm import NuSVR
from sklearn import linear_model
from sklearn import svm
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

toolbar_width = 100


# np.random.seed(1)
start = 0.0
stop = 3.14
dimension = 2
# dimension = 1
def get_observation_mean(observation_num=1):
    global dimension
    with open("all_outputs_"+str(dimension)+"d_good.txt", "r") as file:
            file_lines = file.readlines()

    output_list = []
    for line in file_lines:
        output_points = line.split("--->")[1].replace(" ", "").split(",")
        output_list.append(map(float, output_points)[observation_num]) # Get the target we want.

    observed_mean = sum(output_list) / len(output_list)
    # print observed_mean
    # exit()
    return observed_mean


def uniform_random_sampling():
    global start, stop, dimension
    results = []
    for i in range(dimension):
        random_number = start + (random.random() * (stop - start))
        results.append(str(random_number))
    return results

def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

def sim_calling(parameters):
    generated_parameters = parameters[:]
    test_id = id_generator()
    generated_parameters.insert(0, test_id)
    command = "python /home/omohamme/INRIA/experiments/gp_comparison/test_case_service.py " + " ".join(generated_parameters)
    os.system(command)
    output_file = "output_"+test_id+".txt"
    with open(output_file, "r") as file:
        results_data = file.readlines()[0].split("   ")
        results_data[0] = -1.0 * float(results_data[0])
        results_data[1] = -1.0 * (1.0 - float(results_data[1]))
    command = "rm -rf "+output_file
    os.system(command)
    with open("all_outputs_"+str(dimension)+"d.txt", "a") as file:
        print >> file, ",".join(parameters), "--->",results_data[0], ",", results_data[1]
    return results_data

def robot_problem(parameters):
    used_parameters = parameters[:].split(",")
    results = sim_calling(parameters=used_parameters)
    return parameters, results[1]

def generate_random_test_samples(number_of_test_samples, sampling_function):
    test_points = []
    for i in range(number_of_test_samples):
        # generated_parameters = uniform_random_sampling()
        generated_parameters = sampling_function()
        generated_parameters = np.array(map(float, generated_parameters))
        test_points.append(generated_parameters)
    return test_points
# ----------------------------------------------------------------------
class robot_speed_gp():
    def __init__(self, num_in_samples=2, sampling_function=uniform_random_sampling):
        self.gp = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
        self.results_hash = {}
        self.observations = []
        self.samples = []
        self.num_in_samples = num_in_samples
        self.sampling_function = sampling_function
        self.model_robot_results = {}
        self.generateX_AND_getY_for_training()

    def generateX_AND_getY_for_training(self):
        for i in range(self.num_in_samples):
            # generated_parameters = uniform_random_sampling()
            generated_parameters = self.sampling_function()
            self.results_hash[",".join(generated_parameters)] = 0

        # Run the tests on the simulator in parallel
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
        self.results_hash = dict(pool.map(robot_problem, self.results_hash.keys()))

        for item in self.results_hash:
            self.samples.append(np.array(map(float, item.split(","))))
            self.observations.append(self.results_hash[item])

    def fit_gp(self):
        assert (len(self.samples) > 0)
        assert (len(self.observations) > 0)
        self.gp.fit(self.samples, self.observations)

    def test_gp(self, test_samples):
        y_pred, MSE = self.gp.predict(test_samples, eval_MSE=True)
        sigma = np.sqrt(MSE)
        for i in range(len(test_samples)):
            self.model_robot_results[",".join(map(str, test_samples[i]))] = y_pred[i]
        return self.model_robot_results


class robot_speed_testing():
    def __init__(self, num_test_samples=50, sampling_function=uniform_random_sampling):
        self.num_test_samples = num_test_samples
        self.sampling_function = sampling_function
        self.test_points = []
        self.real_robot_results = {}

        self.generate_test_samples()
        self.get_real_values_from_robot()

    def generate_test_samples(self):
        for i in range(self.num_test_samples):
            generated_parameters = self.sampling_function()
            generated_parameters = np.array(map(float, generated_parameters))
            self.test_points.append(generated_parameters)
        return self.test_points

    def return_test_points(self):
        return self.test_points

    def get_real_values_from_robot(self):
        for i in self.test_points:
            self.real_robot_results[",".join(map(str, i))] = 0
        # Run the tests on the simulator in parallel
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
        self.real_robot_results = dict(pool.map(robot_problem, self.real_robot_results.keys()))
    
    def compute_prediction_error(self, model_results):
        prediction_error = 0
        for test_input in model_results:
            prediction_error += (model_results[test_input] - self.real_robot_results[test_input])**2
        return prediction_error / self.num_test_samples

class robot_speed_testing_excitingData(robot_speed_testing):
    def __init__(self, num_test_samples=50, objective_fun=1, mean_value = 0.0):
        self.mean_value = mean_value
        self.num_test_samples = num_test_samples
        self.test_pairs = []
        self.test_points = []
        self.test_points_results = []
        self.real_robot_results = {}
        self.target = objective_fun
        self.generate_test_samples()

    def generate_test_samples(self):
        global dimension
        with open("all_outputs_"+str(dimension)+"d_good.txt", "r") as file:
        # with open("pareto_clean.txt", "r") as file:
            file_lines = file.readlines()
        random_indices = random.sample(range(0, len(file_lines)), self.num_test_samples)
        X = []
        y = []
        for i in range(self.num_test_samples):
            # Get a random index
            random_index = random_indices[i]
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = np.array(map(float, input_points))
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            # output_points = map(float, output_points)[self.target] # Get the target we want.
            output_points = map(float, output_points)[self.target] - self.mean_value# Get the target we want and deduce the mean from it
            self.test_points.append(input_points)
            self.real_robot_results[",".join(map(str, input_points))] = output_points
            self.test_points_results.append(output_points)
            X.append(input_points)
            y.append(output_points)
        self.test_pairs = [X, y]

    def return_test_pairs(self):
        return self.test_pairs

class robot_speed_gp_excitingData(robot_speed_gp):
    def __init__(self, num_in_samples=5, kernel_type='squared_exponential', objective_fun=1, mean_value=0.0):
        # self.gp = GaussianProcess(corr=kernel_type, theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget = 3.00e-13)
        self.gp = svm.SVR()
        # self.gp = GaussianProcess(corr=kernel_type, theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
        self.mean_value = mean_value
        self.model_problem_r2_score = 0
        self.results_hash = {}
        self.target = objective_fun
        self.observations = []
        self.samples = []
        self.num_in_samples = num_in_samples
        self.model_robot_results = {}
        self.generateX_AND_getY_for_training()

    def generateX_AND_getY_for_training(self): # This function now samples from the files
        global dimension
        with open("all_outputs_"+str(dimension)+"d_good.txt", "r") as file:
            file_lines = file.readlines()

        random_indices = random.sample(range(0, len(file_lines)), self.num_in_samples)
        for i in range(self.num_in_samples):
            # Get a random index
            random_index = random_indices[i]
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = np.array(map(float, input_points))
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            # output_points = map(float, output_points)[self.target] # Get the target we want.
            output_points = map(float, output_points)[self.target] - self.mean_value # Get the target we want and deduce the mean from it
            self.samples.append(input_points)
            self.observations.append(output_points)
        # print "self.samples = ", self.samples
        # print "self.observations = ", self.observations
        # exit()

    def test_gp(self, test_pairs):
        self.model_problem_r2_score = self.gp.score(X=test_pairs[0], y=test_pairs[1])
        # print "GP model score = ", self.model_problem_r2_score
        return self.model_problem_r2_score

    def fit_gp(self):
        assert (len(self.samples) > 0)
        assert (len(self.observations) > 0)
        self.gp.fit(self.samples, self.observations)

    def __str__(self):
        return "robot_speed_gp_excitingData"



class robot_speed_gp_excitingData_priors(robot_speed_gp_excitingData):
    def __init__(self, num_in_samples=500, kernel_type='squared_exponential', objective_fun=1, mean_value=0.0):
        robot_speed_gp_excitingData.__init__(self, num_in_samples=num_in_samples, kernel_type=kernel_type, objective_fun=objective_fun, mean_value=mean_value)
    def generateX_AND_getY_for_training(self): # This function now samples from the files
        with open("pareto_clean.txt", "r") as file:
            file_lines = file.readlines()
            pareto_data_len = len(file_lines)

        for i in range(pareto_data_len):
            # Get a random index
            random_index = i
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = np.array(map(float, input_points))
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            output_points = map(float, output_points)[self.target] # Get the target we want.
            # output_points = map(float, output_points)[self.target] - self.mean_value # Get the target we want and deduce the mean from it
            self.sanity_check(input_points, output_points + self.mean_value)
            self.samples.append(input_points)
            self.observations.append(output_points)

        with open("all_outputs_12d_good.txt", "r") as file:
            file_lines = file.readlines()

        random_indices = random.sample(range(0, len(file_lines)), self.num_in_samples - pareto_data_len)
        for i in range(self.num_in_samples - pareto_data_len):
            # Get a random index
            random_index = random_indices[i]
            # Get the values
            line = file_lines[random_index]
            input_points = line.split("--->")[0].split(",")
            input_points = np.array(map(float, input_points))
            output_points = line.split("--->")[1].replace(" ", "").split(",")
            output_points = map(float, output_points)[self.target] # Get the target we want.
            # output_points = map(float, output_points)[self.target] - self.mean_value # Get the target we want and deduce the mean from it
            self.samples.append(input_points)
            self.observations.append(output_points)

        with open("meta_data.txt","w") as file:
            for item in self.observations:
                print >> file, item

    def sanity_check(self, sample, observation):
        for point in sample:
            assert (point >= 0.0) and (point <= 3.14)
        assert (observation < 0)


def experiment(parameters_list):
    problem_object = parameters_list[0]
    test_object = parameters_list[1]
    problem_object.fit_gp()
    # model_results = problem_object.test_gp(test_samples=test_object.return_test_points())
    model_results = problem_object.test_gp(test_pairs=test_object.return_test_pairs())
    # print model_results
    del problem_object
    del test_object
    sys.stdout.write("-")
    sys.stdout.flush()
    return model_results


def main_func():
    global dimension
    table_of_models = {"GP": GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100),
                      "GLM": linear_model.LinearRegression(),
                      "SVM": svm.SVR()}
    table_of_robot_data = {"NewData": robot_speed_gp,
                           "ExistingData": robot_speed_gp_excitingData}
    gp_kernels = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']
    dimensions_list = [12]
    load_from_file = False
    models_list = ["GP"]
    t0 = time.time()
    for dim in dimensions_list:
        print "Current Dimension = ", dim
        dimension = dim
        if not load_from_file:
            observed_mean = get_observation_mean(observation_num=1)
            test_methods = robot_speed_testing_excitingData(num_test_samples=1000, mean_value=observed_mean)
            # samples_numbers_range = [10, 25, 50, 100, 200, 500]
            samples_numbers_range = [5000, 10000]
            final_prediction_values = {}

            for i in samples_numbers_range: # Fit the models with different points. See how the accuracy changes
                print "current sample = ", i
                inputs_list = []
                for j in range(100):
                    problem = robot_speed_gp_excitingData(num_in_samples=i, mean_value=observed_mean)
                    # problem = robot_speed_gp_excitingData_priors(num_in_samples=i, mean_value=observed_mean)
                    inputs_list.append((copy.deepcopy(problem), copy.deepcopy(test_methods)))
                    del problem

                global toolbar_width
                sys.stdout.write("[%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
                prediction_results = pool.map(experiment, inputs_list)
                # prediction_results = map(experiment, inputs_list)

                sys.stdout.write("\n")

                final_prediction_values[i] = prediction_results

            with open("result_RobotSpeed_"+str(dimension)+"D.json", 'w') as fp:
                json.dump(final_prediction_values, fp)
        else:
            with open("result_RobotSpeed_"+str(dimension)+"D.json", 'r') as fp:
                final_prediction_values = json.loads(fp.readline())

        sorted_keys = map(int, final_prediction_values.keys())
        sorted_keys.sort()
        print sorted_keys
        print final_prediction_values.keys()
        sorted_values = []
        for i in sorted_keys:
            sorted_values.append(final_prediction_values[str(i)])
        current_path = os.getcwd()
        plt.figure(figsize=(15.0, 11.0))
        # plt.boxplot(final_prediction_values.values())
        # plt.xticks(range(1, len(final_prediction_values.keys()) + 1), final_prediction_values.keys().sort(), rotation=45, fontsize=20)
        plt.boxplot(sorted_values)
        plt.xticks(range(1, len(final_prediction_values.keys()) + 1), sorted_keys, rotation=45, fontsize=20)
        plt.yticks(fontsize=15)
        plt.ylim([-2, 1])
        plt.ylabel("R2 Score", fontsize=20)
        plt.xlabel("Number of sampling points", fontsize=20)
        plt.title("results_RobotSpeed_"+str(dimension)+"D.png")
        plt.savefig("results_RobotSpeed_"+str(dimension)+"D.png")
        os.chdir(current_path)

        stat_file = open("Stat_tests_RobotSpeed_"+str(dimension)+"D.txt", "w")
        seen_pairs = []
        for number_of_samples in final_prediction_values:
            for number_of_samples2 in final_prediction_values:
                if (number_of_samples != number_of_samples2) and ((number_of_samples, number_of_samples2) not in seen_pairs):
                    seen_pairs.append((number_of_samples, number_of_samples2))
                    seen_pairs.append((number_of_samples2, number_of_samples))
                    statistical_significance = stats.wilcoxon(final_prediction_values[number_of_samples], final_prediction_values[number_of_samples2])
                    print >> stat_file, number_of_samples, " VS ", number_of_samples2, " -->", statistical_significance
                    print >> stat_file, number_of_samples, " median = ", np.median(final_prediction_values[number_of_samples])
                    print >> stat_file, number_of_samples2, " median = ", np.median(final_prediction_values[number_of_samples2])
                    print >> stat_file, "----------------------------------------------------------"

        # print prediction_results
    t1 = time.time() - t0
    print "Total regression time = ", t1

main_func()