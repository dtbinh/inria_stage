__author__ = 'omohamme'
from PyGMO.problem import base
import math
from sklearn.gaussian_process import GaussianProcess
from PyGMO import *
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import os

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
    with open("all_outputs_12d.txt", "a") as file:
        print >> file, ",".join(parameters), "--->",results_data[0], ",", results_data[1]
    return results_data

def simulator_caller(parameters):
    used_parameters = map(str, parameters[:])
    results = sim_calling(parameters=used_parameters)
    return results[0], results[1]

class robot_problem(base): # For NSGA-2
    """
    This is the robot problem base
    """

    def __init__(self, dim=12):
        # We call the base constructor as 'dim' dimensional problem, with 0 integer parts and 2 objectives.
        self.dim = dim
        self.start = 0.0
        self.stop = 3.14
        super(robot_problem, self).__init__(dim, 0, 2)
        self.set_bounds(self.start, self.stop)
        self.id_size = 6

    # Reimplement the virtual method that defines the objective function
    def _objfun_impl(self, x):
        results = self.sim_calling(parameters=x)
        return results[0], results[1]

    # Add some output to __repr__
    def human_readable_extra(self):
        return "\n\tMulti-Objective problem"

    def id_generator(self):
        chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(self.id_size))

    def sim_calling(self, parameters):
        generated_parameters = map(str, parameters[:])
        test_id = self.id_generator()
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
        # with open("all_outputs_12d.txt", "a") as file:
        #     print >> file, ",".join(parameters), "--->", results_data[0], ",", results_data[1]
        return results_data

class robot_problem_gp_init():
    """
    This is another attempt to crack this problem. What I want to do here is to fit 2 GP with:
    -- All the NSGA-2 priors we have.
    -- Random points.
    In total, this is 500 points.
    Then, use NSGA-2 extensively to optimize the GP to get their PF.
    Then, evaluate this PF on the robot simulator.
    Then, use the sampled points to fit them to the GP again.
    Repeat the previous steps for a the number of evaluation points we have.
    Calculate the hypervolume.
    """
    def __init__(self, number_of_objective=2):
        assert number_of_objective > 1
        self.num_in_samples = 200
        self.gp = []
        self.number_of_objective = number_of_objective
        for i in range(number_of_objective):
            self.gp.append(GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4,
                                           thetaU=1e-1, random_start=100))
        self.model_problem_r2_score_linear_model = 0.0
        self.mean_value = 0.0
        self.model_problem_r2_score = 0
        self.results_hash = {}
        self.observations = [[] for i in range(self.number_of_objective)]
        self.samples = [[] for i in range(self.number_of_objective)]
        self.model_robot_results = {}
        # self.initialization()

    def initialization(self):
        print "GP initialization process"
        for objective in range(self.number_of_objective):
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
                # output_points = map(float, output_points)[objective] - self.mean_value # Get the target we want and deduce the mean from it
                output_points = map(float, output_points)[objective]
                self.samples[objective].append(input_points)
                self.observations[objective].append(output_points)

            with open("all_outputs_12d.txt", "r") as file:
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
                # output_points = map(float, output_points)[objective] - self.mean_value # Get the target we want and deduce the mean from it
                output_points = map(float, output_points)[objective] # Get the target we want and deduce the mean from it
                self.samples[objective].append(input_points)
                self.observations[objective].append(output_points)

            self.gp[objective].fit(self.samples[objective], self.observations[objective])
        return self.gp

    def fit_new_points(self, samples, observations):
        print "Fitting new points to the GP"
        # points is a list of lists
        for objective in range(self.number_of_objective):
            self.samples[objective].append(list(samples))
            self.observations[objective].append(observations[objective])
            self.gp[objective].fit(self.samples[objective], self.observations[objective])

    def return_gp(self):
        return self.gp

class robot_problem_gp (base):
    def __init__(self, dim=12, gaussian_processes = []):
        # We call the base constructor as 'dim' dimensional problem, with 0 integer parts and 2 objectives.
        super(robot_problem_gp, self).__init__(dim, 0, 2)
        self.set_bounds(0.0, 3.14)
        self.gp = gaussian_processes

    # Re-implement the virtual method that defines the objective function
    def _objfun_impl(self, x):
        final_results = []
        for gp in self.gp:
            final_results.append(float(gp.predict(x)))
        return tuple(final_results)

    # Add some output to __repr__
    def human_readable_extra(self):
        return "\n\tRobot problem"

num_of_iterations = 1
gp_init = robot_problem_gp_init()
gp_list = gp_init.initialization()

for i in range(num_of_iterations):
    "Current iterations = ", i
    observations_list = []
    prob = robot_problem_gp(gaussian_processes = gp_list)
    algo = algorithm.nsga_II(gen=500, cr=0.95, eta_c=10, m=0.01, eta_m=50)
    pop = population(prob, 200)
    pop = algo.evolve(pop)
    pf = pop.compute_pareto_fronts()
    total = []
    for i in pf:
        total += i

    pf_pop = [pop[i] for i in total]
    print "Number of pareto points = ", len(pf_pop)
    new_samples = [ind.cur_x for ind in pf_pop]
    exit()
    print "Fitting new points"
    for sample in new_samples:
        new_observations = simulator_caller(sample)
        gp_init.fit_new_points(samples=sample, observations=new_observations)
        observations_list.append(new_observations)
    gp_list = gp_init.return_gp()

    # hv = util.hypervolume(clean_data)
    # ref_point = (2, 2) # x is the 1-speed, y is the variance in height
    # x = str(log_file.split("/")[-1]) + ":::" + str(hv.compute(r=ref_point)) + "\n"
    # hypervolume_report.write(x)

print np.array(observations_list).T
# prob = robot_problem()
# algo = algorithm.nsga_II(gen=2, cr=0.95, eta_c=10, m=0.01, eta_m=50)  # 2000 generations of SMS-EMOA should solve it
# pop = population(prob, 8)
# pop = algo.evolve(pop)

# isl = island(algo, pop)
# isl.evolve(1)
# archi = archipelago(algo,prob,8,20)
# print min([isl.population.champion.f for isl in archi])
# archi.evolve()
# print min([isl.population.champion.f for isl in archi])
#
# F = np.array([ind.cur_f for ind in pop]).T
F = np.array(observations_list).T
plt.scatter(-1*F[1], -1*F[0])
plt.xlabel("$f^{(1)}$")
plt.ylabel("$f^{(2)}$")
plt.show()