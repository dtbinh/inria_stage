__author__ = 'omohamme'

import numpy as np
from utilities import *

os.system("reset")
# np.random.seed(1)
start = 0.0
stop = 3.14
dimension = 2
robot_condition = "good"
def lhs_random_sampling():
    global start, stop, dimension
    min_value = start
    max_value = stop
    num_dim = dimension
    num_bins = 12
    # The theory is to divide the sampling space (max-min) into "num_points" periods. all with the same length
    # Then from each period, use uniform distribution to select a point from it.
    random_points = []

    # Generate the random samples
    for dim in range(num_dim):
        step = float(min_value + max_value) / num_bins
        chose_random_interval = random.randint(0, num_bins - 1)
        start_point = chose_random_interval * step
        end_point = start_point + step
        random_point = random.uniform(start_point, end_point)
        assert random_point <= 3.14
        assert random_point >= 0.0
        random_points.append(random_point)

    return map(str,random_points)

# def uniform_random_sampling():
def uniform_random_sampling(num_samples=3, dimensions=2, min_value=0, max_value=math.pi):
    start = min_value
    stop = max_value
    dimension = dimensions
    all_samples = []
    for sample in range(num_samples):
        results = []
        for i in range(dimension):
            random_number = start + (random.random() * (stop - start))
            results.append(str(random_number))
        all_samples.append(results)
    return all_samples

def id_generator(size=12, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

def sim_calling(parameters):
    global dimension
    global robot_condition
    generated_parameters = parameters[:]
    test_id = id_generator()
    generated_parameters.insert(0, test_id)
    # print generated_parameters
    generated_parameters.insert(0, str(dimension))
    command = "python ./test_case_service.py " + " ".join(generated_parameters)
    os.system(command)
    output_file = "output_"+test_id+".txt"
    print "OUTPUT FILE NAME = ", output_file
    with open(output_file, "r") as file:
        results_data = file.readlines()[0].split("   ")
        results_data[0] = -1.0 * float(results_data[0])
        results_data[1] = -1.0 * (1.0 - float(results_data[1]))
    command = "rm -rf "+output_file
    os.system(command)
    with open("all_outputs_"+str(dimension)+"d_ONLYDAMAGE2.txt", "a") as file:
        print >> file, ",".join(parameters), "--->", results_data[0], ",", results_data[1]
        print ",".join(parameters), "--->", results_data[0], ",", results_data[1]
    return results_data

def robot_problem(parameters):
    used_parameters = parameters[:].split(",")
    results = sim_calling(parameters=used_parameters)
    return parameters, results[1]


class robot_speed_gp():
    def __init__(self, num_in_samples=2, sampling_function=uniform_random_sampling, dimensions=2):
        self.results_hash = {}
        self.observations = []
        self.samples = []
        self.num_in_samples = num_in_samples
        self.dimensions = dimensions
        self.sampling_function = sampling_function
        self.model_robot_results = {}
        self.generateX_AND_getY_for_training()

    def generateX_AND_getY_for_training(self):
        # generated_parameters = self.sampling_function()
        generated_parameters = self.sampling_function(num_samples=self.num_in_samples, dimensions=self.dimensions,
                                                      min_value=0, max_value=math.pi)
        for sample in generated_parameters:
            string_sample = map(str, sample)
            self.results_hash[",".join(string_sample)] = 0

        # Run the tests on the simulator in parallel
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
        # pool = multiprocessing.Pool(processes=1) # This will use the max number of processes available
        pool.map(robot_problem, self.results_hash.keys())


t0 = time.time()
prediction_results = []
# for i in range(2, 7):
for i in [6]:
    samples_numbers = 10000
    dimension = i
    # for i in range(20):
    # print len(lhs_random_sampling())
    problem = robot_speed_gp(num_in_samples=samples_numbers, sampling_function=uniform_random_sampling, dimensions=dimension)
    del problem

t1 = time.time() - t0
print "Total regression time = ", t1