__author__ = 'omohamme'
__author__ = 'omohamme'

__author__ = 'omohamme'
from gp_robot_speed import *

# np.random.seed(1)
start = 0.0
stop = 3.14
dimension = 12
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
    with open("all_outputs_12d.txt", "a") as file:
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

class robot_heightVAR_testing_excitingData(robot_speed_gp_excitingData):
    def __init__(self, num_test_samples=50, objective_fun=0):
        robot_speed_gp_excitingData.__init__(num_in_samples=num_test_samples, objective_fun=objective_fun)


class robot_heightVAR_gp_excitingData(robot_speed_testing_excitingData):
    def __init__(self, num_test_samples=5, objective_fun=0):
        robot_speed_testing_excitingData.__init__(num_test_samples=num_test_samples, objective_fun=objective_fun)

load_from_file = False
if load_from_file:

    t0 = time.time()
    test_methods = robot_speed_testing_excitingData(num_test_samples=160)
    prediction_results = []
    samples_numbers_range = [10, 100, 500, 1000]
    final_prediction_values = {}
    gp_kernels = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']

    for i in samples_numbers_range: # Fit the models with different points. See how the accuracy changes
        prediction_results = []
        print "current sample = ", i
        for j in range(30): # We need to parallize this for loop. With its current form, it is very inefficient.
            print "ITER = ", j
            # problem = robot_speed_gp(num_in_samples=i, sampling_function=uniform_random_sampling)
            problem = robot_speed_gp_excitingData(num_in_samples=i)
            problem.fit_gp()
            model_results = problem.test_gp(test_samples=test_methods.return_test_points())
            prediction_results.append(test_methods.compute_prediction_error(model_results))
            del problem
        final_prediction_values[i] = prediction_results

    with open('result.json', 'w') as fp:
        json.dump(final_prediction_values, fp)
else:
    with open('result.json', 'r') as fp:
        final_prediction_values = json.loads(fp.readlines())

current_path = os.getcwd()
plt.figure(figsize=(15.0, 11.0))
plt.boxplot(final_prediction_values.values())
plt.xticks(range(1, len(final_prediction_values.keys()) + 1), final_prediction_values.keys(), rotation=45, fontsize=20)
plt.yticks(fontsize=15)
plt.ylabel("Prediction Error", fontsize=20)
plt.xlabel("Number of sampling points", fontsize=20)
plt.savefig("results.png")
os.chdir(current_path)

print prediction_results
t1 = time.time() - t0
print "Total regression time = ", t1