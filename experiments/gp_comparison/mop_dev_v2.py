__author__ = 'omohamme'
###################################################################
# The target here is to test a new approach to optimization for my current pro
#
#
#
#
###################################################################
from utilities import *
clear_screen()
###################################################################
# General parameters
dimensions = 6
speed_model = DecisionTreeRegressor(max_depth=20)
height_model = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget=3.00e-13)
num_robot_evaluations = 100
robot_condition = "good"
###################################################################
robot_learning_speed_samples = []
robot_learning_speed_observations = []
robot_learning_height_samples = []
robot_learning_height_observations = []
robot_learning_speed = robot_learning(num_in_samples=3, objectives_list=[1], learninig_model=copy.deepcopy(speed_model), robot_condition="good", dimension=dimensions)
robot_learning_height = robot_learning(num_in_samples=3, objectives_list=[0], learninig_model=copy.deepcopy(speed_model), robot_condition="good", dimension=dimensions)
robot_learning_speed.get_training_data()

robot_learning_speed_samples += robot_learning_speed.samples
robot_learning_speed_observations += robot_learning_speed.observations
robot_learning_speed.fit_learning_model()

robot_learning_height.get_training_data()
robot_learning_height_samples += robot_learning_height.samples

robot_learning_height_observations += robot_learning_height.observations
robot_learning_height.fit_learning_model()

# exit()
for i in range(3):
    # Evaluate how good the models are
    test_method = robot_test_samples(num_test_samples=1000, objectives_list=[1], robot_condition="good", dimension=dimensions)
    test_method.get_test_samples()
    test_points, test_results = test_method.return_test_pairs()
    model = robot_learning_speed.predict(test_points)
    print "speed model r2 score = ", repr(r2_score(test_results[0], model))

    test_method = robot_test_samples(num_test_samples=1000, objectives_list=[0], robot_condition="good", dimension=dimensions)
    test_method.get_test_samples()
    test_points, test_results = test_method.return_test_pairs()
    model = robot_learning_height.predict(test_points)
    print "height model r2 score = ", repr(r2_score(test_results[0], model))

    prob = robot_mop_problem(dim=dimensions, speed_model=robot_learning_speed, height_model=robot_learning_height)
    algo = algorithm.nsga_II(gen=1)
    # algo = algorithm.cmaes(gen=1) # When I try CMA-ES, it tells me that this algorithm is for single objective optimization only! Interesting
    pop = population(prob, 8)
    pop = algo.evolve(pop)

    current_points = [list(ind.cur_x) for ind in pop]

    theoretical_results = [list(ind.cur_f) for ind in pop]

    # Calculation based on the theoretical results
    Xs = [-1*x[0] for x in theoretical_results]
    Ys = [-1*x[1] for x in theoretical_results]
    clean_x, clean_y = pareto_frontier(Xs, Ys)
    clean_data = []
    for i in range(len(clean_x)):
        clean_data.append((clean_x[i], clean_y[i]))

    ref_point = (2, 2)
    hv = util.hypervolume(clean_data)
    print "theoretical_results - Hypervolume = ", hv.compute(r=ref_point)

    # Calculation based on the actual results
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
    process_list = []
    parameters_list = []
    for sample in current_points:
        data_structure = dict(controller_dim=dimensions,
                              sampling_points=sample,
                              robot_condition=robot_condition)
        parameters_list.append(data_structure)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) # This will use the max number of processes available
    actual_results = pool.map(robot_sampling_one_point, parameters_list)
    Xs = [-1*x[0] for x in actual_results]
    Ys = [-1*x[1] for x in actual_results]
    clean_x, clean_y = pareto_frontier(Xs, Ys)
    clean_data = []
    for i in range(len(clean_x)):
        clean_data.append((clean_x[i], clean_y[i]))

    ref_point = (2, 2)
    hv = util.hypervolume(clean_data)
    print "actual_results - Hypervolume = ", hv.compute(r=ref_point)

    robot_learning_speed_samples += current_points
    robot_learning_speed_observations[0] += Xs
    robot_learning_height_samples += current_points
    robot_learning_height_observations[0] += Ys

    robot_learning_speed = robot_learning(num_in_samples=18000, objectives_list=[1], learninig_model=copy.deepcopy(speed_model), robot_condition="good", dimension=dimensions)
    robot_learning_height = robot_learning(num_in_samples=100, objectives_list=[0], learninig_model=copy.deepcopy(height_model), robot_condition="good", dimension=dimensions)

    robot_learning_speed.feed_training_data(inputs=robot_learning_speed_samples, outputs=robot_learning_speed_observations[0])
    robot_learning_speed.fit_learning_model()

    robot_learning_height.feed_training_data(inputs=robot_learning_height_samples, outputs=robot_learning_height_observations[0])
    robot_learning_height.fit_learning_model()